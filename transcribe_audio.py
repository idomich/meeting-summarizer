import argparse
import logging
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import psutil
import whisper
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_MODEL_SIZE = "large-v3-turbo"
DEFAULT_DETECTOR_SIZE = "base"
MIN_LANGUAGE_CONFIDENCE = 0.8

# GPT model configuration
GPT_MODEL_NAME = "gpt-5"
GPT_MODEL_DISPLAY_NAME = "GPT-5 (high)"

# Video file extensions that require conversion
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}

# Chunking configuration for streaming transcription
MIN_FILE_SIZE_FOR_CHUNKING_MB = 10  # Minimum file size to trigger chunking (MB)
CHUNK_OVERLAP_SECONDS = 2  # Overlap between chunks to avoid missing words

# Determine optimal number of parallel processes based on CPU cores
# Cap at 8 to avoid excessive memory usage (each process loads a ~3-5GB Whisper model)
_CPU_CORES = multiprocessing.cpu_count()
MAX_PARALLEL_CHUNKS = min(_CPU_CORES, 8) if _CPU_CORES > 1 else 1

# Setup logging with multiprocessing support
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Force reconfigure for multiprocessing
)
logger = logging.getLogger(__name__)


def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)


def estimate_model_memory_usage(model_size: str) -> float:
    """
    Estimate memory usage of a Whisper model without loading it

    Args:
        model_size: Whisper model size name

    Returns:
        Estimated memory usage in GB
    """
    # Known model specifications based on actual file sizes (MB)
    # Memory usage is approximately 1.3x the model file size (model weights + computation overhead)
    model_specs = {
        "tiny": 72.1,
        "tiny.en": 72.1,
        "base": 138.5,
        "base.en": 138.5,
        "small": 461.2,
        "small.en": 461.2,
        "medium": 1457.2,
        "medium.en": 1457.2,
        "large": 2944.3,  # Default to large-v3
        "large-v1": 2944.3,
        "large-v2": 160.3,  # Compressed version
        "large-v3": 2944.3,
        "large-v3-turbo": 1543.0,
        "turbo": 1543.0,  # Alias for large-v3-turbo
    }

    file_size_mb = model_specs.get(model_size, 2944.3)  # Default to large-v3 for unknown models
    # Memory usage is approximately 1.3x the model file size (model weights + computation overhead)
    estimated_memory_gb = (file_size_mb * 1.3) / 1024

    logger.info(
        f"Model {model_size}: {file_size_mb:.1f}MB file → "
        f"{estimated_memory_gb:.2f}GB estimated memory"
    )
    return estimated_memory_gb


def calculate_safe_process_count(max_requested: int, model_size: str) -> int:
    """
    Calculate a safe number of processes based on available memory and estimated model usage

    Args:
        max_requested: Maximum number of processes requested
        model_size: Whisper model size to estimate memory for

    Returns:
        Safe number of processes that won't exceed available memory
    """

    # Estimate memory usage without loading the model
    estimated_memory_gb = estimate_model_memory_usage(model_size)

    # Multiply by 2 for safety margin
    memory_per_process_gb = estimated_memory_gb * 2

    # Reserve 2GB for the system and main process
    usable_memory_gb = max(0, get_available_memory_gb() - 2.0)

    # Calculate how many processes we can safely run
    safe_processes = max(1, int(usable_memory_gb / memory_per_process_gb))

    # Don't exceed the requested maximum
    final_processes = min(safe_processes, max_requested)

    logger.info(
        f"Memory check: {usable_memory_gb:.1f}GB usable, "
        f"{memory_per_process_gb:.1f}GB per {model_size} model process "
        f"(estimated: {estimated_memory_gb:.2f}GB (x2 safety margin))"
    )
    logger.info(
        f"Safe process limit: {safe_processes}, requested: {max_requested}, "
        f"using: {final_processes}"
    )

    if final_processes < max_requested:
        logger.warning(
            f"Limiting processes from {max_requested} to {final_processes} "
            f"due to memory constraints"
        )

    return final_processes


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def get_audio_duration(audio_file_path: str) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe

    Args:
        audio_file_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            audio_file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


def create_audio_chunks(
    audio_file_path: str,
    chunk_duration_minutes: float,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    temp_dir: str = None,
) -> list[str]:
    """
    Split audio file into chunks for streaming transcription

    Args:
        audio_file_path: Path to the audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        overlap_seconds: Overlap between chunks in seconds
        temp_dir: Optional temporary directory for chunk files

    Returns:
        List of paths to chunk files
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Get audio duration
    total_duration = get_audio_duration(audio_file_path)
    if total_duration <= 0:
        logger.warning("Could not determine audio duration, falling back to single file")
        return [audio_file_path]

    chunk_duration_seconds = chunk_duration_minutes * 60
    logger.info(
        f"Splitting audio into {chunk_duration_minutes}-minute chunks with "
        f"{overlap_seconds}s overlap"
    )
    logger.info(f"Total duration: {format_duration(total_duration)}")

    # Calculate number of chunks needed
    effective_chunk_duration = chunk_duration_seconds - overlap_seconds
    num_chunks = max(
        1, int((total_duration + effective_chunk_duration - 1) // effective_chunk_duration)
    )
    logger.info(f"Creating {num_chunks} chunks")

    # Create chunks
    chunk_files = []
    audio_path = Path(audio_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    for i in range(num_chunks):
        start_time = i * effective_chunk_duration
        # Add overlap to all chunks except the first one
        if i > 0:
            start_time -= overlap_seconds

        # Calculate end time, ensuring we don't exceed total duration
        end_time = min(start_time + chunk_duration_seconds, total_duration)

        # Skip if this chunk would be too short
        if end_time - start_time < 10:  # Skip chunks shorter than 10 seconds
            continue

        chunk_filename = f"{audio_path.stem}_chunk_{i+1:02d}_{timestamp}.mp3"
        chunk_path = os.path.join(temp_dir, chunk_filename)

        # Create chunk using ffmpeg
        cmd = [
            "ffmpeg",
            "-i",
            audio_file_path,
            "-ss",
            str(start_time),
            "-t",
            str(end_time - start_time),
            "-acodec",
            "mp3",
            "-ab",
            "192k",
            "-ar",
            "44100",
            "-y",  # Overwrite output file
            chunk_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
            chunk_files.append(chunk_path)
            chunk_size = os.path.getsize(chunk_path)
            logger.info(
                f"Created chunk {i+1}/{num_chunks}: {format_duration(end_time - start_time)} "
                f"({format_file_size(chunk_size)})"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create chunk {i+1}: {e.stderr}")
            continue

    if not chunk_files:
        logger.warning("No chunks created, falling back to original file")
        return [audio_file_path]

    return chunk_files


def transcribe_single_chunk(chunk_info: tuple) -> tuple[int, dict, float]:
    """
    Transcribe a single audio chunk - designed to be called in parallel

    Args:
        chunk_info: Tuple of (chunk_index, chunk_file_path, chunk_start_time, model_size,
                              transcribe_params)

    Returns:
        Tuple of (chunk_index, transcription_result, chunk_start_time)
    """
    chunk_index, chunk_file, chunk_start_time, model_size, transcribe_params = chunk_info

    # Setup logging for this process (important for multiprocessing)
    import logging

    # Configure logging for child process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    logger = logging.getLogger(f"chunk_{chunk_index + 1}")

    try:
        # Load model for this chunk (each process needs its own model instance)
        logger.info(f"Loading Whisper model ({model_size}) for chunk {chunk_index + 1}...")
        model = whisper.load_model(model_size)

        logger.info(f"Starting transcription of chunk {chunk_index + 1}...")
        start_time = time.time()

        # Disable verbose in multiprocessing to avoid stdout conflicts
        chunk_transcribe_params = transcribe_params.copy()
        chunk_transcribe_params["verbose"] = False

        # Create a progress callback for better feedback
        def progress_callback():
            elapsed = time.time() - start_time
            logger.info(
                f"Chunk {chunk_index + 1} transcribing... ({format_duration(elapsed)} elapsed)"
            )

        # Transcribe the chunk
        result = model.transcribe(chunk_file, **chunk_transcribe_params)

        transcription_time = time.time() - start_time

        # Show some preview of what was transcribed
        text_preview = (
            result.get("text", "")[:100] + "..."
            if len(result.get("text", "")) > 100
            else result.get("text", "")
        )
        segments_count = len(result.get("segments", []))

        logger.info(
            f"✅ Completed chunk {chunk_index + 1} in {format_duration(transcription_time)}"
        )
        logger.info(f"   📝 {segments_count} segments, preview: '{text_preview}'")

        return chunk_index, result, chunk_start_time

    except Exception as e:
        logger.error(f"❌ Failed to transcribe chunk {chunk_index + 1}: {e}")
        # Return empty result for failed chunks
        return chunk_index, {"text": "", "segments": []}, chunk_start_time


def combine_transcription_segments(chunk_results: list[tuple[int, dict, float]]) -> dict:
    """
    Combine transcription results from multiple chunks into a single result

    Args:
        chunk_results: List of tuples containing (chunk_index, transcription_result,
                                                    chunk_start_time)

    Returns:
        Combined transcription result with adjusted timing
    """
    if not chunk_results:
        return {"text": "", "segments": []}

    if len(chunk_results) == 1:
        _, result, _ = chunk_results[0]
        return result

    # Sort results by chunk index to ensure proper order
    sorted_results = sorted(chunk_results, key=lambda x: x[0])

    # Combine text from all chunks
    combined_text = ""
    combined_segments = []
    combined_language = None

    for chunk_index, result, chunk_start_time in sorted_results:
        chunk_text = result.get("text", "")
        if chunk_text.strip():
            if combined_text and not combined_text.endswith(" "):
                combined_text += " "
            combined_text += chunk_text.strip()

        # Store language from first successful chunk
        if combined_language is None and result.get("language"):
            combined_language = result.get("language")

        # Adjust segment timings and combine
        segments = result.get("segments", [])
        for segment in segments:
            adjusted_segment = segment.copy()
            # Adjust timing to account for chunk start time
            adjusted_segment["start"] = segment.get("start", 0) + chunk_start_time
            adjusted_segment["end"] = segment.get("end", 0) + chunk_start_time
            combined_segments.append(adjusted_segment)

    logger.info(f"Combined {len(chunk_results)} chunks into single transcription")
    logger.info(f"Total segments: {len(combined_segments)}")

    return {
        "text": combined_text,
        "segments": combined_segments,
        "language": combined_language or "unknown",
    }


def should_use_chunking(audio_file_path: str, max_duration_minutes: float = None) -> bool:
    """
    Determine if audio file should be processed in chunks

    Args:
        audio_file_path: Path to the audio file
        max_duration_minutes: Optional duration limit (doesn't affect chunking decision)

    Returns:
        True if chunking should be used
    """
    # Check file size
    try:
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        if file_size_mb >= MIN_FILE_SIZE_FOR_CHUNKING_MB:
            logger.info(
                f"Large file detected ({format_file_size(os.path.getsize(audio_file_path))}), "
                "chunked transcription recommended"
            )
            return True
    except OSError:
        pass

    # Check duration (use actual file duration, not the limit)
    duration_seconds = get_audio_duration(audio_file_path)
    duration_minutes = duration_seconds / 60

    # If max_duration is specified and is smaller than file duration, use that for decision
    effective_duration_minutes = duration_minutes
    if max_duration_minutes is not None and max_duration_minutes < duration_minutes:
        effective_duration_minutes = max_duration_minutes

    if effective_duration_minutes > 4:  # Use chunking for files >4 min to allow 2-minute chunks
        logger.info(
            f"Audio detected ({format_duration(effective_duration_minutes * 60)}), "
            "chunked transcription recommended"
        )
        return True

    return False


def format_structured_transcription(result: dict) -> str:
    """
    Format Whisper transcription result with timing and confidence information

    Args:
        result: Whisper transcription result dictionary

    Returns:
        Formatted string with timing and confidence data
    """
    if "segments" not in result:
        return result.get("text", "")

    formatted_lines = []
    formatted_lines.append("=== STRUCTURED TRANSCRIPTION WITH TIMING & CONFIDENCE ===\n")

    for segment in result["segments"]:
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        text = segment.get("text", "").strip()
        avg_logprob = segment.get("avg_logprob", 0)  # Average log probability (confidence)
        no_speech_prob = segment.get("no_speech_prob", 0)  # Probability of no speech

        # Convert log probability to a more intuitive confidence score (0-1)
        confidence = min(max(1 + (avg_logprob / 2), 0), 1)  # Rough conversion

        # Format timestamp
        def format_timestamp(seconds):
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}:{secs:05.2f}"

        # Only include segments with actual speech content
        if text and no_speech_prob < 0.5:
            formatted_lines.append(
                f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}] "
                f"(confidence: {confidence:.2f}) {text}"
            )

    return "\n".join(formatted_lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def clean_and_translate_transcription_with_gpt(
    transcription_data, language_detected: str = None
) -> str:
    """
    Clean and translate transcription using GPT-5 to remove uncertainties,
    correct errors, and translate to English while preserving timestamps

    Args:
        transcription_data: Either a structured transcription string with timing/confidence
                          or the raw Whisper result dictionary
        language_detected: The detected language of the original audio (optional)

    Returns:
        Cleaned and translated transcription in English with timestamps
    """
    client = OpenAI()  # Will use OPENAI_API_KEY environment variable

    # Handle both structured and simple transcription formats
    if isinstance(transcription_data, dict):
        # If we get the raw Whisper result, format it with timing/confidence
        structured_transcription = format_structured_transcription(transcription_data)
    else:
        # If we get a pre-formatted string, use it directly
        structured_transcription = str(transcription_data)

    # Build a concise prompt (minimized instructions to reduce token usage)
    prompt = (
        "Translate the following transcription into clear, fluent English. "
        "Correct obvious errors and remove filler words or uncertainties. "
        "Preserve all timestamps exactly as they appear and keep technical terms intact. "
        "Return only the cleaned English transcription.\n\n"
        f"Transcript:\n{structured_transcription}"
    )

    start_time = time.time()
    logger.info(f"Cleaning and translating transcription with {GPT_MODEL_DISPLAY_NAME}...")
    logger.info("This may take a moment...")

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at cleaning and translating transcriptions. "
                        "Focus on producing clear, accurate English while preserving "
                        "timestamps, technical accuracy, and the original structure. "
                        "Correct errors based on context but do not add new information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="high",
        )

        cleaned_text = response.choices[0].message.content
        cleaning_time = time.time() - start_time

        logger.info(f"{GPT_MODEL_DISPLAY_NAME} transcription cleaning completed successfully!")
        logger.info(f"Cleaning and translation time: {format_duration(cleaning_time)}")
        return cleaned_text

    except Exception as e:
        logger.warning(f"{GPT_MODEL_DISPLAY_NAME} transcription cleaning failed: {e}")
        logger.info("Using original transcription for summarization...")
        return structured_transcription


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_conversation_with_gpt(
    transcription_data, language_detected: str = None, meeting_description: str = None
) -> str:
    """
    Summarize the conversation using GPT-5 to create a detailed summary
    preserving the flow of thinking

    Args:
        transcription_data: Either a structured transcription string with timing/confidence
                          or the raw Whisper result dictionary
        language_detected: The detected language of the original audio (optional)
        meeting_description: Description of the meeting context (optional)

    Returns:
        Detailed conversation summary
    """
    client = OpenAI()  # Will use OPENAI_API_KEY environment variable

    # Handle both structured and simple transcription formats
    if isinstance(transcription_data, dict):
        # If we get the raw Whisper result, format it with timing/confidence
        structured_transcription = format_structured_transcription(transcription_data)
        has_timing_info = True
    else:
        # If we get a pre-formatted string, use it directly
        structured_transcription = str(transcription_data)
        has_timing_info = "confidence:" in structured_transcription

    # Create meeting context section if description provided
    meeting_context = ""
    if meeting_description:
        meeting_context = (
            f"\n\nMeeting context: {meeting_description}\n"
            "Please use this context to better understand technical terms, "
            "acronyms, and subject matter discussed."
        )

    # Language context for non-English audio
    language_context = ""
    if language_detected and language_detected != "en":
        language_context = (
            f"\n\nNote: The original audio was in language "
            f"'{language_detected}', so some transcription errors may be "
            "present due to automatic speech recognition."
        )

    # Timing and confidence context
    timing_context = ""
    if has_timing_info:
        timing_context = (
            "\n\nNote: The transcription includes timing information [MM:SS.ss --> MM:SS.ss] "
            "and confidence scores (0.0-1.0) for each segment. Use this information to:\n"
            "- Understand the natural flow and pacing of the conversation\n"
            "- Identify segments with lower confidence that may need interpretation\n"
            "- Recognize natural breaks, pauses, and transitions in the discussion\n"
            "- Better preserve the chronological structure of the conversation"
        )

    # Summarization instructions
    prompt_header = (
        "You are tasked with creating a comprehensive summary of this "
        "conversation or meeting transcript."
    )

    summarization_goals = [
        "Create a detailed summary that captures all key points and "
        "important details discussed",
        "Preserve the logical flow of thinking and progression of ideas "
        "throughout the conversation",
        "Maintain all technical details, specific examples, numbers, and " "concrete information",
        "Identify and attribute speakers by name when context allows",
        "Organize the content logically while preserving the chronological "
        "flow of the discussion",
        "Focus on substance and ideas rather than conversational style " "or atmosphere",
        "Ensure no important information or insights are lost from the " "original conversation",
    ]

    task_instruction = (
        "Please create a comprehensive summary following these guidelines:\n"
        + "\n".join(f"{i+1}. {goal}" for i, goal in enumerate(summarization_goals))
    )
    output_request = (
        "Please provide a detailed summary that preserves all important "
        "information and the flow of thinking, but presents it in a clear, "
        "organized manner without the conversational style."
    )

    # Build the complete prompt
    prompt = f"""{prompt_header}

{task_instruction}{meeting_context}{language_context}{timing_context}

Original transcript:
{structured_transcription}

{output_request}"""

    start_time = time.time()
    logger.info(f"Summarizing conversation with {GPT_MODEL_DISPLAY_NAME}...")
    logger.info("This may take a moment...")

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at creating comprehensive summaries of "
                        "conversations and meetings. Focus on preserving all "
                        "important information, technical details, and the logical "
                        "flow of ideas while presenting them in a clear, organized "
                        "manner."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="high",
        )

        summary_text = response.choices[0].message.content
        summarization_time = time.time() - start_time

        logger.info(f"{GPT_MODEL_DISPLAY_NAME} summarization completed successfully!")
        logger.info(f"Summarization time: {format_duration(summarization_time)}")
        return summary_text

    except Exception as e:
        logger.warning(f"{GPT_MODEL_DISPLAY_NAME} summarization failed: {e}")
        logger.info("Returning original transcription...")
        return structured_transcription


def convert_video_to_audio(
    video_file_path: str, temp_dir: str = None, max_duration_minutes: float = None
) -> str:
    """
    Convert video file to audio using ffmpeg

    Args:
        video_file_path: Path to the video file
        temp_dir: Optional temporary directory for audio file
        max_duration_minutes: Optional maximum duration in minutes to extract

    Returns:
        Path to the converted audio file
    """
    start_time = time.time()

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Get original file size
    original_size = os.path.getsize(video_file_path)

    # Create temporary audio file path with timestamp to avoid conflicts
    video_path = Path(video_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    temp_audio_path = os.path.join(temp_dir, f"{video_path.stem}_temp_audio_{timestamp}.mp3")

    logger.info(f"Converting video to audio: {video_file_path}")
    logger.info(f"Original file size: {format_file_size(original_size)}")
    logger.debug(f"Temporary audio file: {temp_audio_path}")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. Please install ffmpeg to convert video files."
        )

    # Convert video to audio using ffmpeg
    cmd = ["ffmpeg", "-i", video_file_path]

    # Add duration limit if specified
    if max_duration_minutes is not None:
        duration_seconds = max_duration_minutes * 60
        cmd.extend(["-t", str(duration_seconds)])

    cmd.extend(
        [
            "-vn",  # No video
            "-acodec",
            "mp3",  # Audio codec
            "-ab",
            "192k",  # Audio bitrate
            "-ar",
            "44100",  # Audio sample rate
            "-y",  # Overwrite output file
            temp_audio_path,
        ]
    )

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)

        # Get converted file size and timing
        converted_size = os.path.getsize(temp_audio_path)
        conversion_time = time.time() - start_time

        logger.info("Video conversion completed successfully!")
        logger.info(f"Converted file size: {format_file_size(converted_size)}")
        size_reduction = original_size - converted_size
        reduction_pct = size_reduction / original_size * 100
        logger.info(
            f"Size reduction: {format_file_size(size_reduction)} " f"({reduction_pct:.1f}%)"
        )
        logger.info(f"Conversion time: {format_duration(conversion_time)}")

        return temp_audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert video to audio: {e.stderr}")


def detect_audio_language(
    audio_file_path: str, detector_size: str = DEFAULT_DETECTOR_SIZE
) -> tuple[str, float]:
    """
    Detect the language of an audio file using Whisper

    Args:
        audio_file_path: Path to the audio file
        detector_size: Whisper model size for detection (default: base for better accuracy)

    Returns:
        Tuple of (language_code, confidence) e.g., ('he', 0.85)
    """
    start_time = time.time()
    logger.info(f"Detecting language using {detector_size} model...")

    # Load model for language detection
    logger.info(f"Loading Whisper {detector_size} model for language detection...")
    logger.info("(This may download the model if not already cached)")
    detector = whisper.load_model(detector_size)

    # Load and process audio (uses standard 30-second segment)
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)  # Standard 30-second segment

    mel = whisper.log_mel_spectrogram(audio).to(detector.device)

    # Detect language
    _, probs = detector.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    confidence = probs[detected_language]

    detection_time = time.time() - start_time

    logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
    logger.info(f"Language detection time: {format_duration(detection_time)}")

    # Show top 3 language predictions
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info("Top language predictions:")
    for lang, prob in sorted_probs:
        logger.info(f"  {lang}: {prob:.3f}")

    # Check confidence threshold
    if confidence < MIN_LANGUAGE_CONFIDENCE:
        logger.warning(
            f"Language detection confidence ({confidence:.2f}) is "
            f"below threshold ({MIN_LANGUAGE_CONFIDENCE:.2f})"
        )
        logger.info("Consider manually specifying the language using the --language parameter")

    return detected_language, confidence


def process_audio_with_duration_limit(
    audio_file_path: str, max_duration_minutes: float, temp_dir: str = None
) -> str:
    """
    Process audio file to extract only the first N minutes using ffmpeg

    Args:
        audio_file_path: Path to the audio file
        max_duration_minutes: Maximum duration in minutes to extract
        temp_dir: Optional temporary directory for processed audio file

    Returns:
        Path to the processed audio file
    """
    start_time = time.time()

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Get original file size
    original_size = os.path.getsize(audio_file_path)

    # Create temporary processed audio file path with timestamp to avoid conflicts
    audio_path = Path(audio_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    temp_processed_path = os.path.join(
        temp_dir, f"{audio_path.stem}_processed_{max_duration_minutes}min_{timestamp}.mp3"
    )

    logger.info(f"Extracting first {max_duration_minutes} minutes from audio file...")
    logger.info(f"Original audio file size: {format_file_size(original_size)}")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. Please install ffmpeg to process audio files."
        )

    # Extract duration-limited audio using ffmpeg
    duration_seconds = max_duration_minutes * 60
    cmd = [
        "ffmpeg",
        "-i",
        audio_file_path,
        "-t",
        str(duration_seconds),  # Duration limit
        "-acodec",
        "mp3",  # Audio codec
        "-ab",
        "192k",  # Audio bitrate
        "-ar",
        "44100",  # Audio sample rate
        "-y",  # Overwrite output file
        temp_processed_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)

        # Get processed file size and timing
        processed_size = os.path.getsize(temp_processed_path)
        processing_time = time.time() - start_time

        logger.info(
            f"Audio processing completed successfully! Limited to {max_duration_minutes} minutes."
        )
        logger.info(f"Processed file size: {format_file_size(processed_size)}")
        size_reduction = original_size - processed_size
        reduction_pct = size_reduction / original_size * 100
        logger.info(f"Size reduction: {format_file_size(size_reduction)} ({reduction_pct:.1f}%)")
        logger.info(f"Processing time: {format_duration(processing_time)}")

        return temp_processed_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to process audio file: {e.stderr}")


def trim_silence_from_audio(audio_file_path: str, temp_dir: str = None) -> str:
    """
    Remove silence from the beginning and end of audio file using ffmpeg

    Args:
        audio_file_path: Path to the audio file
        temp_dir: Optional temporary directory for processed audio file

    Returns:
        Path to the silence-trimmed audio file
    """
    start_time = time.time()

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Get original file size
    original_size = os.path.getsize(audio_file_path)

    # Create temporary trimmed audio file path with timestamp to avoid conflicts
    audio_path = Path(audio_file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    temp_trimmed_path = os.path.join(temp_dir, f"{audio_path.stem}_silencetrimmed_{timestamp}.mp3")

    logger.info("Removing silence from audio file...")
    logger.info(f"Original audio file size: {format_file_size(original_size)}")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not available, skipping silence removal")
        return audio_file_path

    # Remove silence using ffmpeg with conservative settings
    cmd = [
        "ffmpeg",
        "-i",
        audio_file_path,
        "-af",
        (
            "silenceremove=start_periods=1:start_duration=1:"
            "start_threshold=-50dB:detection=peak,silenceremove=stop_periods=-1:"
            "stop_duration=1:stop_threshold=-50dB:detection=peak"
        ),
        "-acodec",
        "mp3",
        "-ab",
        "192k",
        "-ar",
        "44100",
        "-y",  # Overwrite output file
        temp_trimmed_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)

        # Get trimmed file size and timing
        trimmed_size = os.path.getsize(temp_trimmed_path)
        trimming_time = time.time() - start_time

        logger.info("Silence removal completed successfully!")
        logger.info(f"Trimmed file size: {format_file_size(trimmed_size)}")
        if trimmed_size < original_size:
            size_reduction = original_size - trimmed_size
            reduction_pct = size_reduction / original_size * 100
            logger.info(
                f"Size reduction: {format_file_size(size_reduction)} ({reduction_pct:.1f}%)"
            )
        logger.info(f"Silence removal time: {format_duration(trimming_time)}")

        return temp_trimmed_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to trim silence ({e.stderr}), using original audio")
        return audio_file_path


def transcribe_audio(
    audio_file_path: str,
    output_file_path: str = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    original_input_file: str = None,
    language: str = None,
    max_duration_minutes: float = None,
    use_gpt_refinement: bool = True,
    meeting_description: str = None,
    use_multiprocessing: bool = False,
    max_workers: int = MAX_PARALLEL_CHUNKS,
) -> tuple[str, str]:
    """
    Transcribe audio file to text using Whisper and optionally create a
    detailed summary using GPT

    Args:
        audio_file_path: Path to the audio file (should be audio format,
            not video)
        output_file_path: Optional path to save transcription (defaults to
            same name with .txt extension)
        model_size: Whisper model size (tiny, base, small, medium, large)
        original_input_file: Optional path to original input file (for
            display in output header)
        language: Language code (e.g., 'he', 'en'). If None, will
            auto-detect language
        max_duration_minutes: Optional maximum duration in minutes to
            transcribe
        use_gpt_refinement: Whether to create a summary of the transcription
            using GPT (default: True)
        meeting_description: Optional description of the meeting context
            for better GPT summarization
        use_multiprocessing: Whether to enable multiprocessing for large files
            (default: False)
        max_workers: Maximum number of parallel processes for chunked transcription
            when multiprocessing is enabled (default: auto-detected from CPU cores, max 8).
            Actual count may be limited by available memory.

    Returns:
        Tuple of (transcribed_text, output_file_path) where transcribed_text
        is summarized with GPT if enabled
    """

    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Process audio file with duration limit if specified
    temp_processed_file = None
    actual_audio_file = audio_file_path

    if max_duration_minutes is not None:
        logger.info(f"Limiting transcription to first {max_duration_minutes} minutes...")
        actual_audio_file = process_audio_with_duration_limit(
            audio_file_path, max_duration_minutes
        )
        temp_processed_file = actual_audio_file  # Keep track for cleanup

    try:
        transcription_start_time = time.time()

        # Detect language if not provided (using the processed audio file if duration was limited)
        if language is None:
            language, confidence = detect_audio_language(actual_audio_file)
        else:
            logger.info(f"Using specified language: {language}")
            confidence = None

        # Apply silence trimming for better accuracy
        logger.info("Applying audio preprocessing...")
        trimmed_audio_file = trim_silence_from_audio(actual_audio_file)

        # Check if we should use chunking for large files (only when multiprocessing is enabled)
        use_chunking = use_multiprocessing and should_use_chunking(
            trimmed_audio_file, max_duration_minutes
        )

        if not use_multiprocessing and should_use_chunking(
            trimmed_audio_file, max_duration_minutes
        ):
            logger.info(
                "Chunked transcription recommended but multiprocessing is disabled - "
                "using single-process transcription instead"
            )
            logger.info(
                "Consider using --multiprocessing flag for faster processing of large files"
            )

        if use_chunking:
            logger.info("Using streaming/chunked transcription for large audio file")

            # First, determine memory-safe worker count
            memory_safe_workers = calculate_safe_process_count(max_workers, model_size)

            # Now optimize chunk size to match the ACTUAL number of workers we can use
            audio_duration_seconds = get_audio_duration(trimmed_audio_file)
            audio_duration_minutes = audio_duration_seconds / 60

            # Account for overlap when calculating optimal chunk duration
            # Each chunk overlaps by CHUNK_OVERLAP_SECONDS, so effective step size is
            # chunk_duration - overlap
            # For memory_safe_workers chunks:
            # (chunk_duration - overlap) * memory_safe_workers = total_duration
            # Therefore:
            # chunk_duration = (total_duration / memory_safe_workers) + overlap
            optimized_chunk_duration_minutes = (
                audio_duration_seconds / memory_safe_workers + CHUNK_OVERLAP_SECONDS
            ) / 60

            # Ensure we don't go below the minimum 2 minutes
            if optimized_chunk_duration_minutes < 2.0:
                optimized_chunk_duration_minutes = 2.0
                # Recalculate actual chunks needed with the minimum duration
                # using the same logic as create_audio_chunks
                effective_chunk_seconds = (
                    optimized_chunk_duration_minutes * 60 - CHUNK_OVERLAP_SECONDS
                )
                actual_num_chunks = max(
                    1, math.ceil(audio_duration_seconds / effective_chunk_seconds)
                )
            else:
                # Perfect case: exactly memory_safe_workers chunks
                actual_num_chunks = memory_safe_workers

            logger.info(
                f"Optimizing for {memory_safe_workers} memory-safe workers: "
                f"{optimized_chunk_duration_minutes:.1f}min chunks "
                f"({actual_num_chunks} total chunks)"
            )

            # Create audio chunks with optimized duration
            chunk_files = create_audio_chunks(trimmed_audio_file, optimized_chunk_duration_minutes)
            temp_chunk_files = chunk_files.copy()  # Keep track for cleanup

            # Define transcription parameters (only non-defaults)
            transcribe_params = {
                "language": language,  # Use detected or specified language
                "best_of": 2,  # Try multiple attempts and pick best (default: 5)
                "beam_size": 3,  # Use beam search for better accuracy (default: 5)
                "patience": 1.0,  # Wait longer for better results (default: None)
                "verbose": True,  # Show progress (default: False)
            }

            # Log parameters being used
            param_summary = ", ".join(
                f"{k}={v}" for k, v in transcribe_params.items() if k != "verbose"
            )
            logger.info(f"Using parameters: {param_summary}")

            total_chunks = len(chunk_files)
            num_workers = min(memory_safe_workers, total_chunks)

            logger.info(f"CPU cores detected: {_CPU_CORES}, max workers configured: {max_workers}")
            logger.info(f"Memory-safe workers: {memory_safe_workers}")
            logger.info(
                f"Processing {total_chunks} optimized chunks using "
                f"{num_workers} processes simultaneously"
            )

            # Prepare chunk information for parallel processing
            chunk_tasks = []
            for i, chunk_file in enumerate(chunk_files):
                chunk_start_time = i * (
                    optimized_chunk_duration_minutes * 60 - CHUNK_OVERLAP_SECONDS
                )
                if i > 0:  # Add overlap back for timing calculation
                    chunk_start_time -= CHUNK_OVERLAP_SECONDS

                chunk_info = (i, chunk_file, chunk_start_time, model_size, transcribe_params)
                chunk_tasks.append(chunk_info)

            # Process chunks in parallel using multiprocessing
            transcription_process_start = time.time()
            chunk_results = []
            completed_chunks = 0

            logger.info(f"🚀 Starting parallel transcription of {total_chunks} chunks...")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all chunk transcription tasks
                future_to_chunk = {
                    executor.submit(transcribe_single_chunk, chunk_task): chunk_task[0]
                    for chunk_task in chunk_tasks
                }

                logger.info(f"📋 All {total_chunks} tasks submitted to processes")

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        chunk_results.append(chunk_result)
                        completed_chunks += 1

                        # Show real-time progress
                        elapsed = time.time() - transcription_process_start
                        logger.info(
                            f"📊 Progress: {completed_chunks}/{total_chunks} chunks completed "
                            f"({format_duration(elapsed)} elapsed)"
                        )

                        if completed_chunks < total_chunks:
                            remaining = total_chunks - completed_chunks
                            est_time_per_chunk = elapsed / completed_chunks
                            est_remaining = est_time_per_chunk * remaining
                            logger.info(
                                f"⏱️  Estimated time remaining: {format_duration(est_remaining)}"
                            )

                    except Exception as e:
                        logger.error(f"❌ Failed to get result for chunk {chunk_index + 1}: {e}")
                        # Add empty result to maintain order
                        chunk_results.append((chunk_index, {"text": "", "segments": []}, 0.0))
                        completed_chunks += 1

            # Combine results from all chunks
            if chunk_results:
                logger.info("🔗 Combining results from all chunks...")
                result = combine_transcription_segments(chunk_results)

                # Show summary of parallel transcription
                total_segments = len(result.get("segments", []))
                total_text_length = len(result.get("text", ""))
                logger.info(
                    f"✅ Parallel transcription complete! Combined {total_segments} segments, "
                    f"{total_text_length} characters"
                )
            else:
                logger.error("❌ No chunks were successfully transcribed")
                result = {"text": "", "segments": []}

            transcription_process_time = time.time() - transcription_process_start

        else:
            # Standard single-file transcription
            model_load_start = time.time()
            logger.info(f"Loading Whisper model ({model_size})...")
            logger.info("(This may download the model if not already cached)")
            model = whisper.load_model(model_size)
            model_load_time = time.time() - model_load_start
            logger.info(f"Model loading time: {format_duration(model_load_time)}")

            duration_info = (
                f" (first {max_duration_minutes} minutes only)" if max_duration_minutes else ""
            )
            logger.info(f"Transcribing audio file: {audio_file_path}{duration_info}")
            logger.info("This may take a few minutes depending on the file size...")

            # Define transcription parameters (only non-defaults)
            transcribe_params = {
                "language": language,  # Use detected or specified language
                "best_of": 2,  # Try multiple attempts and pick best (default: 5)
                "beam_size": 3,  # Use beam search for better accuracy (default: 5)
                "patience": 1.0,  # Wait longer for better results (default: None)
                "verbose": True,  # Show progress (default: False)
            }

            # Log parameters being used
            param_summary = ", ".join(
                f"{k}={v}" for k, v in transcribe_params.items() if k != "verbose"
            )
            logger.info(f"Using parameters: {param_summary}")

            # Transcribe using the silence-trimmed audio file
            transcription_process_start = time.time()
            result = model.transcribe(trimmed_audio_file, **transcribe_params)
            transcription_process_time = time.time() - transcription_process_start

        # Extract the transcribed text
        transcribed_text = result["text"]
        logger.info(f"Whisper transcription time: {format_duration(transcription_process_time)}")

        # Clean and translate transcription, then summarize with GPT if enabled
        cleaned_transcription = None
        if use_gpt_refinement:
            # First, clean and translate the transcription
            cleaned_transcription = clean_and_translate_transcription_with_gpt(
                result, language  # Pass full result with segments
            )
            # Remove Whisper confidence annotations now that cleaning is complete
            cleaned_transcription = re.sub(
                r"\s*\(confidence: [0-9.]+\)", "", cleaned_transcription
            )

            # Then summarize based on the cleaned transcription
            summary_text = summarize_conversation_with_gpt(
                cleaned_transcription, language, meeting_description  # Use cleaned transcription
            )
        else:
            # When not using GPT, still provide structured format for consistency
            summary_text = format_structured_transcription(result)
            logger.info("Skipping GPT cleaning and summarization as requested.")

        total_transcription_time = time.time() - transcription_start_time
        logger.info(
            f"Total transcription processing time: {format_duration(total_transcription_time)}"
        )

        # Determine output file path with timestamp to avoid overwriting
        if output_file_path is None:
            # Use original input file if available, otherwise use audio file path
            source_path = original_input_file if original_input_file else audio_file_path
            base_name = os.path.splitext(source_path)[0]
            # Add timestamp when process ends to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_path = f"{base_name}_transcription_{timestamp}.txt"

        # Save transcription to file
        with open(output_file_path, "w", encoding="utf-8") as f:
            # Show original input file in header (useful when video was converted to audio)
            source_file = original_input_file if original_input_file else audio_file_path
            f.write(f"Transcription of: {os.path.basename(source_file)}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {model_size}\n")
            f.write(f"Language: {language}\n")

            # Add duration information if limited
            if max_duration_minutes is not None:
                f.write(f"Duration limit: {max_duration_minutes} minutes\n")

            # Simple parameter logging
            params_str = ", ".join(
                f"{k}={v}" for k, v in transcribe_params.items() if k != "verbose"
            )
            f.write(f"Parameters: {params_str}\n")
            f.write("Silence trimming: applied\n")
            f.write("Enhanced format: timing & confidence data included\n")

            # Add chunking information
            if use_chunking and "temp_chunk_files" in locals():
                f.write(
                    f"Streaming transcription: {len(temp_chunk_files)} chunks processed "
                    f"using {num_workers} processes\n"
                )
            else:
                multiprocessing_status = (
                    "enabled but not used (single file processing)"
                    if use_multiprocessing
                    else "disabled"
                )
                f.write(f"Streaming transcription: {multiprocessing_status}\n")

            # Add GPT processing information
            gpt_status = "enabled" if use_gpt_refinement else "disabled"
            f.write(f"{GPT_MODEL_DISPLAY_NAME} cleaning & translation: {gpt_status}\n")
            f.write(f"{GPT_MODEL_DISPLAY_NAME} summarization: {gpt_status}\n")

            # Add meeting description if provided
            if meeting_description:
                f.write(f"Meeting description: {meeting_description}\n")

            f.write("=" * 50 + "\n\n")

            # Write summary if GPT was used, otherwise structured transcription
            if use_gpt_refinement:
                f.write(f"=== CONVERSATION SUMMARY ({GPT_MODEL_DISPLAY_NAME}) ===\n\n")
                f.write(summary_text)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write(
                    f"=== CLEANED & TRANSLATED TRANSCRIPTION ({GPT_MODEL_DISPLAY_NAME}) ===\n\n"
                )
                f.write(cleaned_transcription)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("=== ORIGINAL TRANSCRIPTION WITH TIMING & CONFIDENCE (Whisper) ===\n\n")
                f.write(format_structured_transcription(result))
            else:
                f.write(summary_text)  # This is already the structured format

        logger.info("Processing completed!")
        logger.info(f"Output saved to: {output_file_path}")

        # Show file information
        output_file_size = os.path.getsize(output_file_path)
        logger.info(f"Output file size: {format_file_size(output_file_size)}")

        # Show preview of final result (summary if GPT was used)
        final_text = summary_text
        if use_gpt_refinement:
            logger.info("Final summary preview (first 200 characters):")
        else:
            logger.info("Final structured transcription preview (first 200 characters):")
        logger.info(f"{final_text[:200]}...")

        return final_text, output_file_path

    finally:
        # Clean up temporary processed file if it was created
        if temp_processed_file and os.path.exists(temp_processed_file):
            try:
                os.remove(temp_processed_file)
                logger.debug(f"Cleaned up temporary processed file: {temp_processed_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_processed_file}: {e}")

        # Clean up chunk files if they were created
        if "temp_chunk_files" in locals():
            for chunk_file in temp_chunk_files:
                if chunk_file != trimmed_audio_file and os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                        logger.debug(f"Cleaned up temporary chunk file: {chunk_file}")
                    except OSError as e:
                        logger.warning(f"Could not remove temporary chunk file {chunk_file}: {e}")

        # Clean up trimmed audio file if it was created and is different from original
        try:
            if (
                "trimmed_audio_file" in locals()
                and trimmed_audio_file != actual_audio_file
                and os.path.exists(trimmed_audio_file)
            ):
                os.remove(trimmed_audio_file)
                logger.debug(f"Cleaned up temporary trimmed audio file: {trimmed_audio_file}")
        except (OSError, NameError) as e:
            if "trimmed_audio_file" in locals():
                logger.warning(
                    f"Could not remove temporary trimmed file {trimmed_audio_file}: {e}"
                )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using Whisper and summarize with GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s recording.mp4
  %(prog)s recording.mp4 --model large-v3-turbo
  %(prog)s recording.mp4 --description 'Weekly team standup discussing project progress'
  %(prog)s recording.mp4 --language he --description 'Technical discussion about ML models'
  %(prog)s recording.mp4 --max-duration 10 --description 'Demo: first 10 minutes only'
  %(prog)s recording.mp4 --multiprocessing --max-workers 4  # Enable parallel processing
  %(prog)s recording.mp4 --no-gpt

Supported formats:
  Audio: .mp3, .wav, .flac, .m4a, .aac, .ogg
  Video: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v (requires ffmpeg)

Requirements:
  - ffmpeg for video file conversion
  - OPENAI_API_KEY environment variable for {GPT_MODEL_DISPLAY_NAME} summarization
        """,
    )

    parser.add_argument(
        "input_file", help="Path to audio (.mp3, .wav, etc.) or video (.mp4, .avi, etc.) file"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE})",
    )

    parser.add_argument(
        "--language", help="Language code (e.g., 'he', 'en', 'fr') to skip auto-detection"
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        metavar="MINUTES",
        help="Maximum duration in minutes to transcribe (e.g., 1, 5, 10.5)",
    )

    parser.add_argument(
        "--description", help="Meeting description for better GPT summarization context"
    )

    parser.add_argument(
        "--no-gpt",
        action="store_true",
        help=f"Disable {GPT_MODEL_DISPLAY_NAME} summarization and use only Whisper transcription",
    )

    parser.add_argument(
        "--output", help="Output file path (default: auto-generated with timestamp)"
    )

    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help=(
            "Enable multiprocessing for large files "
            "(uses chunked transcription with parallel processing)"
        ),
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_PARALLEL_CHUNKS,
        metavar="N",
        help=(
            f"Maximum number of parallel processes to use when multiprocessing is enabled "
            f"(default: {MAX_PARALLEL_CHUNKS}, auto-detected from CPU cores). "
            f"Actual count may be limited by available memory."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input_file):
        parser.error(f"Input file not found: {args.input_file}")

    if args.max_duration is not None and args.max_duration <= 0:
        parser.error("--max-duration must be a positive number")

    if args.max_workers <= 0:
        parser.error("--max-workers must be a positive number")

    return args


def main():
    """Main entry point with improved error handling and logging"""
    try:
        args = parse_arguments()

        # Set up logging level
        logger.setLevel(getattr(logging, args.log_level))
        for handler in logger.handlers:
            handler.setLevel(getattr(logging, args.log_level))

        input_file = args.input_file
        model_size = args.model
        language = args.language
        max_duration_minutes = args.max_duration
        use_gpt_refinement = not args.no_gpt
        meeting_description = args.description
        output_file_path = args.output

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)

    # Determine if input is video or audio
    file_extension = Path(input_file).suffix.lower()
    is_video = file_extension in VIDEO_EXTENSIONS
    temp_audio_file = None

    try:
        overall_start_time = time.time()

        if is_video:
            logger.info(f"Detected video file: {input_file}")
            # Convert video to audio (with optional duration limit)
            audio_file = convert_video_to_audio(
                input_file, max_duration_minutes=max_duration_minutes
            )
            temp_audio_file = audio_file  # Keep track for cleanup
        else:
            logger.info(f"Detected audio file: {input_file}")
            audio_file = input_file

        logger.info("Starting transcription and processing...")
        processed_text, actual_output_path = transcribe_audio(
            audio_file,
            output_file_path=output_file_path,
            model_size=model_size,
            original_input_file=input_file,
            language=language,
            max_duration_minutes=max_duration_minutes,
            use_gpt_refinement=use_gpt_refinement,
            meeting_description=meeting_description,
            use_multiprocessing=args.multiprocessing,
            max_workers=args.max_workers,
        )

        overall_time = time.time() - overall_start_time

        if use_gpt_refinement:
            logger.info(f"Full summary length: {len(processed_text)} characters")
        else:
            logger.info(f"Full transcription length: {len(processed_text)} characters")

        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total processing time: {format_duration(overall_time)}")
        logger.info(f"Original file size: {format_file_size(os.path.getsize(input_file))}")
        if is_video and audio_file and os.path.exists(audio_file):
            logger.info(f"Audio file size: {format_file_size(os.path.getsize(audio_file))}")
        logger.info(f"Output file size: {format_file_size(os.path.getsize(actual_output_path))}")
        logger.info(f"Output file: {os.path.basename(actual_output_path)}")
        logger.info("=============================")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary audio file if it was created
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                logger.debug(f"Cleaned up temporary audio file: {temp_audio_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_audio_file}: {e}")


if __name__ == "__main__":
    main()
