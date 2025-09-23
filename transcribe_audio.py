import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import whisper
from openai import OpenAI

DEFAULT_MODEL_SIZE = "large-v3-turbo"
DEFAULT_DETECTOR_SIZE = "base"  # Better accuracy for language detection
MIN_LANGUAGE_CONFIDENCE = 0.8  # Minimum confidence threshold

# GPT model configuration
GPT_MODEL_NAME = "gpt-5"
GPT_MODEL_DISPLAY_NAME = "GPT-5 (high)"

# Video file extensions that require conversion
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


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
    print(f"Summarizing conversation with {GPT_MODEL_DISPLAY_NAME}...")
    print("This may take a moment...")

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

        print(f"{GPT_MODEL_DISPLAY_NAME} summarization completed successfully!")
        print(f"Summarization time: {format_duration(summarization_time)}")
        return summary_text

    except Exception as e:
        print(f"Warning: {GPT_MODEL_DISPLAY_NAME} summarization failed: {e}")
        print("Returning original transcription...")
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

    print(f"Converting video to audio: {video_file_path}")
    print(f"Original file size: {format_file_size(original_size)}")
    print(f"Temporary audio file: {temp_audio_path}")

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

        print("Video conversion completed successfully!")
        print(f"Converted file size: {format_file_size(converted_size)}")
        size_reduction = original_size - converted_size
        reduction_pct = size_reduction / original_size * 100
        print(f"Size reduction: {format_file_size(size_reduction)} " f"({reduction_pct:.1f}%)")
        print(f"Conversion time: {format_duration(conversion_time)}")

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
    print(f"Detecting language using {detector_size} model...")

    # Load model for language detection
    print(f"Loading Whisper {detector_size} model for language detection...")
    print("(This may download the model if not already cached)")
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

    print(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
    print(f"Language detection time: {format_duration(detection_time)}")

    # Show top 3 language predictions
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top language predictions:")
    for lang, prob in sorted_probs:
        print(f"  {lang}: {prob:.3f}")

    # Check confidence threshold
    if confidence < MIN_LANGUAGE_CONFIDENCE:
        print(
            f"Warning: Language detection confidence ({confidence:.2f}) is "
            f"below threshold ({MIN_LANGUAGE_CONFIDENCE:.2f})"
        )
        print("Consider manually specifying the language using the " "--language parameter")

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

    print(f"Extracting first {max_duration_minutes} minutes from audio file...")
    print(f"Original audio file size: {format_file_size(original_size)}")

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

        print(
            f"Audio processing completed successfully! Limited to {max_duration_minutes} minutes."
        )
        print(f"Processed file size: {format_file_size(processed_size)}")
        size_reduction = original_size - processed_size
        reduction_pct = size_reduction / original_size * 100
        print(f"Size reduction: {format_file_size(size_reduction)} " f"({reduction_pct:.1f}%)")
        print(f"Processing time: {format_duration(processing_time)}")

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

    print("Removing silence from audio file...")
    print(f"Original audio file size: {format_file_size(original_size)}")

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ffmpeg not available, skipping silence removal")
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

        print("Silence removal completed successfully!")
        print(f"Trimmed file size: {format_file_size(trimmed_size)}")
        if trimmed_size < original_size:
            size_reduction = original_size - trimmed_size
            reduction_pct = size_reduction / original_size * 100
            print(f"Size reduction: {format_file_size(size_reduction)} " f"({reduction_pct:.1f}%)")
        print(f"Silence removal time: {format_duration(trimming_time)}")

        return temp_trimmed_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to trim silence ({e.stderr}), using original audio")
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
        print(f"Limiting transcription to first {max_duration_minutes} minutes...")
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
            print(f"Using specified language: {language}")
            confidence = None

        # Apply silence trimming for better accuracy
        print("\nApplying audio preprocessing...")
        trimmed_audio_file = trim_silence_from_audio(actual_audio_file)

        model_load_start = time.time()
        print(f"Loading Whisper model ({model_size})...")
        print("(This may download the model if not already cached)")
        model = whisper.load_model(model_size)
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {format_duration(model_load_time)}")

        duration_info = (
            f" (first {max_duration_minutes} minutes only)" if max_duration_minutes else ""
        )
        print(f"Transcribing audio file: {audio_file_path}{duration_info}")
        print("This may take a few minutes depending on the file size...")

        # Define transcription parameters (only non-defaults)
        transcribe_params = {
            "language": language,  # Use detected or specified language
            "best_of": 2,  # Try multiple attempts and pick best (default: 5)
            "beam_size": 3,  # Use beam search for better accuracy (default: 5)
            "patience": 1.0,  # Wait longer for better results (default: None)
            "verbose": True,  # Show progress (default: False)
        }

        # Print parameters being used
        param_summary = ", ".join(
            f"{k}={v}" for k, v in transcribe_params.items() if k != "verbose"
        )
        print(f"Using parameters: {param_summary}")

        # Transcribe using the silence-trimmed audio file
        transcription_process_start = time.time()
        result = model.transcribe(trimmed_audio_file, **transcribe_params)
        transcription_process_time = time.time() - transcription_process_start

        # Extract the transcribed text
        transcribed_text = result["text"]
        print(f"\nWhisper transcription time: {format_duration(transcription_process_time)}")

        # Summarize conversation with GPT if enabled
        if use_gpt_refinement:
            summary_text = summarize_conversation_with_gpt(
                result, language, meeting_description  # Pass full result with segments
            )
        else:
            # When not using GPT, still provide structured format for consistency
            summary_text = format_structured_transcription(result)
            print("Skipping GPT summarization as requested.")

        total_transcription_time = time.time() - transcription_start_time
        print(
            f"\nTotal transcription processing time: {format_duration(total_transcription_time)}"
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

            # Add GPT summarization information
            gpt_status = "enabled" if use_gpt_refinement else "disabled"
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
                f.write("=== ORIGINAL TRANSCRIPTION WITH TIMING & CONFIDENCE (Whisper) ===\n\n")
                f.write(format_structured_transcription(result))
            else:
                f.write(summary_text)  # This is already the structured format

        print("Processing completed!")
        print(f"Output saved to: {output_file_path}")

        # Show file information
        output_file_size = os.path.getsize(output_file_path)
        print(f"Output file size: {format_file_size(output_file_size)}")

        # Show preview of final result (summary if GPT was used)
        final_text = summary_text
        if use_gpt_refinement:
            print("Final summary preview (first 200 characters):")
        else:
            print("Final structured transcription preview (first 200 characters):")
        print(f"{final_text[:200]}...")

        return final_text, output_file_path

    finally:
        # Clean up temporary processed file if it was created
        if temp_processed_file and os.path.exists(temp_processed_file):
            try:
                os.remove(temp_processed_file)
                print(f"Cleaned up temporary processed file: {temp_processed_file}")
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_processed_file}: {e}")

        # Clean up trimmed audio file if it was created and is different from original
        try:
            if (
                "trimmed_audio_file" in locals()
                and trimmed_audio_file != actual_audio_file
                and os.path.exists(trimmed_audio_file)
            ):
                os.remove(trimmed_audio_file)
                print(f"Cleaned up temporary trimmed audio file: {trimmed_audio_file}")
        except (OSError, NameError) as e:
            if "trimmed_audio_file" in locals():
                print(
                    f"Warning: Could not remove temporary trimmed file {trimmed_audio_file}: {e}"
                )


def main():
    # Show usage if requested
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(
            "Usage: python transcribe_audio.py [input_file] [model_size] "
            "[--language LANG] [--max-duration MINUTES] [--description DESC] "
            "[--no-gpt]"
        )
        print("  input_file: Path to audio (.mp3, .wav, etc.) or video " "(.mp4, .avi, etc.) file")
        print(f"  model_size: Whisper model size (default: {DEFAULT_MODEL_SIZE})")
        print("  --language: Language code (e.g., 'he', 'en', 'fr') to skip " "auto-detection")
        print("  --max-duration: Maximum duration in minutes to transcribe " "(e.g., 1, 5, 10.5)")
        print("  --description: Meeting description for better GPT " "summarization context")
        print(
            f"  --no-gpt: Disable {GPT_MODEL_DISPLAY_NAME} summarization and "
            "use only Whisper transcription"
        )
        print("  Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, " ".flv, .webm, .m4v")
        print("  Requires ffmpeg for video file conversion")
        print(
            f"  Requires OPENAI_API_KEY environment variable for "
            f"{GPT_MODEL_DISPLAY_NAME} summarization"
        )
        print("\nExamples:")
        print("  python transcribe_audio.py recording.mp4")
        print("  python transcribe_audio.py recording.mp4 large-v3-turbo")
        print(
            "  python transcribe_audio.py recording.mp4 --description "
            "'Weekly team standup discussing project progress'"
        )
        print(
            "  python transcribe_audio.py recording.mp4 --language he "
            "--description 'Technical discussion about machine learning models'"
        )
        print(
            "  python transcribe_audio.py recording.mp4 --max-duration 1 "
            "--description 'Quick client call about requirements'"
        )
        print("  python transcribe_audio.py recording.mp4 --no-gpt")
        sys.exit(0)

    # Check if input file provided
    if len(sys.argv) < 2:
        print("Error: Please provide an input file")
        print(
            "Usage: python transcribe_audio.py [input_file] [model_size] "
            "[--language LANG] [--max-duration MINUTES] [--description DESC] "
            "[--no-gpt]"
        )
        print("Use --help for more information")
        sys.exit(1)

    input_file = sys.argv[1]
    model_size = DEFAULT_MODEL_SIZE
    language = None
    max_duration_minutes = None
    use_gpt_refinement = True
    meeting_description = None

    # Parse remaining arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--language":
            if i + 1 >= len(sys.argv):
                print("Error: --language requires a language code")
                sys.exit(1)
            language = sys.argv[i + 1]
            i += 2
        elif arg == "--max-duration":
            if i + 1 >= len(sys.argv):
                print("Error: --max-duration requires a duration in minutes")
                sys.exit(1)
            try:
                max_duration_minutes = float(sys.argv[i + 1])
                if max_duration_minutes <= 0:
                    print("Error: --max-duration must be a positive number")
                    sys.exit(1)
            except ValueError:
                print("Error: --max-duration must be a valid number")
                sys.exit(1)
            i += 2
        elif arg == "--description":
            if i + 1 >= len(sys.argv):
                print("Error: --description requires a meeting description")
                sys.exit(1)
            meeting_description = sys.argv[i + 1]
            i += 2
        elif arg == "--no-gpt":
            use_gpt_refinement = False
            i += 1
        else:
            # Assume it's model size if not a flag
            model_size = arg
            i += 1

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Determine if input is video or audio
    file_extension = Path(input_file).suffix.lower()
    is_video = file_extension in VIDEO_EXTENSIONS
    temp_audio_file = None

    try:
        overall_start_time = time.time()

        if is_video:
            print(f"Detected video file: {input_file}")
            # Convert video to audio (with optional duration limit)
            audio_file = convert_video_to_audio(
                input_file, max_duration_minutes=max_duration_minutes
            )
            temp_audio_file = audio_file  # Keep track for cleanup
        else:
            print(f"Detected audio file: {input_file}")
            audio_file = input_file

        # Let transcribe_audio function handle the timestamped naming
        output_file_path = None

        print("Starting transcription and processing...")
        processed_text, actual_output_path = transcribe_audio(
            audio_file,
            output_file_path=output_file_path,
            model_size=model_size,
            original_input_file=input_file,
            language=language,
            max_duration_minutes=max_duration_minutes,
            use_gpt_refinement=use_gpt_refinement,
            meeting_description=meeting_description,
        )

        overall_time = time.time() - overall_start_time

        if use_gpt_refinement:
            print(f"\nFull summary length: {len(processed_text)} characters")
        else:
            print(f"\nFull transcription length: {len(processed_text)} characters")

        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Total processing time: {format_duration(overall_time)}")
        print(f"Original file size: {format_file_size(os.path.getsize(input_file))}")
        if is_video and audio_file and os.path.exists(audio_file):
            print(f"Audio file size: {format_file_size(os.path.getsize(audio_file))}")
        print(f"Output file size: {format_file_size(os.path.getsize(actual_output_path))}")
        print(f"Output file: {os.path.basename(actual_output_path)}")
        print("=============================")

    finally:
        # Clean up temporary audio file if it was created
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                print(f"Cleaned up temporary audio file: {temp_audio_file}")
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_audio_file}: {e}")


if __name__ == "__main__":
    main()
