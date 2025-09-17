import whisper
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from openai import OpenAI

DEFAULT_MODEL_SIZE = "large-v3-turbo"
DEFAULT_DETECTOR_SIZE = "base"  # Better accuracy for language detection
MIN_LANGUAGE_CONFIDENCE = 0.8  # Minimum confidence threshold

# GPT model configuration
GPT_MODEL_NAME = "gpt-5"
GPT_MODEL_DISPLAY_NAME = "GPT-5 (high)"

# Video file extensions that require conversion
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def summarize_conversation_with_gpt(transcription: str, language_detected: str = None, meeting_description: str = None) -> str:
    """
    Summarize the conversation using GPT-5 to create a detailed summary preserving the flow of thinking
    
    Args:
        transcription: The raw transcription text from Whisper
        language_detected: The detected language of the original audio (optional)
        meeting_description: Description of the meeting context (optional)
    
    Returns:
        Detailed conversation summary
    """
    client = OpenAI()  # Will use OPENAI_API_KEY environment variable
    
    # Create meeting context section if description provided
    meeting_context = ""
    if meeting_description:
        meeting_context = f"\n\nMeeting context: {meeting_description}\nPlease use this context to better understand technical terms, acronyms, and subject matter discussed."

    # Language context for non-English audio
    language_context = ""
    if language_detected and language_detected != 'en':
        language_context = f"\n\nNote: The original audio was in language '{language_detected}', so some transcription errors may be present due to automatic speech recognition."

    # Summarization instructions
    prompt_header = "You are tasked with creating a comprehensive summary of this conversation or meeting transcript."
    
    summarization_goals = [
        "Create a detailed summary that captures all key points and important details discussed",
        "Preserve the logical flow of thinking and progression of ideas throughout the conversation",
        "Maintain all technical details, specific examples, numbers, and concrete information",
        "Identify and attribute speakers by name when context allows",
        "Organize the content logically while preserving the chronological flow of the discussion",
        "Focus on substance and ideas rather than conversational style or atmosphere",
        "Ensure no important information or insights are lost from the original conversation"
    ]
    
    task_instruction = "Please create a comprehensive summary following these guidelines:\n" + "\n".join(f"{i+1}. {goal}" for i, goal in enumerate(summarization_goals))
    output_request = "Please provide a detailed summary that preserves all important information and the flow of thinking, but presents it in a clear, organized manner without the conversational style."
    
    # Build the complete prompt
    prompt = f"""{prompt_header}

{task_instruction}{meeting_context}{language_context}

Original transcript:
{transcription}

{output_request}"""

    print(f"Summarizing conversation with {GPT_MODEL_DISPLAY_NAME}...")
    print("This may take a moment...")
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at creating comprehensive summaries of conversations and meetings. Focus on preserving all important information, technical details, and the logical flow of ideas while presenting them in a clear, organized manner."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            reasoning_effort="high",
        )
        
        summary_text = response.choices[0].message.content
        print(f"{GPT_MODEL_DISPLAY_NAME} summarization completed successfully!")
        return summary_text
        
    except Exception as e:
        print(f"Warning: {GPT_MODEL_DISPLAY_NAME} summarization failed: {e}")
        print("Returning original transcription...")
        return transcription


def convert_video_to_audio(video_file_path: str, temp_dir: str = None, max_duration_minutes: float = None) -> str:
    """
    Convert video file to audio using ffmpeg
    
    Args:
        video_file_path: Path to the video file
        temp_dir: Optional temporary directory for audio file
        max_duration_minutes: Optional maximum duration in minutes to extract
    
    Returns:
        Path to the converted audio file
    """
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Create temporary audio file path
    video_path = Path(video_file_path)
    temp_audio_path = os.path.join(temp_dir, f"{video_path.stem}_temp_audio.mp3")
    
    print(f"Converting video to audio: {video_file_path}")
    print(f"Temporary audio file: {temp_audio_path}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg to convert video files.")
    
    # Convert video to audio using ffmpeg
    cmd = ['ffmpeg', '-i', video_file_path]
    
    # Add duration limit if specified
    if max_duration_minutes is not None:
        duration_seconds = max_duration_minutes * 60
        cmd.extend(['-t', str(duration_seconds)])
    
    cmd.extend([
        '-vn',  # No video
        '-acodec', 'mp3',  # Audio codec
        '-ab', '192k',  # Audio bitrate
        '-ar', '44100',  # Audio sample rate
        '-y',  # Overwrite output file
        temp_audio_path
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        print("Video conversion completed successfully!")
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert video to audio: {e.stderr}")

def detect_audio_language(audio_file_path: str, detector_size: str = DEFAULT_DETECTOR_SIZE) -> tuple[str, float]:
    """
    Detect the language of an audio file using Whisper
    
    Args:
        audio_file_path: Path to the audio file
        detector_size: Whisper model size for detection (default: base for better accuracy)
    
    Returns:
        Tuple of (language_code, confidence) e.g., ('he', 0.85)
    """
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
    
    print(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
    
    # Show top 3 language predictions
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Top language predictions:")
    for lang, prob in sorted_probs:
        print(f"  {lang}: {prob:.3f}")
    
    # Check confidence threshold
    if confidence < MIN_LANGUAGE_CONFIDENCE:
        print(f"Warning: Language detection confidence ({confidence:.2f}) is below threshold ({MIN_LANGUAGE_CONFIDENCE:.2f})")
        print("Consider manually specifying the language using the --language parameter")
    
    return detected_language, confidence

def process_audio_with_duration_limit(audio_file_path: str, max_duration_minutes: float, temp_dir: str = None) -> str:
    """
    Process audio file to extract only the first N minutes using ffmpeg
    
    Args:
        audio_file_path: Path to the audio file
        max_duration_minutes: Maximum duration in minutes to extract
        temp_dir: Optional temporary directory for processed audio file
    
    Returns:
        Path to the processed audio file
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Create temporary processed audio file path
    audio_path = Path(audio_file_path)
    temp_processed_path = os.path.join(temp_dir, f"{audio_path.stem}_processed_{max_duration_minutes}min.mp3")
    
    print(f"Extracting first {max_duration_minutes} minutes from audio file...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg to process audio files.")
    
    # Extract duration-limited audio using ffmpeg
    duration_seconds = max_duration_minutes * 60
    cmd = [
        'ffmpeg', '-i', audio_file_path,
        '-t', str(duration_seconds),  # Duration limit
        '-acodec', 'mp3',  # Audio codec
        '-ab', '192k',  # Audio bitrate
        '-ar', '44100',  # Audio sample rate
        '-y',  # Overwrite output file
        temp_processed_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        print(f"Audio processing completed successfully! Limited to {max_duration_minutes} minutes.")
        return temp_processed_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to process audio file: {e.stderr}")

def transcribe_audio(audio_file_path: str, output_file_path: str = None, model_size: str = DEFAULT_MODEL_SIZE, original_input_file: str = None, language: str = None, max_duration_minutes: float = None, use_gpt_refinement: bool = True, meeting_description: str = None) -> str:
    """
    Transcribe audio file to text using Whisper and optionally create a detailed summary using GPT
    
    Args:
        audio_file_path: Path to the audio file (should be audio format, not video)
        output_file_path: Optional path to save transcription (defaults to same name with .txt extension)
        model_size: Whisper model size (tiny, base, small, medium, large)
        original_input_file: Optional path to original input file (for display in output header)
        language: Language code (e.g., 'he', 'en'). If None, will auto-detect language
        max_duration_minutes: Optional maximum duration in minutes to transcribe
        use_gpt_refinement: Whether to create a summary of the transcription using GPT (default: True)
        meeting_description: Optional description of the meeting context for better GPT summarization
    
    Returns:
        Transcribed text (summarized with GPT if enabled, otherwise raw transcription)
    """
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Process audio file with duration limit if specified
    temp_processed_file = None
    actual_audio_file = audio_file_path
    
    if max_duration_minutes is not None:
        print(f"Limiting transcription to first {max_duration_minutes} minutes...")
        actual_audio_file = process_audio_with_duration_limit(audio_file_path, max_duration_minutes)
        temp_processed_file = actual_audio_file  # Keep track for cleanup
    
    try:
        # Detect language if not provided (using the processed audio file if duration was limited)
        if language is None:
            language, confidence = detect_audio_language(actual_audio_file)
        else:
            print(f"Using specified language: {language}")
            confidence = None
    
        print(f"Loading Whisper model ({model_size})...")
        print("(This may download the model if not already cached)")
        model = whisper.load_model(model_size)
        
        duration_info = f" (first {max_duration_minutes} minutes only)" if max_duration_minutes else ""
        print(f"Transcribing audio file: {audio_file_path}{duration_info}")
        print("This may take a few minutes depending on the file size...")
    
        # Define transcription parameters (only non-defaults)
        transcribe_params = {
            'language': language,  # Use detected or specified language
            'best_of': 2,  # Try multiple attempts and pick best (default: 5)
            'beam_size': 3,  # Use beam search for better accuracy (default: 5)
            'patience': 1.0,  # Wait longer for better results (default: None)
            'verbose': True,  # Show progress (default: False)
        }
        
        # Print parameters being used
        param_summary = ", ".join(f"{k}={v}" for k, v in transcribe_params.items() if k != 'verbose')
        print(f"Using parameters: {param_summary}")
        
        # Transcribe using the actual audio file (which may be duration-limited)
        result = model.transcribe(actual_audio_file, **transcribe_params)
    
        # Extract the transcribed text
        transcribed_text = result["text"]
        
        # Summarize conversation with GPT if enabled
        if use_gpt_refinement:
            summary_text = summarize_conversation_with_gpt(transcribed_text, language, meeting_description)
        else:
            summary_text = transcribed_text
            print("Skipping GPT summarization as requested.")
        
        # Determine output file path
        if output_file_path is None:
            base_name = os.path.splitext(audio_file_path)[0]
            output_file_path = f"{base_name}_transcription.txt"
        
        # Save transcription to file
        with open(output_file_path, 'w', encoding='utf-8') as f:
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
            params_str = ", ".join(f"{k}={v}" for k, v in transcribe_params.items() if k != 'verbose')
            f.write(f"Parameters: {params_str}\n")
            
            # Add GPT summarization information
            f.write(f"{GPT_MODEL_DISPLAY_NAME} summarization: {'enabled' if use_gpt_refinement else 'disabled'}\n")
            
            # Add meeting description if provided
            if meeting_description:
                f.write(f"Meeting description: {meeting_description}\n")
            
            f.write("=" * 50 + "\n\n")
            
            # Write summary if GPT was used, otherwise original transcription
            if use_gpt_refinement:
                f.write(f"=== CONVERSATION SUMMARY ({GPT_MODEL_DISPLAY_NAME}) ===\n\n")
                f.write(summary_text)
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("=== ORIGINAL TRANSCRIPTION (Whisper) ===\n\n")
                f.write(transcribed_text)
            else:
                f.write(transcribed_text)
        
        print(f"Processing completed!")
        print(f"Output saved to: {output_file_path}")
        
        # Show preview of final result (summary if GPT was used)
        final_text = summary_text if use_gpt_refinement else transcribed_text
        if use_gpt_refinement:
            print(f"Final summary preview (first 200 characters):")
        else:
            print(f"Final transcription preview (first 200 characters):")
        print(f"{final_text[:200]}...")
        
        return final_text
    
    finally:
        # Clean up temporary processed file if it was created
        if temp_processed_file and os.path.exists(temp_processed_file):
            try:
                os.remove(temp_processed_file)
                print(f"Cleaned up temporary processed file: {temp_processed_file}")
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_processed_file}: {e}")


def main():
    # Show usage if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python transcribe_audio.py [input_file] [model_size] [--language LANG] [--max-duration MINUTES] [--description DESC] [--no-gpt]")
        print("  input_file: Path to audio (.mp3, .wav, etc.) or video (.mp4, .avi, etc.) file")
        print(f"  model_size: Whisper model size (default: {DEFAULT_MODEL_SIZE})")
        print("  --language: Language code (e.g., 'he', 'en', 'fr') to skip auto-detection")
        print("  --max-duration: Maximum duration in minutes to transcribe (e.g., 1, 5, 10.5)")
        print("  --description: Meeting description for better GPT summarization context")
        print(f"  --no-gpt: Disable {GPT_MODEL_DISPLAY_NAME} summarization and use only Whisper transcription")
        print("  Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v")
        print("  Requires ffmpeg for video file conversion")
        print(f"  Requires OPENAI_API_KEY environment variable for {GPT_MODEL_DISPLAY_NAME} summarization")
        print("\nExamples:")
        print("  python transcribe_audio.py recording.mp4")
        print("  python transcribe_audio.py recording.mp4 large-v3-turbo")
        print("  python transcribe_audio.py recording.mp4 --description 'Weekly team standup discussing project progress'")
        print("  python transcribe_audio.py recording.mp4 --language he --description 'Technical discussion about machine learning models'")
        print("  python transcribe_audio.py recording.mp4 --max-duration 1 --description 'Quick client call about requirements'")
        print("  python transcribe_audio.py recording.mp4 --no-gpt")
        sys.exit(0)
    
    # Check if input file provided
    if len(sys.argv) < 2:
        print("Error: Please provide an input file")
        print("Usage: python transcribe_audio.py [input_file] [model_size] [--language LANG] [--max-duration MINUTES] [--description DESC] [--no-gpt]")
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
        if arg == '--language':
            if i + 1 >= len(sys.argv):
                print("Error: --language requires a language code")
                sys.exit(1)
            language = sys.argv[i + 1]
            i += 2
        elif arg == '--max-duration':
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
        elif arg == '--description':
            if i + 1 >= len(sys.argv):
                print("Error: --description requires a meeting description")
                sys.exit(1)
            meeting_description = sys.argv[i + 1]
            i += 2
        elif arg == '--no-gpt':
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
        if is_video:
            print(f"Detected video file: {input_file}")
            # Convert video to audio (with optional duration limit)
            audio_file = convert_video_to_audio(input_file, max_duration_minutes=max_duration_minutes)
            temp_audio_file = audio_file  # Keep track for cleanup
        else:
            print(f"Detected audio file: {input_file}")
            audio_file = input_file

        # Determine output file path based on original input file
        base_name = os.path.splitext(input_file)[0]
        output_file_path = f"{base_name}_transcription.txt"
        
        print("Starting transcription and processing...")
        processed_text = transcribe_audio(
            audio_file, 
            output_file_path=output_file_path, 
            model_size=model_size,
            original_input_file=input_file,
            language=language,
            max_duration_minutes=max_duration_minutes,
            use_gpt_refinement=use_gpt_refinement,
            meeting_description=meeting_description
        )

        if use_gpt_refinement:
            print(f"\nFull summary length: {len(processed_text)} characters")
        else:
            print(f"\nFull transcription length: {len(processed_text)} characters")
        
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
