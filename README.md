![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)

# Meeting Summarizer

This repository is a POC for transcribing and summarizing Hebrew/English meeting recordings for use with RAG (Retrieval Augmented Generation) systems.

## Features

- Audio and video file transcription using OpenAI Whisper
- Support for Hebrew and English languages  
- Automatic video-to-audio conversion using ffmpeg
- Optimized for meeting recording formats
- Structured output for RAG integration
- GPT-5 meeting summary generation (enabled by default; can be disabled)

## Requirements

- Python 3.11+
- ffmpeg (for video file conversion)
- OpenAI Whisper
- OpenAI Python SDK (>= 1.0.0) for summarization
- OPENAI_API_KEY environment variable for summarization

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key for summarization:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

Basic usage:
```bash
python transcribe_audio.py [input_file] [model_size] [--language LANG] [--max-duration MINUTES] [--description DESC] [--no-gpt]
```

Options:
- input_file: Path to audio (.mp3, .wav, etc.) or video (.mp4, .avi, etc.) file
- model_size: Whisper model size (default: large-v3-turbo)
- --language: Language code to skip auto-detection (e.g., 'he', 'en')
- --max-duration: Maximum duration in minutes to transcribe (e.g., 1, 5, 10.5)
- --description: Meeting description for better GPT summarization context
- --no-gpt: Disable GPT-5 summarization and use only Whisper transcription
- Requires ffmpeg for video conversion and OPENAI_API_KEY for summarization

Examples:
```bash
# Transcribe an audio file
python transcribe_audio.py meeting.mp3

# Transcribe a video file with specific model
python transcribe_audio.py meeting.mp4 large-v3-turbo

# Add meeting context for better summaries
python transcribe_audio.py recording.mp4 --description 'Weekly team standup discussing project progress'

# Force language and add context
python transcribe_audio.py recording.mp4 --language he --description 'Technical discussion about machine learning models'

# Limit processing to the first minute
python transcribe_audio.py recording.mp4 --max-duration 1 --description 'Quick client call about requirements'

# Disable summarization and keep the raw transcript
python transcribe_audio.py recording.mp4 --no-gpt

# Show help
python transcribe_audio.py --help
```

Supported formats:
- Audio: .mp3, .wav, .m4a, .flac, etc.
- Video: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v

Output is saved to `<input_basename>_transcription.txt`. With GPT summarization enabled (default), the file contains the generated summary. Use `--no-gpt` to save the raw Whisper transcription instead.
