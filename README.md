![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)

# Meeting Summarizer

This repository is a POC for transcribing and summarizing Hebrew/English meeting recordings for use with RAG (Retrieval Augmented Generation) systems.

## Features

- Audio and video file transcription using OpenAI Whisper
- Support for Hebrew and English languages  
- Automatic video-to-audio conversion using ffmpeg
- Optimized for meeting recording formats
- Structured output for RAG integration

## Requirements

- Python 3.11+
- ffmpeg (for video file conversion)
- OpenAI Whisper

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python transcribe_audio.py <input_file> [model_size]
```

Examples:
```bash
# Transcribe an audio file
python transcribe_audio.py meeting.mp3

# Transcribe a video file with specific model
python transcribe_audio.py meeting.mp4 large-v3-turbo

# Show help
python transcribe_audio.py --help
```

Supported formats:
- Audio: .mp3, .wav, .m4a, .flac, etc.
- Video: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v

The transcription will be saved as a .txt file with the same name as the input file.
