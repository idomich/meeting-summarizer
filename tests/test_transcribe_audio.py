#!/usr/bin/env python3
"""
Unit tests for transcribe_audio.py

Tests key functionality including:
- Utility functions (formatting, timestamps)
- Prompt building for Claude summarization
- Error handling and validation
- Mock tests for external dependencies (Claude CLI, Whisper)
"""

import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

# Add the parent directory to the path to import transcribe_audio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transcribe_audio  # noqa: E402


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions that don't require external dependencies"""

    def test_format_file_size(self):
        """Test file size formatting"""
        self.assertEqual(transcribe_audio.format_file_size(512), "512 B")
        self.assertEqual(transcribe_audio.format_file_size(1536), "1.5 KB")
        self.assertEqual(transcribe_audio.format_file_size(2097152), "2.0 MB")
        self.assertEqual(transcribe_audio.format_file_size(3221225472), "3.0 GB")

    def test_format_duration(self):
        """Test duration formatting"""
        self.assertEqual(transcribe_audio.format_duration(0.5), "500ms")
        self.assertEqual(transcribe_audio.format_duration(15.7), "15.7s")
        self.assertEqual(transcribe_audio.format_duration(125.3), "2m 5.3s")
        self.assertEqual(transcribe_audio.format_duration(3665.0), "61m 5.0s")


class TestStructuredTranscription(unittest.TestCase):
    """Test transcription formatting functions"""

    def test_format_structured_transcription_no_segments(self):
        """Test formatting when no segments are present"""
        result = {"text": "text"}
        formatted = transcribe_audio.format_structured_transcription(result)
        self.assertEqual(formatted, "text")

    def test_format_structured_transcription_with_segments(self):
        """Test formatting with timing and confidence data"""
        # Minimal segment with only required fields
        base_segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " test",
            "avg_logprob": -0.5,
            "no_speech_prob": 0.1,
        }

        result = {
            "segments": [
                base_segment,
                {
                    **base_segment,
                    "start": 1.0,
                    "end": 2.0,
                    "no_speech_prob": 0.8,
                },  # High no-speech
            ]
        }

        formatted = transcribe_audio.format_structured_transcription(result)

        # Should include header
        self.assertIn("## Structured Transcription with Timing & Confidence", formatted)

        # Should include speaking segment
        self.assertIn("[00:00.00 --> 00:01.00]", formatted)
        self.assertIn("test", formatted)

        # Should filter out high no-speech segment
        self.assertNotIn("[00:01.00 --> 00:02.00]", formatted)

        # Should include confidence scores
        self.assertIn("confidence:", formatted)


class TestClaudeSummarization(unittest.TestCase):
    """Test Claude Code CLI summarization with mocked subprocess calls"""

    @patch("transcribe_audio.run_claude_cli_prompt", return_value="summary")
    def test_summarize_conversation_success(self, mock_run_claude):
        """Test successful Claude summarization"""
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " test",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.1,
        }
        transcription_data = {"text": "test", "segments": [segment]}

        result = transcribe_audio.summarize_conversation(
            transcription_data,
            language_detected="he",
            meeting_description="meet",
        )

        self.assertEqual(result, "summary")
        mock_run_claude.assert_called_once()
        prompt = mock_run_claude.call_args[0][0]
        self.assertIn("meet", prompt)
        self.assertIn("language 'he'", prompt)
        self.assertIn("timing information", prompt)

    @patch("transcribe_audio.run_claude_cli_prompt", side_effect=RuntimeError("CLI Error"))
    def test_summarize_conversation_failure_raises(self, mock_run_claude):
        """Test that summarization propagates Claude CLI failures after retries"""
        del mock_run_claude
        transcription_data = {"text": "test", "segments": []}

        with self.assertRaises(Exception):
            transcribe_audio.summarize_conversation(transcription_data)

    @patch("transcribe_audio.run_claude_cli_prompt", return_value="Summary")
    def test_prompt_construction_elements(self, mock_run_claude):
        """Test that Claude prompts contain required elements"""
        transcribe_audio.summarize_conversation(
            "Test transcription", language_detected="he", meeting_description="meet"
        )

        prompt = mock_run_claude.call_args[0][0]
        for element in ("summary", "meet", "language 'he'"):
            self.assertIn(element, prompt, f"Missing expected element: {element}")


class TestExtractTranscription(unittest.TestCase):
    """Test extraction helpers for existing transcription files"""

    def test_prefer_whisper_original_over_cleaned(self):
        content = """# Transcription: meeting.mp4

**Language:** he

---

## Cleaned & Translated Transcription (Claude Sonnet 5)

cleaned text

## Original Transcription with Timing & Confidence (Whisper)

## Structured Transcription with Timing & Confidence

whisper text
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name

        try:
            text, _, _ = transcribe_audio.extract_transcription_from_file(
                path, prefer_whisper_original=True
            )
            self.assertIn("whisper text", text)
            self.assertIn("Structured Transcription", text)

            text_default, _, _ = transcribe_audio.extract_transcription_from_file(path)
            self.assertEqual(text_default, "cleaned text")
        finally:
            os.remove(path)


class TestRefineTranscription(unittest.TestCase):
    """Test re-refine flow with mocked Claude calls"""

    @patch("transcribe_audio.summarize_conversation", return_value="summary")
    @patch("transcribe_audio.clean_and_translate_transcription", return_value="cleaned")
    @patch("transcribe_audio.write_transcription_markdown_file")
    def test_refine_transcription_overwrites_by_default(
        self, mock_write, mock_clean, mock_summarize
    ):
        content = """# Transcription: meeting.mp4

**Model used:** large-v3-turbo
**Language:** he
**Parameters:** `language=he, best_of=2`
**Streaming transcription:** 2 chunks processed using 2 processes
**Meeting description:** team sync

---

## Original Transcription with Timing & Confidence (Whisper)

whisper body
"""
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name

        try:
            summary, output_path = transcribe_audio.refine_transcription(path)

            self.assertEqual(summary, "summary")
            self.assertEqual(output_path, path)
            mock_clean.assert_called_once_with("whisper body", "he")
            mock_summarize.assert_called_once()
            mock_write.assert_called_once()
            self.assertEqual(mock_write.call_args.kwargs["whisper_markdown"], "whisper body")
        finally:
            os.remove(path)


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing"""

    def test_parse_arguments_minimal(self):
        """Test parsing with minimal required arguments"""
        with patch("sys.argv", ["transcribe_audio.py", "test.mp3"]):
            with patch("os.path.exists", return_value=True):
                args = transcribe_audio.parse_arguments()

                self.assertEqual(args.input_file, "test.mp3")
                self.assertEqual(args.model, transcribe_audio.DEFAULT_MODEL_SIZE)
                self.assertIsNone(args.language)
                self.assertIsNone(args.max_duration)
                self.assertIsNone(args.description)
                self.assertFalse(args.no_gpt)

    def test_parse_arguments_full(self):
        """Test parsing with all arguments"""
        test_args = [
            "transcribe_audio.py",
            "test.mp4",
            "--model",
            "large-v3-turbo",
            "--language",
            "he",
            "--max-duration",
            "5.5",
            "--description",
            "meet",
            "--no-gpt",
            "--log-level",
            "DEBUG",
        ]

        with patch("sys.argv", test_args):
            with patch("os.path.exists", return_value=True):
                args = transcribe_audio.parse_arguments()

                self.assertEqual(args.input_file, "test.mp4")
                self.assertEqual(args.model, "large-v3-turbo")
                self.assertEqual(args.language, "he")
                self.assertEqual(args.max_duration, 5.5)
                self.assertEqual(args.description, "meet")
                self.assertTrue(args.no_gpt)
                self.assertEqual(args.log_level, "DEBUG")

    def test_parse_arguments_file_not_found(self):
        """Test error handling when input file doesn't exist"""
        with patch("sys.argv", ["transcribe_audio.py", "nonexistent.mp3"]):
            with patch("os.path.exists", return_value=False):
                with self.assertSystemExit():
                    transcribe_audio.parse_arguments()

    def test_parse_arguments_invalid_duration(self):
        """Test error handling for invalid duration"""
        test_args = ["transcribe_audio.py", "test.mp3", "--max-duration", "-5"]

        with patch("sys.argv", test_args):
            with patch("os.path.exists", return_value=True):
                with self.assertSystemExit():
                    transcribe_audio.parse_arguments()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and validation"""

    def test_transcribe_audio_file_not_found(self):
        """Test error handling when audio file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            transcribe_audio.transcribe_audio("nonexistent_file.mp3")

    @patch("transcribe_audio.os.path.exists", return_value=True)
    @patch("transcribe_audio.os.path.getsize", return_value=1000)  # Mock file size
    @patch("transcribe_audio.whisper.load_model")
    @patch("transcribe_audio.trim_silence_from_audio")
    @patch("transcribe_audio.detect_audio_language")
    def test_transcribe_audio_basic_flow(
        self, mock_detect_lang, mock_trim, mock_load_model, mock_getsize, mock_exists
    ):
        """Test basic transcribe_audio flow with mocked dependencies"""
        # Setup mocks
        mock_detect_lang.return_value = ("en", 0.9)
        mock_trim.return_value = "test.mp3"  # Return same file to avoid cleanup issues

        mock_model = Mock()
        # Minimal mock result
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " test",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.1,
        }
        mock_result = {"text": "test", "segments": [segment]}
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.mp3")

            with patch("builtins.open", mock_open()) as mock_file:
                with patch("transcribe_audio.os.remove"):  # Mock file cleanup
                    result_text, output_path = transcribe_audio.transcribe_audio(
                        test_file, use_gpt_refinement=False  # Skip LLM to avoid Claude CLI mocking
                    )

                # Check that key functions were called
                mock_detect_lang.assert_called_once()
                mock_trim.assert_called_once()
                mock_load_model.assert_called_once()
                mock_model.transcribe.assert_called_once()

                # Check that output file was written
                mock_file.assert_called()

                # Check that we got expected results
                self.assertIn("test", result_text)


def assertSystemExit(self):
    """Context manager to assert SystemExit is raised"""
    return self.assertRaises(SystemExit)


# Add the assertSystemExit method to TestCase
unittest.TestCase.assertSystemExit = assertSystemExit


if __name__ == "__main__":
    # Configure logging to avoid noise during tests
    logging.getLogger("transcribe_audio").setLevel(logging.CRITICAL)

    # Run tests
    unittest.main(verbosity=2)
