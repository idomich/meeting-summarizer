#!/usr/bin/env python3
"""
Unit tests for transcribe_audio.py

Tests key functionality including:
- Utility functions (formatting, timestamps)
- Prompt building for GPT summarization
- Error handling and validation
- Mock tests for external dependencies (OpenAI, Whisper)
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
        self.assertIn("=== STRUCTURED TRANSCRIPTION WITH TIMING & CONFIDENCE ===", formatted)

        # Should include speaking segment
        self.assertIn("[00:00.00 --> 00:01.00]", formatted)
        self.assertIn("test", formatted)

        # Should filter out high no-speech segment
        self.assertNotIn("[00:01.00 --> 00:02.00]", formatted)

        # Should include confidence scores
        self.assertIn("confidence:", formatted)


class TestGPTSummarization(unittest.TestCase):
    """Test GPT summarization with mocked OpenAI calls"""

    @patch("transcribe_audio.OpenAI")
    def test_summarize_conversation_success(self, mock_openai_class):
        """Test successful GPT summarization"""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "summary"
        mock_client.chat.completions.create.return_value = mock_response

        # Minimal test data
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " test",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.1,
        }
        transcription_data = {"text": "test", "segments": [segment]}

        # Call function
        result = transcribe_audio.summarize_conversation_with_gpt(
            transcription_data,
            language_detected="he",  # Use Hebrew to test language context
            meeting_description="meet",
        )

        # Assertions
        self.assertEqual(result, "summary")
        mock_client.chat.completions.create.assert_called_once()

        # Check that the call was made with correct parameters
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], transcribe_audio.GPT_MODEL_NAME)
        self.assertEqual(call_args[1]["reasoning_effort"], "high")

        # Check that prompt includes key elements
        user_message = call_args[1]["messages"][1]["content"]
        self.assertIn("meet", user_message)
        self.assertIn("original audio was in language 'he'", user_message)
        self.assertIn("timing information", user_message)

    @patch("transcribe_audio.OpenAI")
    @patch("transcribe_audio.logger")
    def test_summarize_conversation_failure_fallback(self, mock_logger, mock_openai_class):
        """Test that function falls back to structured transcription on API failure"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Minimal test data
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": " test",
            "avg_logprob": -0.3,
            "no_speech_prob": 0.1,
        }
        transcription_data = {"text": "test", "segments": [segment]}

        # Call function
        result = transcribe_audio.summarize_conversation_with_gpt(transcription_data)

        # Should return formatted structured transcription
        self.assertIn("=== STRUCTURED TRANSCRIPTION WITH TIMING & CONFIDENCE ===", result)
        self.assertIn("test", result)

        # Should log warning
        mock_logger.warning.assert_called()

    def test_prompt_construction_elements(self):
        """Test that GPT prompts contain required elements"""
        # This is tested indirectly through the mock above, but we can also test
        # the prompt construction logic by calling the function and checking
        # the arguments passed to the OpenAI client

        with patch("transcribe_audio.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Summary"
            mock_client.chat.completions.create.return_value = mock_response

            transcribe_audio.summarize_conversation_with_gpt(
                "Test transcription", language_detected="he", meeting_description="meet"
            )

            # Check prompt contains expected elements
            call_args = mock_client.chat.completions.create.call_args
            prompt = call_args[1]["messages"][1]["content"]

            expected_elements = ["summary", "meet", "language 'he'"]

            for element in expected_elements:
                self.assertIn(element, prompt, f"Missing expected element: {element}")


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
                        test_file, use_gpt_refinement=False  # Skip GPT to avoid OpenAI mocking
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
