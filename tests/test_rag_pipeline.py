"""
Unit Tests for RAG Generation, Multi-turn History, and Model Fallbacks
======================================================================
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure D:\Rag is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate import generate_answer, FALLBACK_MODELS, GENERATION_MODEL
from retrieve import retrieve


class TestRagGeneration(unittest.TestCase):
    def test_fallback_models_are_valid_google_genai_tags(self):
        """Ensure no deprecated or non-existent 3.x tags exist in fallback models."""
        valid_models = {
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-flash-latest",
        }
        for model in FALLBACK_MODELS:
            self.assertIn(model, valid_models)
            self.assertFalse(model.startswith("gemini-3."))

    @patch("generate.init_gemini_client")
    def test_multi_turn_history_starts_with_user_turn(self, mock_init_client):
        """Google GenAI requires the first content turn in multi-turn chat to be 'user'."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer from model"
        mock_client.models.generate_content.return_value = mock_response
        mock_init_client.return_value = mock_client

        # History starts with an assistant welcome message
        history = [
            {"role": "assistant", "content": "Welcome to PulseFit AI Coach!"},
            {"role": "user", "content": "What is progressive overload?"},
            {"role": "assistant", "content": "Progressive overload means increasing demand over time."},
        ]

        result = generate_answer(
            query="How should I apply it to squats?",
            context="Squat progression guidelines",
            gemini=mock_client,
            history=history,
        )

        self.assertEqual(result, "Answer from model")
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        contents = call_kwargs["contents"]

        self.assertIsInstance(contents, list)
        # Verify the first turn has role 'user'
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "What is progressive overload?")
        # Verify turn 2 is model
        self.assertEqual(contents[1].role, "model")
        # Verify last turn is user
        self.assertEqual(contents[-1].role, "user")
        self.assertIn("How should I apply it to squats?", contents[-1].parts[0].text)

    @patch("generate.init_gemini_client")
    def test_history_with_only_assistant_welcome_drops_safely(self, mock_init_client):
        """If history only contains the initial assistant greeting, it drops cleanly so query is turn 1."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer from model"
        mock_client.models.generate_content.return_value = mock_response

        history = [
            {"role": "assistant", "content": "Welcome to PulseFit AI Coach!"},
        ]

        result = generate_answer(
            query="Hello Coach",
            context="Some context",
            gemini=mock_client,
            history=history,
        )

        self.assertEqual(result, "Answer from model")
        call_kwargs = mock_client.models.generate_content.call_args[1]
        contents = call_kwargs["contents"]
        # When only assistant turn existed, contents falls back to single user prompt
        if isinstance(contents, list):
            self.assertEqual(len(contents), 1)
            self.assertEqual(contents[0].role, "user")
        else:
            self.assertIsInstance(contents, str)

    @patch("generate.init_gemini_client")
    def test_model_fallback_on_api_failure(self, mock_init_client):
        """If first model fails, generate_answer falls back to next available model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Fallback success"

        # Fail first model, succeed second
        mock_client.models.generate_content.side_effect = [
            RuntimeError("404 Model not found"),
            mock_response,
        ]

        result = generate_answer(
            query="Test query",
            context="Test context",
            gemini=mock_client,
        )

        self.assertEqual(result, "Fallback success")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)


class TestRetrieveMetadata(unittest.TestCase):
    @patch("retrieve.init_qdrant_client")
    @patch("retrieve.init_gemini_client")
    def test_retrieve_empty_query_returns_metadata_tuple(self, mock_gemini, mock_qdrant):
        res = retrieve("", return_metadata=True)
        self.assertEqual(res, ("", []))

    @patch("retrieve.init_qdrant_client")
    @patch("retrieve.init_gemini_client")
    def test_retrieve_empty_query_returns_string_by_default(self, mock_gemini, mock_qdrant):
        res = retrieve("")
        self.assertEqual(res, "")


if __name__ == "__main__":
    unittest.main()
