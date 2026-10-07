import io
import json
import os
import sys
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cctv import ai_summary


class AiSummaryLogicTests(unittest.TestCase):
    def test_fallback_summary_includes_event_id(self):
        text = ai_summary.build_fallback_summary(
            reason="no_face",
            headcount=1,
            error="503 Server Error",
            event_id=42,
        )
        self.assertIn("event_id=42", text)
        self.assertIn("reason=no_face", text)

    def test_provider_http_error_includes_sanitized_response_body(self):
        response_body = io.BytesIO(
            b'{"error":{"message":"model access denied for test-secret"}}'
        )
        error = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            403,
            "Forbidden",
            None,
            response_body,
        )

        message = ai_summary._safe_provider_error("Groq", error, "test-secret")

        self.assertIn("model access denied", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("test-secret", message)

    def test_summary_trigger_requires_real_entry_event(self):
        track = {
            "entry_event_id": None,
            "has_approached": True,
            "entry_pending": True,
            "last_ai_summary_at": 0.0,
        }
        self.assertFalse(ai_summary.should_generate_summary(track, now=100.0))

    def test_known_entry_does_not_trigger_summary(self):
        track = {
            "entry_event_id": 55,
            "has_approached": True,
            "entry_pending": True,
            "entry_status": None,
            "locked": True,
            "name": "hiyansh",
            "last_ai_summary_at": 0.0,
        }
        self.assertFalse(ai_summary.should_generate_summary(track, now=100.0))

    def test_confirmed_unknown_entry_waits_for_linger_timeout(self):
        track = {
            "entry_event_id": 55,
            "has_approached": True,
            "entry_pending": False,
            "entry_status": "confirmed_unknown",
            "last_ai_summary_at": 0.0,
        }
        self.assertFalse(ai_summary.should_generate_summary(track, now=100.0))

    def test_summary_trigger_after_unknown_linger_timeout(self):
        track = {
            "entry_event_id": 77,
            "has_approached": True,
            "entry_pending": False,
            "entry_status": None,
            "last_ai_summary_at": 0.0,
            "unknown_linger_started_at": 85.0,
        }
        self.assertTrue(ai_summary.should_generate_summary(track, now=100.0))

    def test_groq_generates_summary_directly(self):
        groq_response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "A visitor is standing at the doorway."}}]
        }).encode("utf-8"))
        environment = {
            "groq_api_key": "test-groq-key",
            "GROQ_API_KEY": "test-groq-key",
        }

        with patch.dict(os.environ, environment), \
             patch.object(ai_summary.urllib_request, "urlopen") as urlopen, \
             patch.object(ai_summary, "create_ai_summary") as save_summary:
            urlopen.return_value = groq_response

            summary = ai_summary.generate_scene_summary(
                81,
                frame=np.zeros((16, 16, 3), dtype=np.uint8),
                headcount=1,
                reason="unknown_linger",
            )

        self.assertEqual(summary, "A visitor is standing at the doorway.")
        urlopen.assert_called_once()
        groq_request = urlopen.call_args.args[0]
        self.assertIn("api.groq.com/openai/v1/chat/completions", groq_request.full_url)
        self.assertEqual(groq_request.get_header("Authorization"), "Bearer test-groq-key")
        self.assertEqual(groq_request.get_header("User-agent"), "NeytraN-CCTV/1.0")
        payload = json.loads(groq_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen/qwen3.8-27b")
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertIn("home security monitoring assistant", parts[0]["text"])
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        save_summary.assert_called_once_with(
            81,
            summary,
            reason="unknown_linger",
            headcount=1,
        )

    def test_missing_groq_key_persists_fallback(self):
        environment = {
            "groq_api_key": "",
            "GROQ_API_KEY": "",
        }
        with patch.dict(os.environ, environment), \
             patch.object(ai_summary, "_request_groq_summary") as groq_request, \
             patch.object(ai_summary, "create_ai_summary") as save_summary:
            summary = ai_summary.generate_scene_summary(
                82,
                frame=np.zeros((16, 16, 3), dtype=np.uint8),
                reason="unknown_linger",
            )

        self.assertIn("missing_groq_api_key", summary)
        groq_request.assert_not_called()
        save_summary.assert_called_once_with(
            82,
            summary,
            reason="unknown_linger",
            headcount=1,
            error="missing_groq_api_key",
        )


if __name__ == "__main__":
    unittest.main()
