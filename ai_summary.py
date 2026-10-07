import base64
import json
import os
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
from dotenv import load_dotenv

from cctv.db import create_ai_summary

load_dotenv()   

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


def build_fallback_summary(reason, headcount, error=None, event_id=None):
    summary = f"Recognizing... | reason={reason} | headcount={headcount}"
    if error:
        summary += f" | error={error}"
    if event_id is not None:
        summary += f" | event_id={event_id}"
    return summary


def should_generate_summary(track, now, min_interval_seconds=60, unknown_linger_seconds=15):
    if not isinstance(track, dict):
        return False

    event_id = track.get("entry_event_id")
    if event_id is None:
        return False
    if not track.get("has_approached", False):
        return False

    last_trigger = float(track.get("last_ai_summary_at", 0.0))
    if (now - last_trigger) < min_interval_seconds:
        return False

    status = track.get("entry_status")
    if track.get("locked", False) or status in {"known", "known_suppressed"}:
        return False

    lingering_since = track.get("unknown_linger_started_at")
    if lingering_since is None or (now - float(lingering_since)) < unknown_linger_seconds:
        return False

    print(f"[AI SUMMARY TRIGGER] reason=unknown_linger event_id={event_id} status={status} now={now}")
    return True


def _frame_to_base64(frame):
    if frame is None:
        return ""
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_groq_summary_text(payload):
    try:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(part["text"]).strip()
                for part in content
                if isinstance(part, dict) and part.get("text")
            ).strip()
    except Exception:
        return ""
    return ""


def _safe_provider_error(provider, exc, *secrets):
    message = str(exc)
    if isinstance(exc, urllib_error.HTTPError):
        try:
            response_body = exc.read(4096).decode("utf-8", errors="replace")
            if response_body:
                try:
                    payload = json.loads(response_body)
                    detail = payload.get("error", {}).get("message", response_body)
                except (AttributeError, json.JSONDecodeError):
                    detail = response_body
                message = f"{message}; response={str(detail)[:500]}"
        except Exception:
            pass
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return f"{provider}: {type(exc).__name__}: {message}"


def _request_groq_summary(api_key, model_name, prompt_text, image_b64):
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ],
        }],
        "max_tokens": 300,
    }
    request = urllib_request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "NeytraN-CCTV/1.0",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=25) as response:
        result = json.loads(response.read().decode("utf-8"))
    text = _extract_groq_summary_text(result)
    if not text:
        raise ValueError("empty_response")
    return text


def generate_scene_summary(event_id, frame=None, headcount=1, reason="no_face", prompt=None):
    if event_id is None:
        return None

    if frame is None:
        fallback = build_fallback_summary(reason, headcount, error="no_frame", event_id=event_id)
        create_ai_summary(event_id, fallback, reason=reason, headcount=headcount, error="no_frame")
        return fallback

    groq_api_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        error = "missing_groq_api_key"
        fallback = build_fallback_summary(reason, headcount, error=error, event_id=event_id)
        create_ai_summary(event_id, fallback, reason=reason, headcount=headcount, error=error)
        return fallback

    image_b64 = _frame_to_base64(frame)
    if not image_b64:
        fallback = build_fallback_summary(reason, headcount, error="frame_encoding_failed", event_id=event_id)
        create_ai_summary(event_id, fallback, reason=reason, headcount=headcount, error="frame_encoding_failed")
        return fallback

    prompt_text = prompt or (
        """
        You are a home security monitoring assistant.
        The camera views the Mehta family's private residential front door balcony. 
        BASELINE SETTING (Usual Environment):
        - Location: Residential front door / balcony area of Mehta residence
        - Expected Static Elements: Potted plants, regular shoe pairs on the floor, standard entryway layout.
        - Ignore describing expected static decor/furniture unless they are altered (e.g., knocked over, broken, missing, out of place).

        TASK INSTRUCTIONS:
        1. Identify who is present, headcount, clothing/appearance, and their current action.
        2. Check timestamp/lighting to evaluate context (e.g., late night, unescorted child, unknown person).
        3. Identify ONLY dynamic changes or true anomalies compared to the Baseline Setting (e.g., knocked-over plants, forced entry signs, unusual behavior, or unescorted individuals).
        4. Output at most two concise, factual sentences emphasizing key security findings without re-describing the expected home background.
        """
    )

    model_name = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    print(f"[AI SUMMARY] Generating scene summary with Groq ({model_name})")
    try:
        text = _request_groq_summary(
            groq_api_key,
            model_name,
            prompt_text,
            image_b64,
        )
        create_ai_summary(event_id, text, reason=reason, headcount=headcount)
        return text
    except Exception as exc:  # pragma: no cover - network path
        error = _safe_provider_error(f"Groq/{model_name}", exc, groq_api_key)
    fallback = build_fallback_summary(reason, headcount, error=error, event_id=event_id)
    create_ai_summary(event_id, fallback, reason=reason, headcount=headcount, error=error)
    return fallback
