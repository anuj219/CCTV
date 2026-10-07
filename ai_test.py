import argparse
import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cctv.ai_summary import (
    DEFAULT_GROQ_MODEL,
    _frame_to_base64,
    _request_groq_summary,
    _safe_provider_error,
)


DEFAULT_TEST_FRAME = PROJECT_ROOT / "cctv" / "known_faces" / "anuj" / "test_frame.jpg"
TEST_PROMPT = (
    "Describe this home security camera frame briefly. "
    "Report who or what is present, headcount, current activity, and any unusual details. "
    "Use at most two factual, concise sentences."
)


def main():
    parser = argparse.ArgumentParser(description="Test Groq vision models with a CCTV frame.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_TEST_FRAME,
        help=f"Image to test (default: {DEFAULT_TEST_FRAME})",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Groq model to test; repeat the option to test multiple models.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "cctv" / ".env")
    api_key = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: groq_api_key is missing from cctv/.env or the environment.")
        return 2

    image_path = args.image.resolve()
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return 2

    image_b64 = _frame_to_base64(frame)
    if not image_b64:
        print("ERROR: Could not encode the test frame.")
        return 2

    models = args.models or [os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL]
    successful_models = 0
    print(f"Test frame: {image_path}")
    for model_name in models:
        print(f"\nTesting Groq model {model_name}...")
        try:
            response = _request_groq_summary(
                api_key,
                model_name,
                TEST_PROMPT,
                image_b64,
            )
            successful_models += 1
            print("SUCCESS")
            print(response)
        except Exception as exc:
            print("FAILED:", _safe_provider_error(model_name, exc, api_key))

    print(f"\n{successful_models}/{len(models)} model(s) succeeded.")
    return 0 if successful_models else 1


if __name__ == "__main__":
    raise SystemExit(main())