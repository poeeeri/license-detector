from __future__ import annotations

import json
import mimetypes
import os
import re
from base64 import b64encode
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    PLATE_JSON_SCHEMA,
)


OCR_PROMPT = (
    "Read the license plate text in this crop. "
    "text_raw should preserve visible spaces or separators when clear. "
    "text_normalized should be uppercase and contain only A-Z, 0-9, "
    "Cyrillic uppercase letters, and \\u0401. "
    "If unreadable, return empty text fields and explain why."
)


def build_client(provider: str, model_arg: str | None) -> tuple[OpenAI, str]:
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        model = model_arg or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        return (
            OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                    "X-Title": os.getenv("OPENROUTER_APP_NAME", "license-plate-ocr-labeler"),
                },
            ),
            model,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model = model_arg or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    return OpenAI(api_key=api_key), model


def read_plate_text(
    client: OpenAI,
    provider: str,
    model: str,
    crop_path: Path,
    detail: str,
) -> dict[str, Any]:
    if provider == "openrouter":
        return read_plate_text_openrouter(client, model, crop_path, detail)
    return read_plate_text_openai(client, model, crop_path, detail)


def read_plate_text_openai(
    client: OpenAI,
    model: str,
    crop_path: Path,
    detail: str,
) -> dict[str, Any]:
    image_url = build_image_data_url(crop_path)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You create OCR training labels from cropped vehicle license plates. "
                            "Return only data that is visible in the image. Do not guess."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": OCR_PROMPT},
                    {"type": "input_image", "image_url": image_url, "detail": detail},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "license_plate_ocr_label",
                "schema": PLATE_JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    raw_output = response.output_text.strip()
    payload = json.loads(raw_output)
    payload["raw_output"] = raw_output
    return payload


def read_plate_text_openrouter(
    client: OpenAI,
    model: str,
    crop_path: Path,
    detail: str,
) -> dict[str, Any]:
    image_url = build_image_data_url(crop_path)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create OCR training labels from cropped vehicle license plates. "
                    "Return only data that is visible in the image. Do not guess."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": detail},
                    },
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "license_plate_ocr_label",
                "strict": True,
                "schema": PLATE_JSON_SCHEMA,
            },
        },
    )
    raw_output = response.choices[0].message.content or ""
    payload = parse_json_payload(raw_output)
    payload["raw_output"] = raw_output.strip()
    return payload


def build_image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"
    image_base64 = b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"


def parse_json_payload(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)