from __future__ import annotations
import re


_PLATE_CHARS_RE = re.compile(r"[^0-9A-ZА-ЯЁ]")


def normalize_plate_text(text):
    return _PLATE_CHARS_RE.sub("", text.upper())