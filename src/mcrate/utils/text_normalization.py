from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def char_ngrams(text: str, n: int) -> set[str]:
    value = normalize_text(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    lhs = set(left)
    rhs = set(right)
    if not lhs and not rhs:
        return 0.0
    union = lhs | rhs
    if not union:
        return 0.0
    return len(lhs & rhs) / len(union)


def longest_common_substring_fraction(left: str, right: str) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                value = prev[j - 1] + 1
                curr.append(value)
                if value > best:
                    best = value
            else:
                curr.append(0)
        prev = curr
    return best / max(1, len(b))


def digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def normalize_phone(text: str) -> str:
    return digits_only(text)


def normalize_email(text: str) -> str:
    return normalize_text(text)


def normalize_date(text: str) -> str:
    value = normalize_text(text)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def normalize_value(field_name: str, value: str) -> str:
    if "email" in field_name:
        return normalize_email(value)
    if "phone" in field_name:
        return normalize_phone(value)
    if "date" in field_name or "time" in field_name:
        return normalize_date(value)
    return normalize_text(value)


def contains_normalized_value(text: str, value: str, field_name: str = "") -> bool:
    return normalize_value(field_name, value) in normalize_value(field_name, text)


def fuzzy_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def max_common_sensitive_substring(prompt: str, values: list[str]) -> int:
    norm_prompt = normalize_text(prompt)
    best = 0
    for value in values:
        norm_value = normalize_text(value)
        for size in range(min(len(norm_value), 32), 0, -1):
            found = False
            for start in range(0, len(norm_value) - size + 1):
                piece = norm_value[start : start + size]
                if piece and piece in norm_prompt:
                    best = max(best, size)
                    found = True
                    break
            if found:
                break
    return best
