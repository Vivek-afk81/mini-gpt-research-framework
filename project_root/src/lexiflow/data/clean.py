"""Load, normalize, validate, and deduplicate LexiFlow JSONL data."""

import json
import re
from pathlib import Path


REQUIRED_FIELDS = ("input", "output", "category", "tone")
HONORIFIC_FIXES = (
    (re.compile(r"\bma'?am\b", re.IGNORECASE), "Ma'am"),
    (re.compile(r"\bmadam\b", re.IGNORECASE), "Madam"),
)


def load_records(path):
    """Load one JSON object per line, tolerating blank lines."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            missing = set(REQUIRED_FIELDS) - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing: {sorted(missing)}")
            records.append(record)
    return records


def normalize_input(text):
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_output(text):
    value = re.sub(r"\s+", " ", str(text).strip())
    for pattern, replacement in HONORIFIC_FIXES:
        value = pattern.sub(replacement, value)
    return value


def clean(records):
    """Normalize records and remove duplicate input/output pairs."""
    cleaned = []
    seen = set()
    for record in records:
        item = dict(record)
        item["input"] = normalize_input(item["input"])
        item["output"] = normalize_output(item["output"])
        item["category"] = str(item["category"]).strip()
        item["tone"] = str(item["tone"]).strip()
        if not item["input"] or not item["output"]:
            continue
        key = (
            item["input"].casefold(),
            item["output"].casefold(),
            item["category"],
            item["tone"],
        )
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    for index, item in enumerate(cleaned, 1):
        item["id"] = index
    return cleaned


def save_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")