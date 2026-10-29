"""Hinglish translation and category detection with local fallback."""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from ..data.clean import load_records, normalize_input
from ..llm.groq_client import chat


CATEGORIES = (
    "leave_request", "examination_query", "complaint_request",
    "assignment_query", "attendance_request", "deadline_extension",
)
KEYWORDS = {
    "leave_request": ("leave", "chhutti", "chutti", "fever", "bukhar", "doctor", "absent", "hospital"),
    "examination_query": ("exam", "paper", "test", "viva", "admit card", "syllabus", "result", "marks"),
    "complaint_request": ("not working", "kaam nahi", "kharab", "error", "portal", "login", "wifi", "lab"),
    "assignment_query": ("assignment", "file", "submit", "format", "topic", "pages", "project", "report"),
    "attendance_request": ("attendance", "absent", "condonation", "present", "attendance sheet"),
    "deadline_extension": ("deadline", "extend", "extra time", "late", "adhura", "submit nahi", "days aur"),
}


def detect_category(text):
    value = text.casefold()
    scores = {category: sum(1 for keyword in words if keyword in value) for category, words in KEYWORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "assignment_query"


def _fallback(text, category):
    templates = {
        "leave_request": "I would like to request leave due to personal circumstances.",
        "examination_query": "Could you please provide information about the examination?",
        "complaint_request": "I would like to report an issue and request assistance.",
        "assignment_query": "Could you please provide the assignment details?",
        "attendance_request": "I would like to request clarification regarding my attendance.",
        "deadline_extension": "I respectfully request an extension for the submission deadline.",
    }
    return templates[category]


class Translator:
    def __init__(self, dataset_path="data/campus_messages_v2.jsonl", backend=None):
        self.backend = backend or os.getenv("TRANSLATOR_BACKEND", "local")
        self.records = []
        path = Path(dataset_path)
        if path.exists():
            self.records = load_records(path)
        processed = Path("data/processed/train.jsonl")
        if not self.records and processed.exists():
            self.records = load_records(processed)

    def _local(self, text, category):
        normalized = normalize_input(text).casefold()
        best = None
        best_score = 0.0
        for record in self.records:
            candidate = normalize_input(record["input"]).casefold()
            score = SequenceMatcher(None, normalized, candidate).ratio()
            if normalized in candidate or candidate in normalized:
                score += 0.2
            if score > best_score:
                best_score, best = score, record
        if best and best_score >= 0.62:
            return best["output"], best.get("category", category), best.get("tone", "formal")
        return _fallback(text, category), category, "formal"

    def translate(self, text):
        text = normalize_input(text)
        if not text:
            raise ValueError("Message cannot be empty")
        category = detect_category(text)
        if self.backend == "groq":
            raw = chat([
                {"role": "system", "content": "Convert Hinglish campus messages to formal English. Return JSON only with english, category, tone."},
                {"role": "user", "content": text},
            ], json_mode=True)
            if raw:
                try:
                    result = json.loads(raw)
                    if result.get("english") and result.get("category") in CATEGORIES:
                        return result
                except (json.JSONDecodeError, TypeError):
                    pass
        english, category, tone = self._local(text, category)
        return {"english": english, "category": category, "tone": tone}