"""Grounded email drafting with a deterministic offline fallback."""

import json
import os

from ..llm.groq_client import chat


def draft(english, category, tone, hits, profile=None, backend="local"):
    profile = profile or {}
    if backend == "groq":
        excerpts = "\n".join(f"[P{i + 1}] {hit['text']}" for i, hit in enumerate(hits)) or "NO POLICY EXCERPTS FOUND"
        raw = chat([
            {"role": "system", "content": "Draft a concise formal college email. Use only supplied policy excerpts and cite claims as [P1]. If no policy applies, say so."},
            {"role": "user", "content": json.dumps({"request": english, "category": category, "tone": tone, "profile": profile, "policies": excerpts})},
        ], model=os.getenv("GROQ_DRAFTER_MODEL", "qwen/qwen3-32b"), max_tokens=300)
        if raw:
            return raw

    greeting = "Ma'am" if "ma'am" in english.casefold() else "Sir/Madam"
    name = profile.get("name") or "the student"
    citation = " Please refer to " + ", ".join(f"[{i + 1}]" for i, _ in enumerate(hits)) + "." if hits else " No matching policy excerpt was found."
    return f"Subject: Request Regarding {category.replace('_', ' ').title()}\n\nDear {greeting},\n\n{english} I kindly request your assistance. {citation}\n\nSincerely,\n{name}"