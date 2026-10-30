"""Small Groq client with disk caching and retry handling."""

import hashlib
import json
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keep local mode usable before optional dependencies are installed.
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def is_configured():
    """Return whether Groq can be attempted in this environment."""
    return bool(os.getenv("GROQ_API_KEY"))


def chat(messages, model=None, json_mode=False, temperature=0.0, max_tokens=400, retries=4):
    """Call Groq when configured; return None when no API key is available."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    cache_path = Path(os.getenv("LEXIFLOW_CACHE", ".cache/groq.jsonl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(
        [messages, model, json_mode, temperature, max_tokens], sort_keys=True
    ).encode()).hexdigest()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("key") == key:
                return item.get("value")

    client = Groq(api_key=api_key)
    kwargs = {
        "model": model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(min(retries, int(os.getenv("GROQ_MAX_RETRIES", "3")))):
        try:
            value = client.chat.completions.create(**kwargs).choices[0].message.content
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            return value
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(min(2 ** attempt, 8))
    return None