"""Policy retrieval with a dependency-free lexical fallback."""

import re
from pathlib import Path


class PolicyRetriever:
    def __init__(self, policy_dir="data/policies"):
        self.policy_dir = Path(policy_dir)
        self.chunks = self._load_chunks()

    def _load_chunks(self):
        chunks = []
        for path in sorted(self.policy_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
            sections = re.split(r"\n(?=## )", text)
            for section in sections:
                lines = section.strip().splitlines()
                if not lines:
                    continue
                heading = lines[0].lstrip("# ").strip()
                body = " ".join(line.strip() for line in lines[1:] if line.strip())
                if body:
                    chunks.append({"doc": path.name, "section": heading, "title": title, "text": f"{title} - {heading}: {body}"})
        return chunks

    def search(self, query, k=3, min_score=0.1):
        query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        ranked = []
        for chunk in self.chunks:
            terms = set(re.findall(r"[a-z0-9]+", chunk["text"].casefold()))
            score = len(query_terms & terms) / max(1, len(query_terms))
            if score >= min_score:
                ranked.append({**chunk, "score": round(score, 4)})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:k]