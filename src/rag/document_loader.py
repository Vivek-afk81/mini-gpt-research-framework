
"""
LexiFlow RAG — Document loader and chunker.

Loads campus policy Markdown files and splits them into
overlapping chunks for embedding and retrieval.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A single chunk of text with source metadata."""
    text: str
    source: str
    title: str
    chunk_index: int


def load_markdown(filepath):
    """Load a Markdown file and return its content."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks by character count.

    Prefers paragraph, sentence, and newline boundaries.
    Guarantees forward progress to avoid getting stuck.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    separators = ["\n\n", ".\n", ". ", "\n"]
    min_boundary = chunk_size // 2

    while start < text_length:
        # Maximum end position for this chunk
        end = min(start + chunk_size, text_length)

        # Prefer a natural boundary, but not too early in the chunk
        if end < text_length:
            for sep in separators:
                last_break = text.rfind(sep, start, end)

                if last_break >= start + min_boundary:
                    end = last_break + len(sep)
                    break

        # Ensure the chunk always advances
        if end <= start:
            end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop after processing the final chunk
        if end >= text_length:
            break

        # Move forward while retaining the requested overlap
        next_start = end - overlap

        # Guarantee forward progress
        start = max(start + 1, next_start)

    return chunks


def extract_title(text):
    """Extract the first Markdown heading as the document title."""
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line[2:].strip()

    return "Untitled"


def load_knowledge_base(
    knowledge_dir="data/knowledge",
    chunk_size=500,
    overlap=100,
):
    """
    Load all Markdown files from the knowledge directory,
    chunk them, and return a list of DocumentChunk objects.
    """
    knowledge_path = Path(knowledge_dir)
    all_chunks = []

    if not knowledge_path.exists():
        print(
            f"Warning: Knowledge directory "
            f"'{knowledge_dir}' not found."
        )
        return all_chunks

    if not knowledge_path.is_dir():
        print(
            f"Warning: '{knowledge_dir}' is not a directory."
        )
        return all_chunks

    # Find Markdown files once
    files = sorted(knowledge_path.glob("*.md"))

    if not files:
        print(
            f"Warning: No Markdown files found in "
            f"'{knowledge_dir}'."
        )
        return all_chunks

    print(f"Found {len(files)} Markdown documents.")

    for doc_number, filepath in enumerate(files, start=1):
        print(
            f"Processing document {doc_number}/{len(files)}: "
            f"{filepath.name}"
        )

        try:
            text = load_markdown(filepath)

            if not text.strip():
                print(f"Skipping empty document: {filepath.name}")
                continue

            title = extract_title(text)

            chunks = chunk_text(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    DocumentChunk(
                        text=chunk,
                        source=filepath.name,
                        title=title,
                        chunk_index=i,
                    )
                )

            print(f"  Created {len(chunks)} chunks.")

        except (OSError, UnicodeError) as exc:
            print(
                f"Warning: Could not read {filepath.name}: {exc}"
            )

    print(
        f"\nLoaded {len(all_chunks)} chunks "
        f"from {len(files)} Markdown documents."
    )

    return all_chunks
