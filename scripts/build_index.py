"""
Build the FAISS index from campus knowledge documents.

Usage:
    python scripts/build_index.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.retriever import Retriever


def main():
    print("Building FAISS index from campus documents ...\n")

    retriever = Retriever()
    retriever.build_index(
        knowledge_dir="data/knowledge",
        chunk_size=500,
        overlap=100,
    )

    # Quick test
    print("\n--- Test Retrieval ---")
    test_queries = [
        "What is the minimum attendance requirement?",
        "Can I submit assignment late?",
        "How to apply for leave?",
        "When is the mid semester exam?",
        "Is ragging allowed in hostel?",
    ]

    for query in test_queries:
        results = retriever.retrieve(query, top_k=2)
        print(f"\nQ: {query}")
        for chunk, score in results:
            print(f"  [{score:.3f}] {chunk.title} — {chunk.text[:80]}...")


if __name__ == "__main__":
    main()
