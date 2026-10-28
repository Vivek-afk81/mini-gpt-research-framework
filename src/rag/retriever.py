"""
LexiFlow RAG — Embedding and FAISS retrieval.

Embeds document chunks using sentence-transformers and provides
semantic search via a FAISS index.
"""

import os
import pickle
from pathlib import Path

import numpy as np

from src.rag.document_loader import DocumentChunk, load_knowledge_base


class Retriever:
    """
    Embeds document chunks and retrieves the most relevant ones
    for a given query using FAISS similarity search.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", index_dir="data/index"):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.chunks = []
        self.index = None
        self._embed_model = None

    @property
    def embed_model(self):
        """Lazy-load the sentence transformer model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    def build_index(self, knowledge_dir="data/knowledge", chunk_size=500, overlap=100):
        """
        Load documents, chunk them, embed, and build a FAISS index.
        Saves the index and chunks to disk for reuse.
        """
        import faiss

        self.chunks = load_knowledge_base(knowledge_dir, chunk_size, overlap)

        if not self.chunks:
            print("No chunks found. Cannot build index.")
            return

        # Embed all chunks
        texts = [c.text for c in self.chunks]
        print(f"Embedding {len(texts)} chunks ...")
        embeddings = self.embed_model.encode(texts, show_progress_bar=True,
                                             convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save to disk
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index built: {self.index.ntotal} vectors, dim={dim}")
        print(f"Saved to {self.index_dir}")

    def load_index(self):
        """Load a pre-built FAISS index and chunks from disk."""
        import faiss

        index_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            print("Index not found. Building from scratch ...")
            self.build_index()
            return

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k most relevant document chunks for a query.

        Returns a list of (chunk, score) tuples.
        """
        import faiss

        if self.index is None:
            self.load_index()

        # Embed query
        query_vec = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Search
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def format_context(self, results):
        """
        Format retrieved chunks into a context string for the LLM prompt,
        including source citations.
        """
        if not results:
            return "No relevant campus documents found."

        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {chunk.title} ({chunk.source})]\n{chunk.text}"
            )

        return "\n\n".join(context_parts)
