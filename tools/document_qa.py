"""
Document Q&A tool: retrieves relevant document chunks for a query.

Pipeline:
  1. Ingest: PDFs in data/pdfs/ -> text -> chunks -> embeddings -> ChromaDB
     - Incremental: already-ingested files (by hash) are skipped
  2. Query: embed query -> vector search in ChromaDB -> return top-k chunks

Chunking strategy:
  - Fixed size (500 chars) with 50-char overlap.
  - Simple, deterministic, cheap to compute.
  - Tradeoff vs semantic chunking: we lose some boundary awareness but
    gain speed and reproducibility. For a 48-hour build, this is the
    right pragmatic choice.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - 22MB, runs locally (no API cost)
  - Good enough for similarity search at this scale
  - Tradeoff: weaker than OpenAI text-embedding-3-small, but free + local
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from agent.config import CHROMA_DIR, DATA_DIR
from tools.base import Tool

logger = logging.getLogger(__name__)


# ----- Chunking -----

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size character chunks with overlap. Respects paragraph breaks when possible."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Try to break at a paragraph or sentence boundary near the end
        window = text[start:end]
        # Prefer paragraph break, then sentence, then word
        for sep in ["\n\n", ". ", " "]:
            idx = window.rfind(sep)
            if idx > chunk_size // 2:  # only use if reasonably far in
                end = start + idx + len(sep)
                break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ----- PDF reading -----

def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            logger.warning(f"Failed to extract page {i} of {path.name}: {e}")
    return "\n\n".join(pages)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ----- Index singleton -----

class _DocumentIndex:
    """
    Wraps ChromaDB + embedding model. Singleton so we don't reload
    the 22MB model on every tool invocation.
    """
    _instance: "_DocumentIndex | None" = None

    def __init__(self) -> None:
        logger.info("Initializing document index (one-time)...")
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        # Lazy-load model; this is the slow bit
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model all-MiniLM-L6-v2...")
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._model

    @classmethod
    def get(cls) -> "_DocumentIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ingest_directory(self, pdf_dir: Path) -> dict[str, Any]:
        """Ingest all PDFs in pdf_dir, skipping already-ingested files."""
        if not pdf_dir.exists():
            return {"ingested": 0, "skipped": 0, "errors": [], "note": f"{pdf_dir} does not exist"}

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        ingested, skipped, errors = 0, 0, []

        # Get existing source hashes from collection
        existing = set()
        try:
            res = self._collection.get(include=["metadatas"])
            for md in res.get("metadatas", []) or []:
                if md and "source_hash" in md:
                    existing.add(md["source_hash"])
        except Exception as e:
            logger.warning(f"Could not read existing index: {e}")

        for pdf in pdfs:
            try:
                fhash = _file_hash(pdf)
                if fhash in existing:
                    skipped += 1
                    logger.info(f"Skipping (already indexed): {pdf.name}")
                    continue
                text = _extract_pdf_text(pdf)
                if not text.strip():
                    errors.append(f"{pdf.name}: empty text after extraction")
                    continue
                chunks = _chunk_text(text)
                if not chunks:
                    errors.append(f"{pdf.name}: no chunks produced")
                    continue

                embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()
                ids = [f"{fhash}-{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": pdf.name, "source_hash": fhash, "chunk_index": i}
                    for i in range(len(chunks))
                ]
                self._collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                ingested += 1
                logger.info(f"Ingested {pdf.name}: {len(chunks)} chunks")
            except Exception as e:
                errors.append(f"{pdf.name}: {type(e).__name__}: {e}")
                logger.error(f"Failed to ingest {pdf.name}: {e}")

        return {
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "total_chunks_in_index": self._collection.count(),
        }

    def query(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        q_emb = self.model.encode([question], show_progress_bar=False).tolist()
        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, md, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": md.get("source", "unknown") if md else "unknown",
                "chunk_index": md.get("chunk_index", -1) if md else -1,
                "similarity": round(1 - dist, 4),  # cosine distance -> similarity
            })
        return hits


# ----- Tool -----

class DocumentQATool(Tool):
    name = "document_qa"
    description = (
        "Searches ingested PDF documents for content relevant to a query "
        "and returns the top matching passages. Use this when the user's "
        "question is about content in uploaded or indexed documents, or "
        "asks 'what does the document/paper say about X'. "
        "\n\n"
        "The tool returns text passages with source filename and "
        "similarity scores. You should synthesise an answer from these "
        "passages — do not just paste them back. If no passages are "
        "relevant (low similarity), say so."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question or topic to search for in the documents.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to retrieve. Default 4, max 8.",
                "default": 4,
                "minimum": 1,
                "maximum": 8,
            },
        },
        "required": ["question"],
    }

    async def _run(self, question: str, top_k: int = 4) -> str:
        top_k = max(1, min(int(top_k), 8))
        index = _DocumentIndex.get()
        hits = index.query(question, top_k=top_k)
        if not hits:
            return (
                "No indexed documents matched this query. "
                "Either no documents are ingested yet, or none are relevant."
            )
        lines = [f"Top {len(hits)} passages for: {question!r}", ""]
        for i, h in enumerate(hits, start=1):
            lines.append(f"[{i}] source={h['source']} chunk={h['chunk_index']} similarity={h['similarity']}")
            passage = h["text"]
            if len(passage) > 600:
                passage = passage[:600] + "..."
            lines.append(passage)
            lines.append("")
        return "\n".join(lines).strip()


# ----- Convenience: run ingestion from CLI -----

def ingest_pdfs() -> dict[str, Any]:
    """Called from a one-off script to populate the index."""
    index = _DocumentIndex.get()
    return index.ingest_directory(DATA_DIR / "pdfs")


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(ingest_pdfs(), indent=2))