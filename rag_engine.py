"""
SilverVault AI - RAG Engine
Document Ingestion (PyMuPDF), Recursive Chunking with Page Tracking,
Hybrid Semantic Retrieval, and Context-Bound LLM Generation with Qwen 0.5B.
"""

import os
import re
import json
from typing import List, Dict, Tuple, Any, Optional
import pymupdf  # Modern PyMuPDF import
from database import save_chunks_with_pages, get_chunks_for_docs
from nlp_ml_engine import HybridVectorRetriever, tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "models", MODEL_FILENAME),
    os.path.join(BASE_DIR, "..", "silvervault_v", "silvervault_ai_learning_analysis", "data", "models", MODEL_FILENAME),
    os.path.join(BASE_DIR, "..", "silvervault_final", "silvervault_ai_complete_project", "data", "models", MODEL_FILENAME),
    os.path.join(os.path.dirname(BASE_DIR), "models", MODEL_FILENAME),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.isfile(p)), MODEL_CANDIDATES[0])

# --- PDF Extraction & Chunking ---

def extract_pdf_pages_data(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text per page along with page numbers from PDF bytes.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        if text.strip():
            pages_data.append({
                "page_number": page_idx + 1,
                "text": text
            })
    doc.close()
    return pages_data

def chunk_pages_data(pages_data: List[Dict[str, Any]], max_chars: int = 1000, overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Split extracted pages into clean text chunks with preserved page boundaries and overlap.
    """
    chunks = []
    chunk_idx = 0

    for page_info in pages_data:
        raw_text = page_info["text"]
        p_num = page_info["page_number"]
        # Normalize whitespace and page artifacts
        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not cleaned:
            continue

        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + max_chars)
            if end < len(cleaned):
                # Try breaking at paragraph, sentence, or newline boundary
                boundary = max(
                    cleaned.rfind("\n\n", start, end),
                    cleaned.rfind(". ", start, end),
                    cleaned.rfind("\n", start, end),
                    cleaned.rfind(" ", start, end)
                )
                if boundary > start + max_chars // 2:
                    end = boundary + (1 if cleaned[boundary] != "\n" else 0)

            piece = cleaned[start:end].strip()
            if len(piece) >= 35:
                approx_tokens = len(piece.split())
                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": p_num,
                    "content": piece,
                    "token_count": approx_tokens
                })
                chunk_idx += 1

            if end >= len(cleaned):
                break
            start = max(start + 1, end - overlap)

    return chunks

def index_document_chunks(document_id: int, chunks: List[Dict[str, Any]]):
    """Save processed chunks into the SQLite database."""
    save_chunks_with_pages(document_id, chunks)

# --- Hybrid Retrieval ---

def retrieve_context(
    document_ids: List[int],
    user_id: int,
    query: str,
    max_chunks: int = 6
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve semantically relevant chunks across the student's selected subject documents.
    Returns:
        context_string: Grounded text blocks for the LLM
        sources: List of metadata dicts with title, page, chunk_index, score
    """
    if not document_ids or not query.strip():
        return "", []

    raw_chunks = get_chunks_for_docs(document_ids, user_id)
    if not raw_chunks:
        return "", []

    retriever = HybridVectorRetriever(raw_chunks)
    ranked = retriever.score_query(query)
    chosen = ranked[:max_chunks]

    if not chosen:
        return "", []

    context_parts = []
    sources = []
    seen_sources = set()

    for score, ch in chosen:
        title = ch.get("title", "Book")
        page = ch.get("page_number", 1)
        c_idx = ch.get("chunk_index", 0) + 1
        source_label = f"{title} (p. {page}, chunk {c_idx})"

        context_parts.append(f"[Source: {title} | Page {page} | Chunk {c_idx}]\n{ch['content']}")
        if source_label not in seen_sources:
            sources.append({
                "title": title,
                "page": page,
                "chunk_index": c_idx,
                "score": score,
                "content_preview": ch["content"][:180] + "..."
            })
            seen_sources.add(source_label)

    return "\n\n---\n\n".join(context_parts), sources

# --- Local LLM Inference (Qwen 0.5B via llama-cpp-python) ---

def load_llama_model(model_path: str = MODEL_PATH):
    """
    Load and cache local Qwen 0.5B model instance.
    """
    if not os.path.isfile(model_path):
        # Check fallback candidates
        candidate = next((p for p in MODEL_CANDIDATES if os.path.isfile(p)), None)
        if candidate:
            model_path = candidate
        else:
            raise FileNotFoundError(f"Local Qwen GGUF model not found at {model_path}. Please check MODEL_PATH.")

    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError("llama-cpp-python is not installed in the active environment.") from e

    if not hasattr(load_llama_model, "_cached") or load_llama_model._cached[0] != model_path:
        threads = max(2, (os.cpu_count() or 4) - 1)
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=threads,
            verbose=False
        )
        load_llama_model._cached = (model_path, llm)
    return load_llama_model._cached[1]

def run_llm_completion(prompt: str, system_prompt: str = "", max_tokens: int = 350, temperature: float = 0.15) -> str:
    """Execute chat completion with local Qwen 0.5B."""
    llm = load_llama_model()
    sys_content = system_prompt or (
        "You are SilverVault AI, a precise academic assistant. "
        "Strictly answer using ONLY the supplied source excerpts. "
        "Never use outside knowledge. If the excerpts do not contain the answer, "
        "explicitly state that the information was not found in the selected book."
    )
    res = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.85,
        max_tokens=max_tokens
    )
    return res["choices"][0]["message"]["content"].strip()

# --- Subject-Grounded Question Answering ---

def generate_grounded_answer(question: str, context: str, sources: List[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """
    Generate context-bound academic answer. Returns (answer_text, is_grounded).
    """
    if not context.strip():
        return (
            "⚠️ **Information Not Found in Selected Book**\n\n"
            "This concept could not be retrieved from the uploaded materials for this subject. "
            "SilverVault AI is strictly context-bound to prevent hallucinations and ensure study integrity.",
            False
        )

    prompt = f"""Answer the student's question using ONLY the source excerpts below.
If the excerpts do not contain sufficient facts to answer, respond exactly:
'This information was not found in the selected subject book.'

Do not invent facts, citations, or extrapolate beyond the text. Keep the explanation structured and academic.

SOURCE EXCERPTS:
{context}

STUDENT QUESTION:
{question}

Answer grounded strictly in the source excerpts:"""

    try:
        answer = run_llm_completion(prompt, max_tokens=400, temperature=0.1)
    except Exception as e:
        return f"Error executing local model: {e}", False

    low = answer.lower()
    if not answer or any(x in low for x in ["not found in the selected", "not provided in the context", "context does not contain", "no mention of"]):
        return (
            "⚠️ **Information Not Found in Selected Book**\n\n"
            "The selected subject textbook does not contain sufficient reference material to answer this question. "
            "Please consult other chapters or upload the complete syllabus notes.",
            False
        )

    return answer, True
