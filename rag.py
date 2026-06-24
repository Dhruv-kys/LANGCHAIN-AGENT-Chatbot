"""Hybrid Retrieval-Augmented Generation (RAG) pipeline.

Combines two complementary retrievers over the uploaded documents:

* **Dense / semantic** – Chroma vector store over sentence-transformer
  embeddings, using MMR so the returned chunks are relevant *and* diverse.
* **Sparse / lexical** – BM25 (``rank_bm25``) which excels at exact keyword,
  code, name and acronym matches that embeddings often miss.

Both are fused with an :class:`EnsembleRetriever` (Reciprocal Rank Fusion),
giving "hybrid" retrieval that is more robust than either signal alone.

The module is deliberately framework-agnostic (no Streamlit imports) so it can
be reused from any app, notebook or test.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# EnsembleRetriever moved to ``langchain_classic`` in LangChain 1.x but still
# lives under ``langchain.retrievers`` on 0.x. Import defensively so the module
# works on both.
try:  # LangChain 0.x
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # LangChain 1.x
    from langchain_classic.retrievers import EnsembleRetriever


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_K = 5
# Weight given to (sparse_bm25, dense_semantic) when fusing results.
DEFAULT_WEIGHTS: Tuple[float, float] = (0.4, 0.6)

RAG_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using "
    "ONLY the context retrieved from their uploaded documents. "
    "Cite the supporting source(s) inline like [source: filename p.N]. "
    "If the answer is not contained in the context, say so plainly and then "
    "give your best general-knowledge answer, clearly marked as such. "
    "Be concise and do not invent citations."
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


@dataclass
class HybridRAG:
    """Bundles the fused retriever with its backing vector store.

    Attributes
    ----------
    retriever:
        The :class:`EnsembleRetriever` to call ``.invoke(query)`` on.
    vectorstore:
        Underlying Chroma store (kept so the caller can add docs / inspect it).
    num_chunks:
        Number of chunks indexed – handy for UI feedback.
    """

    retriever: EnsembleRetriever
    vectorstore: Chroma
    num_chunks: int


# --------------------------------------------------------------------------- #
# Document loading & splitting
# --------------------------------------------------------------------------- #
def load_pdfs(uploaded_files: Iterable) -> List[Document]:
    """Load Streamlit ``UploadedFile`` objects (or any obj with ``.name`` and
    ``.getvalue()``) into LangChain ``Document``s.

    The original filename is stamped into ``metadata['source']`` so citations
    reference the real file rather than a temp path.
    """
    documents: List[Document] = []
    for uploaded in uploaded_files:
        name = getattr(uploaded, "name", "document.pdf")
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            docs = PyPDFLoader(tmp_path).load()
            for doc in docs:
                doc.metadata["source"] = name
            documents.extend(docs)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Chunk documents for indexing, preserving a ``start_index`` for traceability."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------------------------------------------------------------------------- #
# Hybrid retriever construction
# --------------------------------------------------------------------------- #
def build_hybrid_retriever(
    splits: Sequence[Document],
    embeddings,
    k: int = DEFAULT_K,
    weights: Tuple[float, float] = DEFAULT_WEIGHTS,
    use_mmr: bool = True,
) -> HybridRAG:
    """Build a fused BM25 + semantic retriever over ``splits``.

    Parameters
    ----------
    splits:
        Pre-chunked documents (see :func:`split_documents`).
    embeddings:
        Any LangChain embeddings object (e.g. ``HuggingFaceEmbeddings``).
    k:
        Final number of chunks each sub-retriever contributes.
    weights:
        ``(bm25_weight, dense_weight)`` for rank fusion.
    use_mmr:
        Use Maximal Marginal Relevance for the dense retriever to reduce
        near-duplicate chunks.
    """
    if not splits:
        raise ValueError("Cannot build a retriever from an empty document set.")

    # Dense / semantic retriever ------------------------------------------------
    vectorstore = Chroma.from_documents(documents=list(splits), embedding=embeddings)
    if use_mmr:
        dense = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(k * 4, 20), "lambda_mult": 0.5},
        )
    else:
        dense = vectorstore.as_retriever(search_kwargs={"k": k})

    # Sparse / lexical retriever ------------------------------------------------
    bm25 = BM25Retriever.from_documents(list(splits))
    bm25.k = k

    # Fuse the two signals ------------------------------------------------------
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=list(weights),
    )

    return HybridRAG(retriever=ensemble, vectorstore=vectorstore, num_chunks=len(splits))


def build_from_uploads(uploaded_files: Iterable, embeddings, **kwargs) -> HybridRAG:
    """Convenience: PDFs -> chunks -> hybrid retriever in one call."""
    documents = load_pdfs(uploaded_files)
    if not documents:
        raise ValueError("No readable content found in the uploaded PDF(s).")
    splits = split_documents(documents)
    return build_hybrid_retriever(splits, embeddings, **kwargs)


# --------------------------------------------------------------------------- #
# Context formatting & answering
# --------------------------------------------------------------------------- #
def _source_label(doc: Document) -> str:
    src = os.path.basename(str(doc.metadata.get("source", "document")))
    page = doc.metadata.get("page")
    return f"{src} p.{page + 1}" if isinstance(page, int) else src


def format_context(docs: Sequence[Document]) -> str:
    """Render retrieved chunks into a labelled, citation-friendly context block."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] (source: {_source_label(doc)})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def answer_with_rag(llm, hybrid: HybridRAG, question: str) -> Tuple[str, List[Document]]:
    """Retrieve hybrid context and produce a grounded answer.

    Returns ``(answer_text, retrieved_docs)`` so the UI can show sources.
    """
    docs: List[Document] = hybrid.retriever.invoke(question)
    context = format_context(docs) if docs else "No relevant context found."
    messages = _RAG_PROMPT.format_messages(context=context, question=question)
    answer = llm.invoke(messages).content
    return answer, docs


def unique_sources(docs: Sequence[Document]) -> List[str]:
    """De-duplicated, ordered list of human-readable source labels."""
    seen, out = set(), []
    for doc in docs:
        label = _source_label(doc)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out
