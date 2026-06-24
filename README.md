# LangChain Search Chatbot

An AI chatbot built with Streamlit, LangChain, and Groq. It can search the web,
Wikipedia, and arXiv, and answer questions about your own PDFs using hybrid
retrieval (semantic + BM25).

**Live demo:** https://langchain-agent-chatbot-md7lccmr9isaedlcv5jqvp.streamlit.app/

## Features

- **Search** — query Wikipedia, arXiv, and the web (DuckDuckGo) through a
  tool-using agent.
- **PDF Q&A** — upload one or more PDFs and ask questions about them.
- **Hybrid RAG** — dense semantic search (Chroma + embeddings) fused with BM25
  lexical search for accurate, source-cited answers.
- **Fast inference** — powered by Groq LLMs.

## Limitations

- **No conversation memory.** Each question is answered on its own — the bot
  does not remember previous turns, so follow-ups like "what about its
  population?" won't resolve against earlier messages. The chat history is shown
  in the UI but is not sent back to the model.

## Hybrid RAG (BM25 + semantic)

PDF question answering uses **hybrid retrieval** — it combines two complementary
search methods instead of relying on either one alone:

- **Semantic (dense)** — documents are embedded with a HuggingFace
  sentence-transformer and stored in Chroma. Retrieval uses MMR so the returned
  chunks are both relevant and diverse. Good at matching *meaning* even when the
  wording differs.
- **BM25 (sparse/lexical)** — a classic keyword-ranking algorithm
  (`rank-bm25`). Good at exact matches: names, acronyms, codes, and rare terms
  that embeddings often miss.

The two result sets are fused with LangChain's `EnsembleRetriever` (weighted
rank fusion), giving more robust retrieval than either signal alone. Answers are
returned with inline source citations (file name and page). The whole pipeline
lives in [`rag.py`](rag.py).

## Tech stack

| Component   | Used for             |
| ----------- | -------------------- |
| Streamlit   | UI                   |
| LangChain   | Orchestration        |
| Groq        | LLM inference        |
| HuggingFace | Embeddings           |
| Chroma      | Vector store (dense) |
| rank-bm25   | Lexical retrieval    |

## Project structure

```
main.py            Streamlit app (UI + chat + agent)
rag.py             Hybrid RAG module (semantic + BM25 retrieval)
requirements.txt   Dependencies
.env               API keys (gitignored; copy from .env.example)
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your API key. Copy `.env.example` to `.env` and fill it in:

   ```
   GROQ_API_KEY=your_groq_key_here
   ```

   Get a free key at https://console.groq.com/keys. `HF_TOKEN` is optional and
   only needed for gated embedding models.

3. Run the app:

   ```bash
   streamlit run main.py
   ```

## Deployment (Streamlit Cloud)

`.env` is gitignored, so it is **not** deployed. Instead, add your key under the
app's **Settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

The app reads it automatically and never prompts users for a key.
