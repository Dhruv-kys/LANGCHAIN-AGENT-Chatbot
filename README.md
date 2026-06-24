🤖 LangChain Search Chatbot
    An AI-powered chatbot built with Streamlit, LangChain, and Groq.
    It supports web, Wikipedia, and arXiv search, along with PDF question answering using embeddings and Chroma.

👾 Working Cloud Link : 


✨ Features
    🔍 Search Integration: Query Wikipedia, arXiv, or DuckDuckGo.
    📄 PDF Q&A: Upload multiple PDFs and ask questions directly.
    🔀 Hybrid RAG: Semantic (Chroma + embeddings) fused with BM25 lexical search for accurate, source-cited answers.
    ⚡ Groq LLMs: Use high-speed models
    🎨 Streamlit UI with chat interface and sidebar controls.
    ❤️ Built for research, learning, and document exploration.

🛠️ Tech Stack
    Streamlit – UI Framework
    LangChain – Orchestration
    Groq – Fast LLM Inference
    HuggingFace – Embeddings
    Chroma – Vector DB

📦 LangChain Search Chatbot
 ┣ 📄 main.py              # Main Streamlit app
 ┣ 📄 rag.py               # Hybrid RAG module (semantic + BM25 retrieval)
 ┣ 📄 requirements.txt     # Dependencies
 ┣ 📄 .env                 # Environment variables (ignored in git)
 ┗ 📂 (uploaded PDFs handled at runtime)

