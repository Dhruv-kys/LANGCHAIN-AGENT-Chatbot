import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env that sits next to this file, regardless of the working
# directory the app was launched from.
load_dotenv(Path(__file__).with_name(".env"))

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# Hybrid RAG pipeline (semantic + BM25) lives in its own module.
import rag

# Tool-using agent. Prefer the supported `langchain.agents.create_agent`
# (LangChain 1.x); fall back to the older langgraph prebuilt if needed.
try:
    from langchain.agents import create_agent
    AGENT_AVAILABLE = True
except ImportError:
    try:
        from langgraph.prebuilt import create_react_agent as create_agent
        AGENT_AVAILABLE = True
    except ImportError:
        AGENT_AVAILABLE = False

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None  # Holds a rag.HybridRAG bundle once PDFs are indexed

# Get HF_TOKEN from secrets or env
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
except:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing embeddings: {e}")
    embeddings = None

# Tool setup
# Build tools individually so one missing dependency (e.g. ddgs for
# DuckDuckGo) doesn't disable every other tool.
search_tools = []
_tool_errors = []


def _add_tool(label, factory):
    try:
        search_tools.append(factory())
    except Exception as e:  # noqa: BLE001 - surface, but keep other tools alive
        _tool_errors.append(f"{label}: {e}")


_add_tool("arXiv", lambda: ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)))
_add_tool("Wikipedia", lambda: WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)))
_add_tool("Web Search", lambda: DuckDuckGoSearchResults(name="Search"))

if _tool_errors:
    st.warning("Some tools are unavailable:\n\n" + "\n".join(f"- {e}" for e in _tool_errors))

# Streamlit UI Configuration
st.set_page_config(
    page_title="AI BOT",
    page_icon="🤖",
)

# Header
st.title("AI + RAG Chatbot")
st.caption("📔BM25 + Semantic retrieval")

# Resolve the Groq API key automatically from Streamlit secrets or the
# environment (.env). The app never prompts the user for it.
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None
api_key = api_key or os.getenv("GROQ_API_KEY")

# Fixed model — gpt-oss is the most reliable for web/Wikipedia/arXiv search.
model_name = "openai/gpt-oss-120b"

if not api_key:
    st.error("No GROQ_API_KEY found. Add it to .env or Streamlit secrets.")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hi! I can search the web, Wikipedia, and arXiv papers. Upload PDFs to ask questions about your documents, or just ask me anything.",
    }]

# Document upload
st.subheader("Documents")
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF documents to enable Q&A functionality"
)

# Process uploaded files into a hybrid (semantic + BM25) retriever
if uploaded_files and embeddings:
    if st.session_state.rag is None or st.button("Re-process"):
        with st.spinner("Indexing documents (semantic + BM25)…"):
            try:
                st.session_state.rag = rag.build_from_uploads(uploaded_files, embeddings)
                st.success(
                    f"Indexed {len(uploaded_files)} file(s) • "
                    f"{st.session_state.rag.num_chunks} chunks • hybrid retrieval ready"
                )
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Error indexing documents: {e}")
    elif st.session_state.rag is not None:
        st.info(f"{len(uploaded_files)} file(s) ready for hybrid Q&A")
elif uploaded_files and not embeddings:
    st.error("Embeddings not initialized")

# Chat
st.subheader("Chat")

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    if not api_key:
        st.warning("No API key configured. Set GROQ_API_KEY in your .env file or Streamlit secrets.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process response
        with st.chat_message("assistant"):
            try:
                # Check if PDF (hybrid RAG) Q&A is available
                if st.session_state.rag is not None:
                    with st.spinner("Searching documents (hybrid retrieval)…"):
                        try:
                            llm_temp = ChatGroq(groq_api_key=api_key, model_name=model_name)

                            response, source_docs = rag.answer_with_rag(
                                llm_temp, st.session_state.rag, prompt
                            )

                            # Surface which documents grounded the answer
                            sources = rag.unique_sources(source_docs)
                            if sources:
                                response += "\n\n---\n**Sources:** " + " · ".join(sources)

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            response = f"I encountered an error while processing your question: {str(e)}"
                else:
                    # Use search agent
                    with st.spinner("Thinking..."):
                        try:
                            llm = ChatGroq(
                                groq_api_key=api_key,
                                model_name=model_name,
                                streaming=False  # Changed for agent compatibility
                            )
                            
                            response = None
                            if AGENT_AVAILABLE and search_tools:
                                try:
                                    # Tool-using agent (web/Wikipedia/arXiv search)
                                    agent = create_agent(llm, search_tools)
                                    result = agent.invoke({
                                        "messages": [HumanMessage(content=prompt)]
                                    })
                                    # Extract the final AI message
                                    if isinstance(result, dict) and result.get("messages"):
                                        last_message = result["messages"][-1]
                                        response = getattr(last_message, "content", None)
                                        if response is None and isinstance(last_message, dict):
                                            response = last_message.get("content")
                                        if response is None:
                                            response = str(last_message)
                                    else:
                                        response = str(result)
                                except Exception as agent_err:
                                    # Groq + llama tool-calling can intermittently emit
                                    # malformed function calls. Degrade gracefully to a
                                    # direct answer instead of surfacing a raw error.
                                    st.info("Live search hit a snag; answering from the model's own knowledge.")
                                    response = None

                            if not response:
                                if not AGENT_AVAILABLE:
                                    st.info("Search agent unavailable; answering from the model's own knowledge.")
                                elif not search_tools:
                                    st.info("No search tools are available; answering from the model's own knowledge.")
                                response = llm.invoke([HumanMessage(content=prompt)]).content

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            response = f"I encountered an error: {str(e)}. Please try again."
                
                # Display and store response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
if st.button("Clear chat"):
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hi! I can search the web, Wikipedia, and arXiv papers. Upload PDFs to ask questions about your documents, or just ask me anything.",
    }]
    st.rerun()
st.caption("Powered by LangChain & Groq")
