import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback: use direct tool calling
    from langchain_core.tools import tool

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

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
try:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    search_tool = DuckDuckGoSearchResults(name="Search")
    search_tools = [arxiv_tool, wiki_tool, search_tool]
except Exception as e:
    st.error(f"Error setting up tools: {e}")
    search_tools = []

# Streamlit UI Configuration
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Eye-catching Dark Mode CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Animated gradient background */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 50%, #0f1419 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header with gradient text */
        .main-header {
            padding: 2rem 0;
            margin-bottom: 3rem;
            text-align: left;
            position: relative;
        }
        
        .main-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, #ec4899, transparent);
            background-size: 200% 100%;
            animation: shimmer 3s linear infinite;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        .main-header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            padding: 0;
            letter-spacing: -0.02em;
            animation: fadeInUp 0.8s ease;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main-header p {
            color: #94a3b8;
            font-size: 1.1rem;
            margin: 0.8rem 0 0 0;
            font-weight: 300;
            animation: fadeInUp 1s ease 0.2s both;
        }
        
        /* Sidebar with subtle gradient */
        .css-1d391kg {
            background: linear-gradient(180deg, #1a1d29 0%, #0f1419 100%);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1d29 0%, #0f1419 100%);
            border-right: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        /* Text colors */
        h1, h2, h3, h4, h5, h6 {
            color: #fafafa !important;
        }
        
        p, label, div {
            color: #d1d5db !important;
        }
        
        /* Enhanced input fields with glow */
        .stTextInput > div > div > input {
            background-color: rgba(31, 41, 55, 0.6);
            color: #fafafa;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), 0 0 20px rgba(99, 102, 241, 0.3);
            outline: none;
        }
        
        .stSelectbox > div > div > select {
            background-color: rgba(31, 41, 55, 0.6);
            color: #fafafa;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div > select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        /* Glowing buttons */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        /* File uploader with gradient border */
        .uploadedFile {
            background: rgba(31, 41, 55, 0.4);
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.1);
        }
        
        /* Chat messages with subtle glow */
        .stChatMessage {
            background-color: transparent;
            padding: 1rem;
            border-radius: 16px;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .stChatMessage:hover {
            background: rgba(99, 102, 241, 0.05);
        }
        
        /* Enhanced status boxes */
        .stSuccess {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            border-left: 4px solid #10b981;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
            border-left: 4px solid #ef4444;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
        }
        
        /* Dividers with gradient */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
            margin: 2rem 0;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #d1d5db;
        }
        
        /* Enhanced chat input */
        .stChatInput > div > div > input {
            background: rgba(31, 41, 55, 0.6);
            color: #fafafa;
            border: 2px solid rgba(99, 102, 241, 0.3);
            border-radius: 24px;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stChatInput > div > div > input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1), 0 0 30px rgba(99, 102, 241, 0.4);
            outline: none;
        }
        
        /* Scrollbar with gradient */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(26, 29, 41, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #6366f1, #8b5cf6);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #8b5cf6, #ec4899);
        }
        
        /* Caption styling */
        .stCaption {
            color: #94a3b8;
            font-size: 0.85rem;
        }
        
        /* Spinner enhancement */
        .stSpinner > div {
            border-color: #6366f1 transparent transparent transparent;
        }
        
        /* Floating animation for elements */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
    </style>
""", unsafe_allow_html=True)

# Eye-catching Header
st.markdown("""
    <div class="main-header">
        <h1>✨ AI Chatbot</h1>
        <p>Search the web • Research papers • Document Q&A</p>
    </div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Key input with secrets support
    try:
        api_key_secret = st.secrets.get("GROQ_API_KEY")
    except:
        api_key_secret = None
    
    if api_key_secret:
        api_key = api_key_secret
        st.success("✅ API Key loaded")
    else:
        api_key = st.text_input(
            "🔑 API Key",
            type="password",
            placeholder="Enter your API key"
        )
    
    model_name = st.selectbox(
        "🤖 Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama-4-scout",
            "llama-4-maverick"
        ]
    )
    
    st.markdown("---")
    
    st.markdown("### 💡 Features")
    st.markdown("""
    - 🔍 **Web Search**  
    - 📚 **Wikipedia**  
    - 📄 **arXiv Papers**  
    - 📎 **PDF Q&A**
    """)
    
    st.markdown("---")
    
    st.caption("💬 Upload PDFs or ask questions to get started")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hi! 👋 I'm your AI assistant. I can search the web, Wikipedia, and arXiv papers. Upload PDFs to ask questions about your documents, or just ask me anything!"
    }]

# Enhanced File Upload Section
st.markdown("### 📎 Documents")
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload PDF documents to enable Q&A functionality"
)

# Process uploaded files
if uploaded_files and embeddings:
    if st.session_state.vectorstore is None or st.button("Re-process", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                documents = []
                temp_files = []
                
                for uploaded_file in uploaded_files:
                    # Use tempfile for better cloud compatibility
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)
                    
                    try:
                        loader = PyPDFLoader(tmp_file.name)
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
                
                if documents:
                    with st.spinner("Embedding..."):
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Create vectorstore
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings
                        )
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                    
                    st.success(f"✅ Processed {len(uploaded_files)} file(s) • {len(splits)} chunks ready")
                else:
                    st.warning("⚠️ No documents could be loaded")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif st.session_state.vectorstore is not None:
        st.info(f"📚 {len(uploaded_files)} file(s) ready for Q&A")
elif uploaded_files and not embeddings:
    st.error("❌ Embeddings not initialized")

# Enhanced Chat Interface
st.markdown("### 💬 Chat")
st.markdown("---")

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Enhanced Chat Input
if prompt := st.chat_input("💬 Ask anything..."):
    if not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process response
        with st.chat_message("assistant"):
            try:
                # Check if PDF Q&A is available
                if st.session_state.retriever is not None:
                    with st.spinner("Searching..."):
                        try:
                            llm_temp = ChatGroq(groq_api_key=api_key, model_name=model_name)
                            retriever = st.session_state.retriever
                            
                            # Retrieve relevant documents
                            docs = retriever.get_relevant_documents(prompt)
                            context = "\n\n".join([doc.page_content for doc in docs])
                            
                            # Create prompt template
                            prompt_template = ChatPromptTemplate.from_messages([
                                ("system", "You are a helpful assistant. Use the following context from uploaded documents to answer the question. If the context doesn't contain the answer, say so and provide a general response."),
                                ("human", "Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
                            ])
                            
                            formatted_prompt = prompt_template.format_messages(
                                context=context,
                                question=prompt
                            )
                            
                            response = llm_temp.invoke(formatted_prompt).content
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
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
                            
                            if LANGGRAPH_AVAILABLE and search_tools:
                                # Use modern LangGraph agent
                                agent = create_react_agent(llm, search_tools)
                                
                                # Invoke agent with messages format
                                result = agent.invoke({
                                    "messages": [HumanMessage(content=prompt)]
                                })
                                
                                # Extract response from agent output
                                if isinstance(result, dict) and "messages" in result:
                                    # Get the last message which should be the AI response
                                    last_message = result["messages"][-1]
                                    if hasattr(last_message, 'content'):
                                        response = last_message.content
                                    elif isinstance(last_message, dict) and 'content' in last_message:
                                        response = last_message['content']
                                    else:
                                        response = str(last_message)
                                elif isinstance(result, dict):
                                    # Try to find content in result
                                    response = result.get('output', result.get('content', str(result)))
                                else:
                                    response = str(result)
                            else:
                                # Fallback: direct tool calling without agent
                                response_parts = []
                                response_parts.append(f"I can help you search, but agent functionality requires langgraph. Please install it with: pip install langgraph")
                                response_parts.append("\n\nAlternatively, you can ask me questions and I'll try to answer directly.")
                                
                                # Try to use tools directly if available
                                if search_tools:
                                    try:
                                        # Simple tool execution for search
                                        if "search" in prompt.lower() or "find" in prompt.lower():
                                            search_result = search_tools[2].invoke({"query": prompt})
                                            response_parts.append(f"\n\nSearch result: {search_result}")
                                    except:
                                        pass
                                
                                response = " ".join(response_parts)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            response = f"I encountered an error: {str(e)}. Please try again."
                
                # Display and store response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Enhanced Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "Hi! 👋 I'm your AI assistant. I can search the web, Wikipedia, and arXiv papers. Upload PDFs to ask questions about your documents, or just ask me anything!"
        }]
        st.rerun()

with col3:
    st.markdown("""
    <div style="text-align: right; padding-top: 0.5rem;">
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
            Powered by <span style="color: #6366f1;">LangChain</span> & <span style="color: #8b5cf6;">Groq</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
