import os
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
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent, AgentType

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Tool setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
search_tools = load_tools(
    ["wikipedia", "arxiv", "duckduckgo-search"],
    api_wrappers={"wikipedia": wiki_wrapper, "arxiv": arxiv_wrapper}
)

# Streamlit UI
st.set_page_config(page_title="🔎LangChain Search Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 LangChain Chatbot with Search & Tools")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Select Model", ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I'm a chatbot who can search the web, Wikipedia, and arXiv. How can I help you today? ❤️"}]

pdf_qa = None
uploaded_files = st.file_uploader("Upload your files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temppdf.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf)
        documents.extend(loader.load())
        os.remove(temp_pdf)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    if api_key:
        llm_temp = ChatGroq(groq_api_key=api_key, model_name=model_name)
        pdf_qa = RetrievalQA.from_chain_type(llm=llm_temp, retriever=retriever, chain_type="stuff")

chat_container = st.container()
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask anything, e.g., latest AI papers or Wikipedia info"):
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar to use the chatbot.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        llm = ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)
        search_agent = initialize_agent(
            tools=search_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pdf_qa.run(prompt) if pdf_qa else search_agent.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

st.markdown(
    """
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 5px; color: #888;">
        Created with ❤️ by Dhruv
    </div>
    """,
    unsafe_allow_html=True
)
