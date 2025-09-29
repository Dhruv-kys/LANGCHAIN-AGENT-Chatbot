import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(name="Search")
tools = [arxiv, wiki, search]

st.set_page_config(page_title="🔎LangChain Search Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 LangChain Chatbot with Search & Tools")
st.secrets["HF_TOKEN"]

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Select Model", ["gemma2-9b-it","deepseek-r1-distill-llama-70b","llama-3.3-70b-versatile"])
    

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot who can search the web, Wikipedia, and arXiv. How can I help you today? ❤️"}
    ]
    
pdf_qa = None
uploaded_files = st.file_uploader("Upload your files",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf= f"./temppdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm_temp = ChatGroq(groq_api_key=api_key,model_name=model_name)
    pdf_qa = RetrievalQA.from_chain_type(
    llm=llm_temp,
    retriever=retriever,
    chain_type="stuff"  # combines all retrieved chunks into one prompt
    )
        
        
        
chat_container = st.container()
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask anything, e.g., latest AI papers or Wikipedia info"):

    # Check if API key is provided
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar to use the chatbot.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Initialize LLM
        llm = ChatGroq(groq_api_key=api_key, model=model_name, streaming=True)
        
        # Initialize agent for web/Wikipedia/arXiv search
        search_agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            handle_parsing_errors=True
        )
        
        # Generate assistant response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            if pdf_qa:
                # If PDF uploaded, always search from PDF
                response = pdf_qa.run(prompt)
            else:
                # Otherwise use web/Wikipedia/arXiv search
                response = search_agent.run(input=prompt, callbacks=[st_cb])
            
            # Append and display response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


st.markdown(
    """
    <div style="position: fixed; bottom: 0; width: 100%; 
                text-align: center; padding: 5px; color: #888;">
        Created with ❤️ by Dhruv
    </div>
    """,
    unsafe_allow_html=True
)
