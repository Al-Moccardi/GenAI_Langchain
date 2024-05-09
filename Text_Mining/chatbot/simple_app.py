import streamlit as st
import os
import tempfile
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

# CSS for styling the app
def local_css():
    st.markdown("""
        <style>
            .big-font {
                font-size:20px !important;
                color: #3498db;
            }
            .info-text {
                padding: 10px;
                background-color: #f0f2f6;
                border-left: 5px solid #3498db;
            }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if "uploaded_texts" not in st.session_state:
        st.session_state["uploaded_texts"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

def upload_and_process_files():
    uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True, key="file_uploader")
    if uploaded_files:
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension in [".pdf", ".docx", ".doc", ".txt"]:
                loader = {"pdf": PyPDFLoader, "docx": Docx2txtLoader, "doc": Docx2txtLoader, "txt": TextLoader}.get(file_extension[1:], PyPDFLoader)(temp_file_path)
                document_text = loader.load()
                st.session_state["uploaded_texts"].extend(document_text)
                os.remove(temp_file_path)

def setup_vector_store():
    if st.session_state["uploaded_texts"]:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=768, chunk_overlap=128, length_function=len)
        text_chunks = text_splitter.split_documents(st.session_state["uploaded_texts"])
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        vector_store = Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory="chroma_store")
        return vector_store
    return None

def create_conversational_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}), memory=memory)
    return chain

def display_chat():
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key="input")
            submit_button = st.form_submit_button(label="Send")
        
        if submit_button and user_input:
            with st.spinner("Generating response..."):
                result = st.session_state["chain"]({"question": user_input, "chat_history": st.session_state["chat_history"]})
                answer = result["answer"]
                st.session_state["chat_history"].append((user_input, answer))

    if st.session_state["chat_history"]:
        with reply_container:
            for query, response in st.session_state["chat_history"]:
                message(query, is_user=True, key=f"user_{query}", avatar_style="thumbs")
                message(response, key=f"resp_{response}", avatar_style="fun-emoji")

def main():
    local_css()  # Apply custom CSS styles
    st.title("Enhanced RAG ChatBot Using LangChain and ChatGPT")
    st.markdown("<p class='info-text'>Upload your documents in the sidebar and ask questions related to them.</p>", unsafe_allow_html=True)
    st.sidebar.title("Document Management")
    initialize_session_state()
    upload_and_process_files()
    if "chain" not in st.session_state or not st.session_state["chain"]:
        vector_store = setup_vector_store()
        if vector_store:
            st.session_state["chain"] = create_conversational_chain(vector_store)
    display_chat()

if __name__ == "__main__":
    main()
