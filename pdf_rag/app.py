from typing import Optional, List
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document


@dataclass
class AppConfig:
    """Application configuration."""

    page_title: str = "PDF Chat Assistant"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"


@st.cache_resource
def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize and cache Gemini embeddings."""
    return GoogleGenerativeAIEmbeddings(model=AppConfig.embedding_model)


@st.cache_resource
def get_qa_chain():
    """Initialize and cache QA chain."""
    return load_qa_chain(ChatGoogleGenerativeAI(model=AppConfig.model_name), chain_type="stuff")


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def create_text_chunks(text: str) -> List[str]:
    """Split text into manageable chunks."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=AppConfig.chunk_size,
        chunk_overlap=AppConfig.chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def create_knowledge_base(chunks: List[str]) -> Optional[FAISS]:
    """Create FAISS knowledge base from text chunks."""
    try:
        embeddings = get_gemini_embeddings()
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating knowledge base: {str(e)}")
        return None


def process_query(knowledge_base: FAISS, question: str) -> Optional[str]:
    """Process user query and return response."""
    try:
        with st.spinner("Searching for answer..."):
            docs = knowledge_base.similarity_search(question)
            chain = get_qa_chain()
            return chain.run(input_documents=docs, question=question)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None


def initialize_session_state():
    """Initialize session state variables."""
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False


def display_chat_history():
    """Display the chat history in a conversational format."""
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(user_msg)
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(assistant_msg)


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Google API key not found. Please set GOOGLE_API_KEY in your environment."
        )
        return

    # Configure page
    st.set_page_config(
        page_title=AppConfig.page_title,
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()

    # Header
    st.header(AppConfig.page_title)
    st.markdown("Upload a PDF document and start asking questions about its content.")

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("PDF Upload")
        pdf_file = st.file_uploader("Upload your PDF", type="pdf")
        
        if pdf_file:
            with st.spinner("Processing PDF..."):
                # Extract and process text
                text = extract_text_from_pdf(pdf_file)
                if text:
                    chunks = create_text_chunks(text)
                    st.session_state.knowledge_base = create_knowledge_base(chunks)
                    st.session_state.pdf_processed = True
                    st.success("PDF processed successfully!")
                    
                    # Clear chat history when new PDF is uploaded
                    st.session_state.chat_history = []
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("Please upload a PDF file to start chatting.")
    else:
        # Display chat history
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.chat_history.append((prompt, ""))
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_query(st.session_state.knowledge_base, prompt)
                    if response:
                        st.write(response)
                        # Update the last assistant message in chat history
                        st.session_state.chat_history[-1] = (prompt, response)
                    else:
                        st.error("Sorry, I couldn't find an answer to your question.")
                        # Remove the failed conversation from history
                        st.session_state.chat_history.pop()


if __name__ == "__main__":
    main()
