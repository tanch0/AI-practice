from typing import Optional, List
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document


@dataclass
class AppConfig:
    """Application configuration."""

    page_title: str = "Ask your PDF"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model_name: str = "text-embedding-3-large"


@st.cache_resource
def get_openai_embeddings() -> OpenAIEmbeddings:
    """Initialize and cache OpenAI embeddings."""
    return OpenAIEmbeddings(model=AppConfig.model_name)


@st.cache_resource
def get_qa_chain():
    """Initialize and cache QA chain."""
    return load_qa_chain(OpenAI(), chain_type="stuff")


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
        embeddings = get_openai_embeddings()
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


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
        )
        return

    # Configure page
    st.set_page_config(page_title=AppConfig.page_title)
    st.header(AppConfig.page_title)

    # Initialize session state for knowledge base
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # File upload
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file:
        with st.spinner("Processing PDF..."):
            # Extract and process text
            text = extract_text_from_pdf(pdf_file)
            if text:
                chunks = create_text_chunks(text)
                st.session_state.knowledge_base = create_knowledge_base(chunks)
                st.success("PDF processed successfully!")

    # Query interface
    if st.session_state.knowledge_base:
        question = st.text_input("Ask your question:")
        if question:
            response = process_query(st.session_state.knowledge_base, question)
            if response:
                st.write(response)


if __name__ == "__main__":
    main()
