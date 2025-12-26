import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- LangChain & Pinecone Imports ---
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Environment Variables
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- Helper Functions ---

def get_vectorstore():
    """
    Initializes the connection to the Pinecone Index with OpenAI Embeddings.
    """
    # Initialize OpenAI Embeddings (1536 Dimensions)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    
    # Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

def process_file(uploaded_file):
    """
    Saves uploaded file to temp, loads it, fixes metadata, and cleans up.
    """
    # Create a temp file with the correct extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Select Loader based on file extension
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)
        
        documents = loader.load()
        
        # --- METADATA FIX ---
        # Overwrite the 'source' metadata to be the actual filename,
        # not the temporary system path (e.g., /tmp/tmp123.pdf)
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            
    finally:
        # Clean up: Delete the temp file from disk
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            
    return documents

def chunk_data(docs):
    """
    Splits text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Characters per chunk
        chunk_overlap=200, # Overlap to preserve context between chunks
        add_start_index=True
    )
    return text_splitter.split_documents(docs)

# --- Main Streamlit UI ---

st.set_page_config(page_title="Zygotrix Ingest", page_icon="🧬")

st.title("🧬 Zygotrix Knowledge Base")
st.write("Upload documents to embed and store them in the `zygotrix-embeddings` index.")

# Sidebar status check
st.sidebar.header("Configuration")
if PINECONE_API_KEY and OPENAI_API_KEY and INDEX_NAME:
    st.sidebar.success("✅ API Keys Loaded")
    st.sidebar.info(f"Target Index: {INDEX_NAME}")
else:
    st.sidebar.error("❌ Missing .env configuration")
    st.stop() # Stop execution if keys are missing

# File Uploader
uploaded_files = st.file_uploader(
    "Select files (PDF or TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

# Process Button
if uploaded_files and st.button("Process and Upload"):
    
    vectorstore = get_vectorstore()
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        # Create a status container for this specific file
        status_text = st.empty()
        status_text.text(f"Processing {file.name}...")
        
        try:
            # 1. Load File
            raw_docs = process_file(file)
            
            # 2. Split Text
            chunks = chunk_data(raw_docs)
            status_text.text(f"✂️ {file.name}: Split into {len(chunks)} chunks. Uploading...")
            
            # 3. Add to Pinecone
            # This automatically handles embedding generation and upsert
            vectorstore.add_documents(chunks)
            
            st.success(f"✅ Successfully uploaded {file.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_files)
        
    st.balloons()
    st.success("All operations complete!")