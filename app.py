import streamlit as st
import os
import tempfile
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# --- LangChain & Pinecone Imports ---
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Environment Variables
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- Helper Functions ---

def get_vectorstore():
    """
    Initializes the connection to the Pinecone Index with OpenAI Embeddings.
    """
    logger.info("🔄 Initializing OpenAI Embeddings (model: text-embedding-3-small)...")
    
    # Initialize OpenAI Embeddings (1536 Dimensions)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    
    logger.info(f"🌲 Connecting to Pinecone index: {INDEX_NAME}...")
    
    # Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    logger.info("✅ Successfully connected to Pinecone!")
    return vectorstore

def process_file(uploaded_file):
    """
    Saves uploaded file to temp, loads it, fixes metadata, and cleans up.
    """
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    logger.info(f"📂 Loading file: {uploaded_file.name} ({file_size_kb:.2f} KB)")
    
    # Create a temp file with the correct extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Select Loader based on file extension
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            logger.info(f"   📄 Using PDF loader for {uploaded_file.name}")
        else:
            loader = TextLoader(tmp_file_path)
            logger.info(f"   📝 Using Text loader for {uploaded_file.name}")
        
        documents = loader.load()
        logger.info(f"   ✅ Loaded {len(documents)} page(s)/document(s) from {uploaded_file.name}")
        
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

def chunk_data(docs, filename=""):
    """
    Splits text into manageable chunks.
    """
    logger.info(f"   ✂️  Splitting {filename} into chunks (size=1000, overlap=200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Characters per chunk
        chunk_overlap=200, # Overlap to preserve context between chunks
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"   ✅ Created {len(chunks)} chunks from {filename}")
    return chunks

# --- Main Streamlit UI ---

st.set_page_config(page_title="Zygotrix Ingest", page_icon="🧬")

st.title("🧬 Zygotrix Knowledge Base")
st.write("Upload documents to embed and store them in the `zygotrix-embeddings` index.")

# Sidebar status check
st.sidebar.header("Configuration")
if PINECONE_API_KEY and OPENAI_API_KEY and INDEX_NAME:
    st.sidebar.success("✅ API Keys Loaded")
    st.sidebar.info(f"Target Index: {INDEX_NAME}")
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"   Target Index: {INDEX_NAME}")
else:
    st.sidebar.error("❌ Missing .env configuration")
    logger.error("❌ Missing .env configuration - check your API keys!")
    st.stop() # Stop execution if keys are missing

# File Uploader
uploaded_files = st.file_uploader(
    "Select files (PDF or TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

# Process Button
if uploaded_files and st.button("Process and Upload"):
    
    # --- Processing Start ---
    total_files = len(uploaded_files)
    total_chunks_processed = 0
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info(f"🚀 STARTING PROCESSING: {total_files} file(s)")
    logger.info("=" * 60)
    
    vectorstore = get_vectorstore()
    progress_bar = st.progress(0)
    
    # Create a log display area in the sidebar
    st.sidebar.header("📊 Processing Log")
    log_container = st.sidebar.empty()
    
    for idx, file in enumerate(uploaded_files):
        file_start_time = time.time()
        current_file_num = idx + 1
        remaining_files = total_files - current_file_num
        progress_pct = (idx / total_files) * 100
        
        logger.info("-" * 60)
        logger.info(f"📁 FILE {current_file_num}/{total_files} ({progress_pct:.1f}% complete)")
        logger.info(f"   Processing: {file.name}")
        logger.info(f"   Remaining: {remaining_files} file(s)")
        
        # Create a status container for this specific file
        status_text = st.empty()
        status_text.text(f"Processing {file.name}... ({current_file_num}/{total_files})")
        
        # Update sidebar log
        log_container.markdown(f"""
        **Current Progress:**
        - 📄 File: `{file.name}`
        - 📊 Progress: {current_file_num}/{total_files}
        - ⏳ Remaining: {remaining_files} file(s)
        """)
        
        try:
            # 1. Load File
            raw_docs = process_file(file)
            
            # 2. Split Text
            chunks = chunk_data(raw_docs, file.name)
            total_chunks_processed += len(chunks)
            status_text.text(f"✂️ {file.name}: Split into {len(chunks)} chunks. Uploading...")
            
            # 3. Add to Pinecone
            logger.info(f"   🌲 Uploading {len(chunks)} chunks to Pinecone...")
            upload_start = time.time()
            vectorstore.add_documents(chunks)
            upload_duration = time.time() - upload_start
            
            file_duration = time.time() - file_start_time
            logger.info(f"   ⏱️  Upload completed in {upload_duration:.2f}s")
            logger.info(f"   ✅ File processed in {file_duration:.2f}s total")
            
            st.success(f"✅ Successfully uploaded {file.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"   ❌ ERROR processing {file.name}: {str(e)}")
            st.error(f"Error processing {file.name}: {e}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_files)
    
    # --- Processing Complete ---
    total_duration = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("🎉 PROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📊 SUMMARY:")
    logger.info(f"   • Total files processed: {total_files}")
    logger.info(f"   • Total chunks created: {total_chunks_processed}")
    logger.info(f"   • Total time: {total_duration:.2f}s")
    logger.info(f"   • Average time per file: {total_duration/total_files:.2f}s")
    logger.info("=" * 60)
    
    # Update final sidebar status
    log_container.markdown(f"""
    **✅ Processing Complete!**
    - 📁 Files processed: {total_files}
    - 📦 Total chunks: {total_chunks_processed}
    - ⏱️ Total time: {total_duration:.2f}s
    - 📈 Avg per file: {total_duration/total_files:.2f}s
    """)
    
    st.balloons()
    st.success(f"All operations complete! Processed {total_files} files ({total_chunks_processed} chunks) in {total_duration:.2f}s")