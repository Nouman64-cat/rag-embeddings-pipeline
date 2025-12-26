# 🧬 RAG Embeddings Pipeline

A Streamlit-based application for uploading documents, generating embeddings using OpenAI, and storing them in a Pinecone vector database. This tool is designed to power the **Zygotrix Knowledge Base** for retrieval-augmented generation (RAG) workflows.

---

## ✨ Features

- 📄 **Document Upload** – Supports PDF and TXT file formats
- 🔀 **Text Chunking** – Automatically splits documents into optimized chunks
- 🧠 **OpenAI Embeddings** – Generates high-quality embeddings using `text-embedding-3-small`
- 🌲 **Pinecone Integration** – Stores vectors in a scalable, low-latency vector database
- 🎯 **Real-time Progress** – Visual feedback during document processing

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed on your system
- **OpenAI API Key** – [Get one here](https://platform.openai.com/api-keys)
- **Pinecone API Key** – [Get one here](https://www.pinecone.io/)
- A Pinecone index with **1536 dimensions** (to match OpenAI embeddings)

---

### 📦 Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-embeddings-pipeline.git
cd rag-embeddings-pipeline
```

#### 2. Set Up Environment Variables

Copy the example environment file and configure it with your actual API keys:

```bash
cp .env.example .env
```

Open the `.env` file in your editor and fill in your values:

```env
# API Keys (Required)
OPENAI_API_KEY=sk-proj-your-actual-openai-key
PINECONE_API_KEY=pcsk_your-actual-pinecone-key

# Pinecone Configuration (Required)
PINECONE_INDEX_NAME=your-index-name
PINECONE_HOST=https://your-index-name.svc.your-region.pinecone.io
```

#### 3. Create and Activate Virtual Environment

**🍎 macOS / 🐧 Linux (Ubuntu)**

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

**🪟 Windows**

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Command Prompt)
.venv\Scripts\activate.bat

# OR Activate virtual environment (PowerShell)
.venv\Scripts\Activate.ps1
```

> 💡 **Tip:** You'll know the virtual environment is active when you see `(.venv)` at the beginning of your terminal prompt.

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Launch the app** using `streamlit run app.py`
2. **Verify configuration** – Check the sidebar for API status (green checkmark means you're good!)
3. **Upload documents** – Drag and drop or browse for PDF/TXT files
4. **Click "Process and Upload"** – Watch as your documents are chunked, embedded, and stored
5. **Success!** 🎈 – Your documents are now searchable in your Pinecone index

---

## 📁 Project Structure

```
rag-embeddings-pipeline/
├── .env                  # Your environment variables (git-ignored)
├── .env.example          # Template for environment variables
├── .gitignore            # Git ignore rules
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── verify.py             # Verification script
└── README.md             # This file
```

---

## 🛠️ Troubleshooting

| Issue                           | Solution                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `❌ Missing .env configuration` | Ensure your `.env` file exists and contains all required keys                  |
| `Module not found` errors       | Make sure your virtual environment is activated and dependencies are installed |
| `Invalid API Key`               | Double-check your OpenAI and Pinecone API keys in `.env`                       |
| `Dimension mismatch`            | Ensure your Pinecone index has 1536 dimensions                                 |

---

## 📄 License

This project is part of the Zygotrix ecosystem.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Made with ❤️ for the Zygotrix Knowledge Base
</p>
