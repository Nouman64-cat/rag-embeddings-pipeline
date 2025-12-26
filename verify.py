import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Setup
load_dotenv()

# 2. Connect to Pinecone & OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Check Stats
stats = index.describe_index_stats()
print(f"\n📊 Index Stats:\n{stats}\n")

# 4. Run a Test Query
# (Since you uploaded a "Prompt Engineering" PDF, let's ask about that)
query = "What are the key principles of prompt engineering?"

print(f"🔎 Testing Retrieval for: '{query}'...")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

results = vectorstore.similarity_search(query, k=2)

print("\n✅ Top 2 Results from your PDF:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"📄 Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"📝 Content Snippet: {doc.page_content[:200]}...") # Show first 200 chars