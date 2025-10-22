import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY not found. Please set it in .env")
    exit()

# Setup loaders for different file types
loader = DirectoryLoader(
    './knowledge_base/',
    glob="**/*",
    loader_map={
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
    },
    show_progress=True,
    use_multithreading=True
)

# Load and split
print("Loading knowledge base...")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(f"Loaded and split {len(texts)} chunks.")

# Setup ChromaDB
db_path = "chroma_db"
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenAIEmbeddings()

# Get or create the collection
collection_name = "consultant_copilot"
collection = client.get_or_create_collection(
    name=collection_name, 
    embedding_function=embedding_function
)

# Ingest into ChromaDB
print("Ingesting documents into ChromaDB...")
# We add documents in batches of 100
for i in range(0, len(texts), 100):
    batch_texts = [doc.page_content for doc in texts[i:i+100]]
    batch_metadatas = [doc.metadata for doc in texts[i:i+100]]
    batch_ids = [f"doc_{i+j}" for j in range(len(batch_texts))]
    
    collection.add(
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

print(f"✅ Ingestion complete! {len(texts)} chunks added to '{collection_name}' at {db_path}")