from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    print("Error: OPENAI_API_KEY not found. Please set it in .env")
    exit()

print("Loading knowledge base...")
kb_path = './knowledge_base/'

# Load .txt files
txt_loader = DirectoryLoader(
    kb_path,
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True
)
txt_documents = txt_loader.load()

# Load .pdf files
pdf_loader = DirectoryLoader(
    kb_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
pdf_documents = pdf_loader.load()

# Combine the documents
documents = txt_documents + pdf_documents
if not documents:
    print("No documents found in ./knowledge_base/. Please add .txt or .pdf files.")
    exit()
    
print(f"Loaded {len(documents)} total documents.")

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(f"Split into {len(texts)} chunks.")

# --- Simplified Ingestion using LangChain's Chroma wrapper ---
print("Ingesting documents into ChromaDB...")
db_path = "chroma_db"
collection_name = "consultant_copilot"
embedding_function = OpenAIEmbeddings()

# This one command creates the db, embeds the docs, and saves it
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embedding_function,
    persist_directory=db_path,
    collection_name=collection_name
)

print(f"✅ Ingestion complete! {len(texts)} chunks added to '{collection_name}' at {db_path}")
