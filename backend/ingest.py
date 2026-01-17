import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/UET_Prospectus.pdf")
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")

def ingest_data():
    print("🚀 Starting Data Ingestion...")
    
    # 1. Load PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Please add the file.")
    
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(splits)} chunks.")

    # 3. Create Vector Store
    # --- FIX: Switch to a dedicated embedding model ---
    print("⏳ Generating Embeddings (this may take a moment)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    
    if os.path.exists(DB_PATH):
        print("🗑️ Clearing old database...")
        import shutil
        shutil.rmtree(DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"💾 Vector Database saved to {DB_PATH}")

if __name__ == "__main__":
    ingest_data()