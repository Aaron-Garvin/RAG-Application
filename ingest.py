"""
Document Ingestion Script
-------------------------
Loads PDF documents from the `data/` directory, splits them into semantic chunks,
and generates both a dense vector store (ChromaDB) and a sparse lexical index (BM25)
to support hybrid retrieval.
"""

import os
import glob
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# Load environment variables (useful if embeddings or loaders need keys, e.g., OpenAI)
load_dotenv()


def main():
    # 1. Search for PDF files in the data directory
    pdf_paths = glob.glob('data/*.pdf')
    if not pdf_paths:
        print("[Error] No PDF files found in the 'data/' directory.")
        print("Please place your source PDFs in 'data/' and run ingestion again.")
        return

    print(f"Found {len(pdf_paths)} PDF file(s) to ingest.")
    
    # 2. Load documents
    docs = []
    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[Error] Failed to load {pdf_path}: {e}")
            return
            
    print(f"Successfully loaded {len(docs)} pages.")

    # 3. Split documents into semantic chunks
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Generate dense vector store (ChromaDB)
    # Using 'sentence-transformers/all-MiniLM-L6-v2' for local fast embedding generation
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Initializing embedding model: {embedding_model}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    print("Generating ChromaDB embeddings and saving to './chroma_db'...")
    try:
        vectorstore = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory='./chroma_db'
        )
        print("ChromaDB vector store successfully saved.")
    except Exception as e:
        print(f"[Error] Failed to create ChromaDB vector store: {e}")
        return

    # 5. Generate sparse lexical index (BM25Retriever)
    print("Building BM25 lexical index...")
    try:
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 20  # Retrieve top 20 candidates for later RRF merging
        
        with open('bm25_index.pkl', 'wb') as f:
            pickle.dump(bm25, f)
        print("BM25 index successfully saved to 'bm25_index.pkl'.")
    except Exception as e:
        print(f"[Error] Failed to build or save BM25 index: {e}")
        return

    print("\nIngestion pipeline completed successfully! Ready for queries.")


if __name__ == "__main__":
    main()