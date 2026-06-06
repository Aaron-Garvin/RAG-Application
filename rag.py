"""
RAG Query Engine
----------------
Implements hybrid retrieval (BM25 + Chroma Vector DB), merges results using
Reciprocal Rank Fusion (RRF), reranks candidates with a Cross-Encoder model,
and synthesizes the final answer using Gemini-2.5-Flash with source attribution.
"""

import os
import pickle
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# UPDATED IMPORT: use the new module path for HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from reranker import rerank

# Lazy-loaded resources
_vectorstore = None
_bm25 = None


def get_resources():
    """Lazily load and cache embedding models, vector store, and BM25 index."""
    global _vectorstore, _bm25
    if _vectorstore is None or _bm25 is None:
        load_dotenv()

        # Check for required database files
        if not os.path.exists("./chroma_db") or not os.path.exists("bm25_index.pkl"):
            raise FileNotFoundError(
                "Search indices not found! Please run 'python ingest.py' first to build indices."
            )

        print("Loading search indices and embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )

        with open("bm25_index.pkl", "rb") as f:
            _bm25 = pickle.load(f)

    return _vectorstore, _bm25


def rrf_merge(list_a, list_b, k=60):
    """
    Merge two lists of retrieved documents using Reciprocal Rank Fusion (RRF).
    Uses document page content prefix to deduplicate and rank documents.
    """
    scores = {}
    for rank, doc in enumerate(list_a):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        scores[key + "__doc"] = doc

    for rank, doc in enumerate(list_b):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key + "__doc" not in scores:
            scores[key + "__doc"] = doc

    ranked = sorted(
        [(v, k) for k, v in scores.items() if not k.endswith("__doc")],
        reverse=True,
    )
    return [scores[key_str + "__doc"] for _, key_str in ranked[:10]]


def ask_with_sources(question: str) -> dict:
    """
    Queries the hybrid RAG system and returns the generated answer alongside
    the retrieved contexts and document metadata.
    """
    # 1. Ensure resources are loaded
    vectorstore, bm25 = get_resources()

    # 2. Check for LLM API key
    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please create a '.env' file containing your API key."
        )

    # 3. Hybrid Retrieve
    vec_results = vectorstore.similarity_search(question, k=10)
    bm25_results = bm25.invoke(question)

    # 4. Merge results using Reciprocal Rank Fusion
    merged = rrf_merge(vec_results, bm25_results)

    # 5. Rerank candidates using Cross-Encoder
    reranked = rerank(question, merged, top_n=5)

    # 6. Format retrieved context for the prompt
    context_str = "\n\n".join(
        [
            f"[Source: {d.metadata.get('source', 'doc')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in reranked
        ]
    )

    # 7. Query LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

    prompt = f"""You are a document assistant.
Rules you MUST follow:
1. Answer ONLY using information from the Context below.
2. Every sentence must end with [Source: filename, page N].
3. If the answer is not in the context, say:
    "I cannot find this in the provided documents."
4. Never make up information.

Context:
{context_str}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "contexts": [d.page_content for d in reranked],
        "sources": [
            {
                "source": os.path.basename(d.metadata.get("source", "doc")),
                "page": d.metadata.get("page", "?"),
            }
            for d in reranked
        ],
    }


def ask(question: str) -> str:
    """
    Wraps ask_with_sources to return only the answer text for backward compatibility.
    """
    res = ask_with_sources(question)
    return res["answer"]


if __name__ == "__main__":
    import sys

    load_dotenv()

    # Check if a question was passed as arguments
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Ask a question: ")

    print(f"\nQuestion: {q}")
    try:
        res = ask_with_sources(q)
        print("\nAnswer:")
        print(res["answer"])
        print("\nRetrieved Sources:")
        for i, src in enumerate(res["sources"]):
            print(f"  [{i+1}] {src['source']} (Page {src['page']})")
    except Exception as e:
        print(f"\n[Error] {e}")