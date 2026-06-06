# Production RAG Application: "Ask My Docs"

A production-grade, domain-specific Retrieval-Augmented Generation (RAG) system implementing hybrid retrieval, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, strict citation enforcement, and a CI/CD-gated evaluation pipeline.

---

## 🏗️ System Architecture

The pipeline uses a multi-stage retrieval and synthesis strategy to ensure highly accurate, grounded answers:

```
                  ┌───────────────┐
                  │  User Query   │
                  └───────┬───────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌─────────────────────┐     ┌─────────────────────┐
 │ Dense Vector Search │     │ Sparse Lexical      │
 │ (ChromaDB +         │     │ (BM25 Retriever)    │
 │  all-MiniLM-L6-v2)  │     │                     │
 └──────────┬──────────┘     └──────────┬──────────┘
  [Top 10]  │                           │  [Top 20]
            └─────────────┬─────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Reciprocal Rank Fusion  │
             │      (RRF Merge)        │
             └────────────┬────────────┘
               [Top 10]   │
                          ▼
             ┌─────────────────────────┐
             │ Cross-Encoder Reranker  │
             │ (ms-marco-MiniLM-L-6-v2)│
             └────────────┬────────────┘
                [Top 5]   │
                          ▼
             ┌─────────────────────────┐
             │   Gemini-2.5-Flash      │
             │   (Strict Citations)    │
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Grounded Response +     │
             │ Source Attribution      │
             └─────────────────────────┘
```

### Key Stages
1. **Document Ingestion**: PDF files are loaded using `PyPDFLoader`, split into semantic chunks (size 500, overlap 50) using `RecursiveCharacterTextSplitter`.
2. **Hybrid Indexing**: Chunks are indexed in two ways:
   - **Dense**: Chroma vector store using local Hugging Face `all-MiniLM-L6-v2` embeddings.
   - **Sparse**: Lexical `BM25` index.
3. **Retrieval Merging (RRF)**: Reciprocal Rank Fusion merges results from both indices, resolving the trade-off between semantic search and keyword matches.
4. **Cross-Encoder Reranking**: Re-evaluates top 10 candidates with `ms-marco-MiniLM-L-6-v2` to select the top 5 context chunks, optimizing context relevancy.
5. **Grounded Synthesis**: Prompts Gemini-2.5-Flash using strict constraints: answer only using context, cite source/page for every sentence, and decline to answer if context is insufficient.

---

## 🚦 CI/CD Evaluation & Regression Gate

The project contains a built-in evaluation framework using **Ragas** and **Hugging Face Datasets** to verify output quality:
- **Faithfulness (Groundedness)**: Assesses whether the generated answer is strictly derived from the retrieved contexts (detects hallucinations).
- **Answer Relevancy**: Evaluates how well the generated response matches the user's initial question.
- **Regression Gate**: Comparing new scores against `eval/baseline.json`. If faithfulness or relevancy scores drop by more than **5%** (defined in `check_regression.py`), the CI/CD build fails.

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.10 or 3.11
- A Gemini API Key (obtain from Google AI Studio)

### 🔧 Installation
1. Clone the repository and navigate to the root directory:
   ```bash
   git clone <your-repo-url>
   cd rag-portfolio
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell/CMD):
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your API key:
   Copy `.env.example` to `.env` and fill in your Gemini API key:
   ```env
   GOOGLE_API_KEY=AIzaSy...
   ```

---

## 💻 Running the Application

### 1. Ingest Documents
Place your PDFs into the `data/` directory (e.g., the default `data/rag_paper.pdf`) and run ingestion to build search indices:
```bash
python ingest.py
```
This saves the Chroma database to `./chroma_db/` and the BM25 index to `./bm25_index.pkl`.

### 2. Run Diagnostic
Verify API connectivity to Gemini:
```bash
python test_llm.py
```

### 3. Query the RAG System
Run the RAG command-line interface to ask questions about your documents:
```bash
python rag.py "How does RAG-Token differ from RAG-Sequence in marginalization?"
```
Or run `python rag.py` interactively.

---

## 📊 Running Evaluations

Run evaluations against the golden dataset `eval/golden_qa.json`:
```bash
python eval/run_evals.py
```
This runs queries, scores answers using Ragas, and saves scores to `eval/latest_scores.json`.

To check for performance regression:
```bash
python eval/check_regression.py
```

---

## 📂 Project Structure

```
├── .github/workflows/
│   └── eval.yml          # GitHub Actions regression test gate
├── data/
│   └── rag_paper.pdf     # Source PDF documents
├── eval/
│   ├── baseline.json      # Target metrics baseline
│   ├── golden_qa.json     # Test dataset (Questions & Ground Truth)
│   ├── latest_scores.json # Saved scores from last evaluation
│   ├── run_evals.py       # Ragas evaluation runner
│   └── check_regression.py# Performance regression verifier
├── ingest.py             # Document loading, splitting, & indexing
├── rag.py                # Main RAG querying module (CLI interface)
├── reranker.py           # Cross-encoder reranking utility
├── test_llm.py           # Gemini API connectivity test script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── .gitignore            # Git exclusion rules
```
