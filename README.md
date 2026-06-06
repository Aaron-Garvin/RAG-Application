# 📚 Ask My Docs — Production RAG System

> A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you ask natural language questions about your own PDF documents and receive accurate, cited answers — powered by hybrid search, intelligent reranking, and Gemini 2.5 Flash.

---

## ✨ What Does This Do?

You drop in a PDF. You ask a question. You get a grounded answer with source citations — no hallucinations, no guessing.

Under the hood, it runs a multi-stage pipeline:

1. Searches your document using **two strategies simultaneously** (semantic meaning + keyword matching)
2. **Merges and reranks** results to surface the most relevant passages
3. Feeds the best context to **Gemini 2.5 Flash**, which is strictly instructed to only answer from what it found — and to cite every sentence

---

## 🗺️ How the Pipeline Works

```
Your Question
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
Dense Vector Search              Sparse BM25 Search
(semantic similarity)            (keyword matching)
   Top 10 results                   Top 20 results
      │                                  │
      └──────────────┬───────────────────┘
                     ▼
         Reciprocal Rank Fusion (RRF)
         Merges both result lists smartly
                  Top 10
                     │
                     ▼
         Cross-Encoder Reranker
         Scores each chunk against your query
                  Top 5
                     │
                     ▼
         Gemini 2.5 Flash
         Answers only from context, cites sources
                     │
                     ▼
         ✅ Grounded Answer + Source Citations
```

**Why hybrid search?** Pure semantic search misses exact keywords. Pure keyword search misses paraphrased meaning. Together, they cover both — and RRF merges them without needing manual weight tuning.

---

## ⚡ Quick Start (5 Steps)

### Prerequisites

- Python **3.10 or 3.11**
- A free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/)

---

### Step 1 — Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd rag-portfolio

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add Your API Key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```env
GOOGLE_API_KEY=AIzaSy...your-key-here...
```

### Step 4 — Add Your PDF and Ingest It

Drop your PDF into the `data/` folder (a sample `rag_paper.pdf` is included), then run:

```bash
python ingest.py
```

This builds two search indices saved locally:
- `./chroma_db/` — vector index for semantic search
- `./bm25_index.pkl` — keyword index for lexical search

> ⏱️ First run takes a minute — it's downloading the embedding model locally.

### Step 5 — Ask a Question

```bash
python rag.py "How does RAG-Token differ from RAG-Sequence in marginalization?"
```

Or run it interactively (no argument needed):

```bash
python rag.py
```

---

## 🔍 Verifying Your Setup

Before querying, confirm your Gemini API key is working:

```bash
python test_llm.py
```

You should see a successful response from the model. If it fails, double-check your `.env` file.

---

## 📊 Running Evaluations

The system includes a built-in quality evaluation framework using **Ragas**.

### Run the Evaluation Suite

```bash
python eval/run_evals.py
```

This runs a set of pre-defined questions from `eval/golden_qa.json` through the full pipeline, scores the answers, and saves results to `eval/latest_scores.json`.

**What gets measured:**

| Metric | What It Checks |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (Detects hallucinations) |
| **Answer Relevancy** | Does the answer actually address the question asked? |

### Check for Regressions

```bash
python eval/check_regression.py
```

This compares your latest scores against `eval/baseline.json`. If either metric drops by more than **5%**, it exits with an error — which also gates the CI/CD pipeline (see `.github/workflows/eval.yml`).

---

## 🧪 End-to-End Test Checklist

Use this to verify everything is working after setup:

```
[ ] python test_llm.py           → Should print a valid Gemini response
[ ] python ingest.py             → Should create chroma_db/ and bm25_index.pkl
[ ] python rag.py "your question" → Should return an answer with citations
[ ] python eval/run_evals.py     → Should produce eval/latest_scores.json
[ ] python eval/check_regression.py → Should pass with no errors
```

---

## 📁 Project Structure

```
rag-portfolio/
│
├── data/
│   └── rag_paper.pdf          ← Put your PDFs here
│
├── eval/
│   ├── golden_qa.json         ← Test questions + expected answers
│   ├── baseline.json          ← Minimum acceptable scores
│   ├── latest_scores.json     ← Output from last eval run
│   ├── run_evals.py           ← Runs Ragas evaluation
│   └── check_regression.py   ← Fails if scores drop > 5%
│
├── .github/workflows/
│   └── eval.yml               ← CI/CD regression gate
│
├── ingest.py                  ← Loads PDFs, builds search indices
├── rag.py                     ← Main query interface (CLI)
├── reranker.py                ← Cross-encoder reranking logic
├── test_llm.py                ← Gemini connectivity check
├── requirements.txt
├── .env.example               ← Copy to .env and fill in your API key
└── .gitignore
```

---

## 🛠️ Troubleshooting

**"No module named X"** — Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**"API key invalid"** — Verify your key is correctly copied in `.env` (no extra spaces or quotes). Run `python test_llm.py` to isolate the issue.

**"No documents found"** — Ensure your PDF is inside the `data/` folder and you've run `python ingest.py` after adding it.

**Slow first ingest** — The `all-MiniLM-L6-v2` embedding model downloads on first run (~90MB). Subsequent runs use the cached version.

**Evaluation scores unexpectedly low** — Try re-ingesting with `python ingest.py` to rebuild fresh indices, then re-run evals.

---

## 🧱 Tech Stack

| Component | Technology |
|---|---|
| Vector Store | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (local, via HuggingFace) |
| Keyword Search | BM25 |
| Reranker | `ms-marco-MiniLM-L-6-v2` (Cross-Encoder) |
| LLM | Gemini 2.5 Flash |
| Evaluation | Ragas + HuggingFace Datasets |
| CI/CD | GitHub Actions |
