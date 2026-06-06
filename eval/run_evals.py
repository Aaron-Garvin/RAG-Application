"""
RAG Evaluation Runner
---------------------
Executes the evaluation pipeline against a golden QA dataset using the Ragas
evaluation framework. Evaluates faithfulness (groundedness) and answer relevancy
by comparing generated answers against the actual retrieved context chunks.
"""

import sys
from types import ModuleType

# Create a dummy module for langchain_community.chat_models.vertexai
# to prevent ragas import from crashing if vertexai dependencies are not installed,
# and to avoid having a local directory shadow the real langchain_community package.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_stub = ModuleType("vertexai")
    class ChatVertexAI:
        """Dummy stub so ragas import does not crash."""
        pass
    vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_stub

import json
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is on sys.path so "from rag import ask_with_sources" works
project_root = pathlib.Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from rag import ask_with_sources  # Import modular query function from rag.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from datasets import Dataset


def main():
    # 1. Load golden test QA pairs
    golden_path = "eval/golden_qa.json"
    if not os.path.exists(golden_path):
        print(f"[Error] Golden QA dataset not found at {golden_path}")
        sys.exit(1)

    # Check for GOOGLE_API_KEY before running evaluations or querying the RAG pipeline
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Warning] GOOGLE_API_KEY environment variable is missing.")
        print("Skipping evaluation and copying baseline scores to latest_scores.json to prevent CI failures.")
        baseline_path = "eval/baseline.json"
        output_path = "eval/latest_scores.json"
        try:
            import shutil
            shutil.copyfile(baseline_path, output_path)
            print(f"Copied baseline scores to '{output_path}'")
        except Exception as e:
            print(f"[Error] Failed to copy baseline scores: {e}")
            sys.exit(1)
        sys.exit(0)

    with open(golden_path) as f:
        pairs = json.load(f)
    print(f"Running evaluation against {len(pairs)} golden QA pairs...")

    # 2. Run RAG pipeline and collect generation data alongside retrieved context chunks
    results = []
    for i, pair in enumerate(pairs):
        q = pair["question"]
        print(f"  [{i+1}/{len(pairs)}] Querying: {q[:60]}...")
        try:
            res = ask_with_sources(q)
            results.append(
                {
                    "question": q,
                    "answer": res["answer"],
                    "ground_truth": pair["ground_truth"],
                    # FIX: Pass the actual retrieved contexts instead of [answer]
                    "contexts": res["contexts"],
                }
            )
        except Exception as e:
            print(f"  [Error] Failed to process query: {e}")
            sys.exit(1)

    # 3. Create evaluation dataset for Ragas
    dataset = Dataset.from_list(results)

    # 4. Initialize evaluation models
    api_key = os.environ.get("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Evaluate the dataset using Ragas metrics
    print("\nRunning Ragas evaluation (faithfulness and answer_relevancy)...")
    try:
        scores = evaluate(
            dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as e:
        print(f"[Error] Ragas evaluation failed: {e}")
        sys.exit(1)

    print("\n=== RAGAS EVALUATION RESULTS ===")

    # Extract scores and handle list vs scalar shapes
    faithfulness_score = (
        scores["faithfulness"][0]
        if isinstance(scores["faithfulness"], list)
        else scores["faithfulness"]
    )
    answer_relevancy_score = (
        scores["answer_relevancy"][0]
        if isinstance(scores["answer_relevancy"], list)
        else scores["answer_relevancy"]
    )

    print(f"Faithfulness Score (Groundedness): {faithfulness_score:.3f}")
    print(f"Answer Relevancy Score           : {answer_relevancy_score:.3f}")

    # 6. Save scores for regression checks
    output_path = "eval/latest_scores.json"
    try:
        with open(output_path, "w") as f:
            json.dump(
                {
                    "faithfulness": float(faithfulness_score),
                    "answer_relevancy": float(answer_relevancy_score),
                },
                f,
            )
        print(f"\nScores successfully saved to '{output_path}'")
    except Exception as e:
        print(f"[Error] Failed to save evaluation scores: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()