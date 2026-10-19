# RAG: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

This repository implements and evaluates two Retrieval-Augmented Generation (RAG) pipelines over the *NASA Systems Engineering Handbook*.

---

## Architecture Overview

1. **Pipeline 1: Baseline Retrieval**
   - **Retriever:** Dense Bi-Encoder (`sentence-transformers/all-MiniLM-L6-v2`) retrieves the Top 10 chunks using cosine similarity.
   - **Context:** The Top 3 chunks are directly fed to the generator LLM.
   - **Generator:** `Qwen/Qwen2.5-3B-Instruct` generates Answer 1.

2. **Pipeline 2: Re-Ranking**
   - **First Stage:** The Bi-Encoder retrieves the initial Top 10 candidates.
   - **Second Stage:** A Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly scores all 10 `(query, chunk)` pairs with full cross-attention.
   - **Context:** Candidates are re-sorted by relevance score, and the new Top 3 chunks are fed to the generator LLM with the identical prompt structure.
   - **Generator:** `Qwen/Qwen2.5-3B-Instruct` generates Answer 2.

---

## Repository Structure

```text
├── RAG_Baseline_vs_Reranking.ipynb        # End-to-end executable notebook
├── requirements.txt                       # Project dependencies
├── README.md                              # Project documentation
│
└── outputs/                               # Evaluation results and artifacts
    ├── answer_comparison_and_metrics.csv  # Side-by-side answers, token F1 scores
    ├── retrieval_metrics_recomputed.csv   # Recall@3, MRR@3, nDCG@3
    ├── rank_shift_analysis.csv            # Candidate rankings (dense vs. re-ranked)
    ├── rank_shift_analysis_recomputed.csv # Rank shifts for relevant chunks
    ├── answer_token_f1.png                # Token F1 score comparison plot
    ├── rank_shifts.png                    # Scatter plot of candidate rank shifts
    ├── baseline_results_copy.jsonl        # Baseline answers, prompts, and chunks
    ├── reranked_results_copy.jsonl        # Re-ranked answers, prompts, and chunks
    ├── evaluation_questions_copy.json     # Evaluation benchmark questions
    ├── document_metadata.json             # Document chunking metadata
    └── document_chunks.jsonl              # Extracted text chunks corpus
```

---

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Notebook**:
   Open and execute `RAG_Baseline_vs_Reranking.ipynb` in Google Colab (recommended with GPU runtime) or locally with a CUDA-enabled GPU.
