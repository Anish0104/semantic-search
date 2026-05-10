# Semantic Search Project

This project implements a semantic search system using BM25, Bi-Encoders, and Cross-Encoder reranking.

## Structure

- `src/`: Core logic for data loading and retrieval.
- `eval/`: Evaluation metrics and scripts.
- `api/`: FastAPI based search interface.
- `data/`: Folder for datasets.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `uvicorn api.main:app --reload`
