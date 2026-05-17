# Two-Stage Semantic Search

Semantic search pipeline using bi-encoder retrieval and cross-encoder reranking, evaluated on MS MARCO. Outperforms BM25 by 143% NDCG@10.

## Results

Evaluated on 100K passages, 34 queries from the MS MARCO validation set.

| System | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 (baseline) | 0.2844 | 0.2256 | 0.7941 |
| Bi-encoder (MiniLM-L6) | 0.6154 | 0.5403 | 0.9706 |
| Bi-encoder + Reranker | **0.692** | **0.6126** | **0.9706** |

- Bi-encoder vs BM25: **+116.4% NDCG@10**
- Reranker vs Bi-encoder: **+12.4% NDCG@10**
- Full pipeline vs BM25: **+143.3% NDCG@10**

![Eval Output](results/eval_output.png)

## Quick Start

```bash
git clone https://github.com/Anish0104/semantic-search.git
cd semantic-search
pip install -r requirements.txt

# Run evaluation (downloads MS MARCO data on first run)
python eval/run_eval.py

# Start API server
uvicorn api.main:app --reload
```

## Project Structure


```
semantic-search/
├── src/
│   ├── biencoder_retriever.py   # Qdrant-backed dense retrieval with MiniLM-L6
│   ├── bm25_retriever.py        # BM25 baseline using rank-bm25
│   ├── reranker.py              # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   └── data_loader.py           # MS MARCO data loading and preprocessing
├── eval/
│   ├── run_eval.py              # End-to-end evaluation across all three systems
│   └── metrics.py               # NDCG@10, MRR@10, Recall@K implementations
├── api/
│   └── main.py                  # FastAPI endpoints for search and benchmarking
├── data/
│   └── msmarco_100000p_200q.json  # 100K passages, 200 queries (MS MARCO val)
├── results/                     # Saved eval outputs and plots
├── Dockerfile
└── requirements.txt
```

## How It Works

The pipeline runs in two stages to balance recall and precision:

**Stage 1 — Bi-encoder retrieval (recall):** The query and all passages are embedded independently using `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in Qdrant and retrieved via approximate nearest-neighbor search. This stage fetches the top 100 candidates in milliseconds, but cosine similarity on independent embeddings misses subtle relevance signals.

**Stage 2 — Cross-encoder reranking (precision):** The query and each of the 100 candidates are passed *together* through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike the bi-encoder, the cross-encoder performs full attention across both texts simultaneously, producing a fine-grained relevance score. The top 10 results are returned.

**Why this beats BM25:** BM25 relies on exact keyword overlap. The bi-encoder captures semantic meaning (paraphrases, synonyms) at scale. The reranker then applies expensive, accurate relevance scoring only where it matters — the 100 candidates already likely to be relevant — making the system both fast and precise.

```
Query
  │
  ▼
[Stage 1] Bi-encoder (MiniLM-L6) + Qdrant
  │  → top-100 candidates via ANN vector search
  │
  ▼
[Stage 2] Cross-encoder (ms-marco-MiniLM-L-6-v2)
  │  → reranks top-100, returns top-10
  │
  ▼
FastAPI endpoint @ <80ms p99
```

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Retrieval | sentence-transformers + Qdrant | Dense embedding + ANN vector search |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Query-passage joint relevance scoring |
| Baseline | rank-bm25 | Sparse keyword retrieval for comparison |
| Evaluation | NDCG@10, MRR@10, Recall@100 | Standard IR metrics (TREC-style) |
| Serving | FastAPI | REST API with benchmark endpoint |
| Containerization | Docker | Reproducible deployment |
| Dataset | MS MARCO passage retrieval | 8.8M passage corpus, TREC-style qrels |
| Hardware | Apple Silicon MPS / CUDA / CPU | Auto-detected; 3-4x speedup on MPS |

## API

```bash
uvicorn api.main:app --reload
```

**`POST /search`**
```json
{
  "query": "how does machine learning work",
  "mode": "reranker",
  "top_k": 10
}
```

**`GET /benchmark?query=what+is+deep+learning`**  
Returns p50 and p99 latency for all three retrieval modes.

**`GET /health`**

## Docker

```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

## Apple Silicon

MPS is auto-detected at startup:

```
Device: Apple Silicon MPS (GPU)
Encoding 100000 passages with sentence-transformers/all-MiniLM-L6-v2 on mps...
```

Gives 3-4x encoding speedup over CPU. Falls back to CUDA or CPU automatically.


