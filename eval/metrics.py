import numpy as np
from typing import List, Dict, Set

def recall_at_k(retrieved: List[Dict], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_ids = {r["id"] for r in retrieved[:k]}
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)

def mrr_at_k(retrieved: List[Dict], relevant_ids: Set[str], k: int) -> float:
    for rank, result in enumerate(retrieved[:k], start=1):
        if result["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: List[Dict], relevant_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for rank, result in enumerate(retrieved[:k], start=1):
        if result["id"] in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retriever(retriever, queries: List[Dict], top_k: int = 10, retrieve_k: int = 100) -> Dict:
    ndcg_scores, mrr_scores, recall_scores = [], [], []
    for q in queries:
        relevant = set(q["relevant_passage_ids"])
        results = retriever.retrieve(q["text"], top_k=retrieve_k)
        ndcg_scores.append(ndcg_at_k(results, relevant, k=top_k))
        mrr_scores.append(mrr_at_k(results, relevant, k=top_k))
        recall_scores.append(recall_at_k(results, relevant, k=retrieve_k))
    return {f"NDCG@{top_k}": round(np.mean(ndcg_scores), 4), f"MRR@{top_k}": round(np.mean(mrr_scores), 4), f"Recall@{retrieve_k}": round(np.mean(recall_scores), 4)}

def evaluate_reranker(biencoder, reranker, queries: List[Dict], retrieve_k: int = 100, top_k: int = 10) -> Dict:
    ndcg_scores, mrr_scores, recall_scores = [], [], []
    for q in queries:
        relevant = set(q["relevant_passage_ids"])
        candidates = biencoder.retrieve(q["text"], top_k=retrieve_k)
        results = reranker.rerank(q["text"], candidates, top_k=top_k)
        ndcg_scores.append(ndcg_at_k(results, relevant, k=top_k))
        mrr_scores.append(mrr_at_k(results, relevant, k=top_k))
        recall_scores.append(recall_at_k(candidates, relevant, k=retrieve_k))
    return {f"NDCG@{top_k}": round(np.mean(ndcg_scores), 4), f"MRR@{top_k}": round(np.mean(mrr_scores), 4), f"Recall@{retrieve_k}": round(np.mean(recall_scores), 4)}
