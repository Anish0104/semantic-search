from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class CrossEncoderReranker:
    def __init__(self):
        device = get_device()
        print(f"Loading cross-encoder on {device}")
        self.model = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [{"id": cand["id"], "text": cand["text"], "score": float(score), "rank": rank + 1, "biencoder_rank": cand["rank"]} for rank, (cand, score) in enumerate(reranked[:top_k])]
