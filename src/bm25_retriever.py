from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.passages = []

    def index(self, passages: List[Dict]):
        self.passages = passages
        tokenized = [p["text"].lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 indexed {len(passages)} passages")

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"id": self.passages[i]["id"], "text": self.passages[i]["text"], "score": float(scores[i]), "rank": rank + 1} for rank, i in enumerate(top_indices)]
