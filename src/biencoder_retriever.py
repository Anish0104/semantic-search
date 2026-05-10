from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import torch

COLLECTION_NAME = "msmarco_passages"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_device() -> str:
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon MPS (GPU)")
        return "mps"
    elif torch.cuda.is_available():
        print("Device: CUDA GPU")
        return "cuda"
    print("Device: CPU")
    return "cpu"

def get_batch_size(device: str) -> int:
    return 512 if device in ("mps", "cuda") else 256

class BiEncoderRetriever:
    def __init__(self, qdrant_path: str = ":memory:"):
        self.device = get_device()
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.batch_size = get_batch_size(self.device)
        self.dim = self.model.get_embedding_dimension()
        self.client = QdrantClient(path=qdrant_path) if qdrant_path != ":memory:" else QdrantClient(":memory:")
        self.passages = {}

    def index(self, passages: List[Dict], force_reindex: bool = False):
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in collections and not force_reindex:
            print("Collection already exists, skipping indexing")
            self.passages = {p["id"]: p["text"] for p in passages}
            return
        if COLLECTION_NAME in collections:
            self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
        self.passages = {p["id"]: p["text"] for p in passages}
        texts = [p["text"] for p in passages]
        ids = [p["id"] for p in passages]
        print(f"Encoding {len(passages)} passages on {self.device}...")
        points = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            embeddings = self.model.encode(batch_texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
            for pid, emb in zip(batch_ids, embeddings):
                points.append(PointStruct(id=abs(hash(pid)) % (2**63), vector=emb.tolist(), payload={"passage_id": pid}))
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Indexed {len(points)} passages into Qdrant")

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        ).points
        return [{"id": hit.payload["passage_id"], "text": self.passages.get(hit.payload["passage_id"], ""), "score": float(hit.score), "rank": rank + 1} for rank, hit in enumerate(results)]
