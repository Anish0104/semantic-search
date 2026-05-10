from datasets import load_dataset
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_msmarco(num_passages=10000, num_queries=500):
    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / f"msmarco_{num_passages}p_{num_queries}q.json"
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
        return data["passages"], data["queries"]
    print("Downloading MS MARCO (this runs once)...")
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
    passages = [{"id": str(row["_id"]), "text": row["text"]} for row in list(corpus)[:num_passages]]
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries", trust_remote_code=True)
    qrels = load_dataset("BeIR/msmarco-qrels", split="validation", trust_remote_code=True)
    passage_ids = {p["id"] for p in passages}
    qrel_map = {}
    for row in qrels:
        qid, pid = str(row["query-id"]), str(row["corpus-id"])
        if pid in passage_ids:
            qrel_map.setdefault(qid, []).append(pid)
    queries = []
    for row in queries_ds:
        qid = str(row["_id"])
        if qid in qrel_map:
            queries.append({"id": qid, "text": row["text"], "relevant_passage_ids": qrel_map[qid]})
        if len(queries) >= num_passages:
            break
    print(f"Loaded {len(passages)} passages, {len(queries)} queries")
    data = {"passages": passages, "queries": queries}
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return passages, queries
