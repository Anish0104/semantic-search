import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_msmarco
from src.bm25_retriever import BM25Retriever
from src.biencoder_retriever import BiEncoderRetriever
from src.reranker import CrossEncoderReranker
from eval.metrics import evaluate_retriever, evaluate_reranker

def print_table(results):
    print("\n" + "=" * 65)
    print(f"{'System':<30} {'NDCG@10':<12} {'MRR@10':<12} {'Recall@100'}")
    print("-" * 65)
    for system, metrics in results.items():
        vals = list(metrics.values())
        print(f"{system:<30} {vals[0]:<12} {vals[1]:<12} {vals[2]}")
    print("=" * 65)

def main():
    passages, queries = load_msmarco(num_passages=100000, num_queries=200)
    print(f"\nEvaluating on {len(queries)} queries, {len(passages)} passages\n")
    results = {}

    print(">>> Running BM25 baseline...")
    bm25 = BM25Retriever()
    bm25.index(passages)
    t0 = time.time()
    results["BM25 (baseline)"] = evaluate_retriever(bm25, queries, top_k=10, retrieve_k=100)
    print(f"    Done in {time.time()-t0:.1f}s | {results['BM25 (baseline)']}")

    print("\n>>> Running Bi-encoder...")
    biencoder = BiEncoderRetriever(qdrant_path=":memory:")
    biencoder.index(passages)
    t0 = time.time()
    results["Bi-encoder (MiniLM-L6)"] = evaluate_retriever(biencoder, queries, top_k=10, retrieve_k=100)
    print(f"    Done in {time.time()-t0:.1f}s | {results['Bi-encoder (MiniLM-L6)']}")

    print("\n>>> Running Bi-encoder + Reranker...")
    reranker = CrossEncoderReranker()
    t0 = time.time()
    results["Bi-encoder + Reranker"] = evaluate_reranker(biencoder, reranker, queries, retrieve_k=100, top_k=10)
    print(f"    Done in {time.time()-t0:.1f}s | {results['Bi-encoder + Reranker']}")

    print_table(results)

    bm25_ndcg = results["BM25 (baseline)"]["NDCG@10"]
    bi_ndcg = results["Bi-encoder (MiniLM-L6)"]["NDCG@10"]
    rr_ndcg = results["Bi-encoder + Reranker"]["NDCG@10"]
    print(f"\nBi-encoder vs BM25:     +{((bi_ndcg-bm25_ndcg)/bm25_ndcg)*100:.1f}% NDCG@10")
    print(f"Reranker vs Bi-encoder: +{((rr_ndcg-bi_ndcg)/bi_ndcg)*100:.1f}% NDCG@10")
    print(f"Full pipeline vs BM25:  +{((rr_ndcg-bm25_ndcg)/bm25_ndcg)*100:.1f}% NDCG@10")
    print("\n^^^ These are your resume numbers.")

if __name__ == "__main__":
    main()
