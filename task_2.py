import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
k = 10
HNSW_EF_VALUES = [10, 20, 50, 100, 200]   # [Task]
WARMUP_QUERIES = 10                        # [Mine] how many throwaway queries to run first
RUNS_PER_EF = 5   # [Mine] times to repeat all 100 queries for each ef value; the first run is discarded

# [Task] Qdrant client. [Mine] timeout=60.
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


# [Task, filled in] Test each hnsw_ef value and return one result dict per value.
def evaluate_hnsw_ef(k, hnsw_ef_values, test_dataset):

    # --- 1. Ground truth: exact search once per query, store the IDs ---
    ground_truth = {}
    for query_text, embedding in test_dataset.items():
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        ground_truth[query_text] = set(point.id for point in knn_result)

    # --- 2. Warm-up: throwaway ANN queries, not timed or scored ---
    for embedding in list(test_dataset.values())[:WARMUP_QUERIES]:
        client.query_points(collection_name=COLLECTION_NAME, query=embedding, limit=k)

    # --- 3. Sweep: for each ef value, repeat all queries RUNS_PER_EF times.
    #        Discard the first run, average the rest. ---
    results = []
    for hnsw_ef in hnsw_ef_values:
        precisions = []
        query_times = []

        for run in range(RUNS_PER_EF):
            for query_text, embedding in test_dataset.items():
                start_time = time.time()
                ann_result = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=embedding,
                    limit=k,
                    search_params=models.SearchParams(hnsw_ef=hnsw_ef),
                ).points
                elapsed = time.time() - start_time

                # Skip the first run: it's a warm-up for this ef value.
                if run == 0:
                    continue

                query_times.append(elapsed)
                ann_ids = set(point.id for point in ann_result)
                precisions.append(len(ann_ids.intersection(ground_truth[query_text])) / k)

        results.append({
            "hnsw_ef": hnsw_ef,
            "avg_precision": sum(precisions) / len(precisions),
            "avg_query_time_ms": sum(query_times) / len(query_times) * 1000,
        })

    return results


# [Task] Load the test dataset: {query text: embedding}.
with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

# [Mine] Run the sweep and print one line per hnsw_ef value.
results = evaluate_hnsw_ef(k, HNSW_EF_VALUES, test_dataset)
for result in results:
    print(
        f"hnsw_ef={result['hnsw_ef']:>3}: "
        f"avg precision@{k} = {result['avg_precision']:.4f}, "
        f"avg query time = {result['avg_query_time_ms']:.2f} ms"
    )