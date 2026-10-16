import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
k = 10
OVERSAMPLING = 2.0        # [Task]
WARMUP_QUERIES = 10       # [Mine] throwaway queries before timing, as in Task 2

# [Task] Qdrant client. [Mine] timeout=60.
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


# [From Task 1] Prints the three averages.
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


# [Task] Add int8 scalar quantization to the collection.
def apply_scalar_quantization():
    client.update_collection(
        collection_name=COLLECTION_NAME,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            ),
        ),
    )


# [Mine] Qdrant builds the quantized vectors in the background.
# Check the collection's status every 10 seconds until it's GREEN.
def wait_until_green():
    time.sleep(5)   # give Qdrant a moment to start the background work
    while True:
        status = client.get_collection(COLLECTION_NAME).status
        print(f"Collection status: {status}")
        if status == models.CollectionStatus.GREEN:
            return
        time.sleep(10)


# [Task, filled in] Task 1's evaluation, changed to use quantization settings.
def evaluate(test_dataset, k, rescore):
    # Exact baseline: skip the HNSW graph AND the quantized vectors.
    exact_params = models.SearchParams(
        exact=True,
        quantization=models.QuantizationSearchParams(ignore=True),
    )
    # ANN: use the quantized vectors, get k x OVERSAMPLING candidates, rescore or not.
    ann_params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=OVERSAMPLING,
        )
    )

    # [Mine] Warm-up: throwaway ANN queries, not timed or scored.
    for embedding in list(test_dataset.values())[:WARMUP_QUERIES]:
        client.query_points(collection_name=COLLECTION_NAME, query=embedding,
                            limit=k, search_params=ann_params)

    precisions = []
    ann_times = []
    knn_times = []

    for query_text, embedding in test_dataset.items():
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=ann_params,
        ).points
        ann_times.append(time.time() - start_time_ann)

        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=exact_params,
        ).points
        knn_times.append(time.time() - start_time_knn)

        ann_ids = set(point.id for point in ann_result)
        knn_ids = set(point.id for point in knn_result)
        precisions.append(len(ann_ids.intersection(knn_ids)) / k)

    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)
    return avg_precision, avg_ann_time, avg_knn_time


# --- Run ---
apply_scalar_quantization()
wait_until_green()
print("Quantization config:", client.get_collection(COLLECTION_NAME).config.quantization_config)

with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

# [Task] Run the evaluation twice: with rescoring, then without.
for rescore in [True, False]:
    print(f"\nrescore={rescore}")
    avg_precision, avg_ann_time, avg_knn_time = evaluate(test_dataset, k, rescore)
    result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)