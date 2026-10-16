import json
import time

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
QUERIES_FILE = "dataset/queries_embeddings.json"
k = 10

# [Task] Qdrant client. [Mine] timeout=60, as before.
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


# [Task] Prints the three averages.
def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


# [Task, filled in] For each test embedding: run both searches, time each one,
# and compute precision. Return the three averages.
def evaluate(test_dataset, k):
    precisions = []
    ann_times = []
    knn_times = []

    for query_text, embedding in test_dataset.items():

        # Approximate search (Qdrant's default, using the HNSW index), timed.
        start_time_ann = time.time()
        ann_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
        ).points
        ann_times.append(time.time() - start_time_ann)

        # Exact search (compares with every vector), timed.
        start_time_knn = time.time()
        knn_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        knn_times.append(time.time() - start_time_knn)

        # Precision: how many of ANN's IDs are also in the exact results, divided by k.
        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precisions.append(len(ann_ids.intersection(knn_ids)) / k)

    # [Mine] Average of each list: its sum divided by how many items it has.
    avg_precision = sum(precisions) / len(precisions)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)
    return avg_precision, avg_ann_time, avg_knn_time


# [Task] Load the test dataset: a dict of {query text: embedding}.
with open(QUERIES_FILE, "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

# [Mine] Run the evaluation and print the results.
avg_precision, avg_ann_time, avg_knn_time = evaluate(test_dataset, k)
result_formatting(k, avg_precision, avg_ann_time, avg_knn_time)