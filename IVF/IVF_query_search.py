import faiss
import numpy as np
import pyarrow.parquet as pq

# --------------------------------------------------
# 1. LOAD FAISS INDEX + IDS
# --------------------------------------------------

index = faiss.read_index("toploc_ivf_34m.index")
ids = np.load("toploc_ids.npy", allow_pickle=True)

index.nprobe = 20  # tune: 10–50

print("Index loaded:", index.ntotal)

# --------------------------------------------------
# 2. LOAD QUERY EMBEDDINGS FROM PARQUET
# --------------------------------------------------

query_file = "query_embeddings.parquet"

table = pq.read_table(query_file)

# assume column name: "embedding"
query_list = table["embedding"].to_pylist()

query_embeddings = np.array(query_list, dtype=np.float32)

print("Query shape:", query_embeddings.shape)

# --------------------------------------------------
# 3. NORMALIZE (if cosine similarity)
# --------------------------------------------------

faiss.normalize_L2(query_embeddings)

# --------------------------------------------------
# 4. FAISS SEARCH
# --------------------------------------------------

TOP_K = 1000

D, I = index.search(query_embeddings, TOP_K)

print("Search completed")

# --------------------------------------------------
# 5. MAP RESULTS (indices → IDs)
# --------------------------------------------------

results = []

for q_idx in range(len(I)):

    doc_ids = [ids[i] for i in I[q_idx]]
    scores = D[q_idx]

    results.append({
        "query_id": q_idx,
        "results": list(zip(doc_ids, scores))
    })

print("Mapping completed")

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

np.save(
    "retrieval_results.npy",
    results,
    allow_pickle=True
)

print("Saved results")