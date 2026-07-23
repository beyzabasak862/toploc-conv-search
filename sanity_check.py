import faiss
import numpy as np
import pyarrow.parquet as pq
import os

# ======================================================================
# SANITY CHECK for the Dragon IVF index.
#
# Goal: figure out whether the near-zero NDCG@10 (~0.014) comes from
#   (a) a genuinely broken/mismatched index (wrong metric, wrong vectors,
#       query/doc space mismatch), or
#   (b) a technically-correct but very low-quality index (bad centroids
#       from undertrained k-means), which just needs much higher nprobe
#       or better training to recover reasonable recall.
#
# Method: take a handful of query vectors, run them through the index at
# a VERY high nprobe (nearly exhaustive), and separately run a brute-force
# exact search over the SAME set of document vectors read directly from
# the index (via reconstruct). Compare:
#   1. Does high-nprobe IVF search agree with exact brute-force search?
#      If yes -> the index's search mechanics are fine, centroids are
#      just poor at low nprobe (case b). If no -> something structural
#      is broken (case a): wrong metric, wrong id mapping, etc.
#   2. Sanity-check the query/doc vector scale: print norms. If document
#      norms are wildly different from query norms (e.g. doc norms ~50,
#      query norms ~1), that alone can break inner-product ranking if
#      one side was accidentally normalized and the other wasn't -- this
#      is a scale-mismatch check independent of the grep result.
# ======================================================================

INDEX_PATH = os.environ.get("INDEX_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_dragon_ivf_2e18.index")
IDS_PATH = os.environ.get("IDS_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_dragon_ivf_2e18_ids.npy")
QUERY_EMB_PATH = os.environ.get("QUERY_EMB_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/topics_dragon_embeddings.parquet")

N_SAMPLE_QUERIES = 5
EXHAUSTIVE_NPROBE = 4096   # ~1.5% of nlist=262144, should be close to exact
TOP_K = 10

print(f"Loading index from {INDEX_PATH} (mmap)")
index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
print(f"Index: {index.ntotal:,} vectors, nlist={index.nlist}, "
      f"metric={'INNER_PRODUCT' if index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}")

all_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
assert len(all_ids) == index.ntotal, \
    f"MISMATCH: {len(all_ids)} ids vs {index.ntotal} vectors -- id map is out of sync with the index!"
print(f"ID map: {len(all_ids):,} entries -- matches index.ntotal OK")

q_table = pq.read_table(QUERY_EMB_PATH)
q_ids = q_table["id"].to_pylist()
q_emb_raw = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
print(f"Queries loaded: {len(q_ids)} | dim={q_emb_raw.shape[1]}")

# --- Scale check: are query and (sampled) document vectors on comparable
#     norms? A big mismatch breaks inner-product ranking. ---
sample_doc_ids = np.random.default_rng(0).choice(index.ntotal, size=200, replace=False)
index.make_direct_map()
sample_doc_vecs = np.vstack([index.reconstruct(int(i)) for i in sample_doc_ids])
q_norms = np.linalg.norm(q_emb_raw[:50], axis=1)
d_norms = np.linalg.norm(sample_doc_vecs, axis=1)
print(f"\n--- Vector scale check ---")
print(f"Query norms   (n=50):  mean={q_norms.mean():.3f}  std={q_norms.std():.3f}  "
      f"min={q_norms.min():.3f}  max={q_norms.max():.3f}")
print(f"Document norms(n=200): mean={d_norms.mean():.3f}  std={d_norms.std():.3f}  "
      f"min={d_norms.min():.3f}  max={d_norms.max():.3f}")
if q_norms.mean() < 0.1 or d_norms.mean() < 0.1:
    print("WARNING: near-zero norms detected -- possible all-zero vectors "
          "(e.g. wrong column read, encoding failure).")
ratio = d_norms.mean() / max(q_norms.mean(), 1e-9)
print(f"doc_norm_mean / query_norm_mean = {ratio:.2f}")
if ratio > 3 or ratio < 0.33:
    print("WARNING: query and document vector scales differ substantially. "
          "For an inner-product index this can dominate the ranking "
          "regardless of direction/similarity, and is consistent with "
          "one side having been normalized and the other not.")
else:
    print("Scales look comparable -- unlikely to be a normalize mismatch.")

# --- Exact vs high-nprobe IVF agreement check ---
print(f"\n--- Exact vs IVF@nprobe={EXHAUSTIVE_NPROBE} agreement (first "
      f"{N_SAMPLE_QUERIES} queries) ---")

# Brute-force exact search over the SAME sampled 200 doc vectors (small,
# fast, and guarantees we're comparing against vectors actually read from
# this index -- not a separate/mismatched source).
quantizer = faiss.downcast_index(index.quantizer)

for qi in range(N_SAMPLE_QUERIES):
    q = q_emb_raw[qi:qi + 1]

    # exact (brute-force) ranking over the 200-doc sample, by inner product
    sims = sample_doc_vecs @ q[0]
    exact_top = sample_doc_ids[np.argsort(-sims)[:TOP_K]]

    # IVF search at near-exhaustive nprobe, restricted post-hoc to the
    # same 200-doc sample for a fair apples-to-apples comparison
    index.nprobe = EXHAUSTIVE_NPROBE
    D, I = index.search(q, 2000)  # generous depth, then filter to sample
    sample_set = set(int(x) for x in sample_doc_ids)
    ivf_top_in_sample = [int(idx) for idx in I[0] if int(idx) in sample_set][:TOP_K]

    overlap = len(set(exact_top.tolist()) & set(ivf_top_in_sample))
    print(f"  query {qi} (id={q_ids[qi]}): exact_top10 vs ivf_top10 overlap "
          f"in 200-doc sample = {overlap}/10")

    # Also: what does the index return at LOW nprobe (what the grid used)?
    index.nprobe = 8
    D8, I8 = index.search(q, TOP_K)
    print(f"    nprobe=8  top-1 doc id: {all_ids[I8[0][0]] if I8[0][0] != -1 else 'NONE'}, "
          f"score={D8[0][0]:.3f}")
    index.nprobe = EXHAUSTIVE_NPROBE
    D_hi, I_hi = index.search(q, TOP_K)
    print(f"    nprobe={EXHAUSTIVE_NPROBE} top-1 doc id: "
          f"{all_ids[I_hi[0][0]] if I_hi[0][0] != -1 else 'NONE'}, score={D_hi[0][0]:.3f}")

print("\n--- Full scan test (nprobe = nlist) ---")
index.nprobe = index.nlist   # exhaustive: tüm 262144 merkez taranır
D, I = index.search(q_emb_raw[:5], 10)   # sadece ilk 5 sorgu, hızlı olsun

for qi in range(5):
    print(f"query {qi}: top-1 doc_id={all_ids[I[qi][0]]} score={D[qi][0]:.3f}")

print("\n--- nprobe sweep test ---")
q = q_emb_raw[0:1]  # query 0, üstteki testle aynı sorgu

for np_test in [8, 512, 2048, 8192, 32768, 65536, 131072, 262144]:
    index.nprobe = np_test
    actual = index.nprobe   # gerçekten set edildi mi kontrol
    D, I = index.search(q, 10)
    print(f"nprobe requested={np_test:>7} actual={actual:>7} | "
          f"top-1: {all_ids[I[0][0]]} score={D[0][0]:.3f}")

print("\nDone. Read the WARNING lines above first -- they point at the "
      "specific mechanism if something is structurally broken. If no "
      "warnings and overlap counts are high (>=5/10), the index mechanics "
      "are fine and the low NDCG is a training-quality issue (undertrained "
      "k-means), not a bug -- meaning you'd need much higher nprobe or a "
      "retrained index with more training vectors/iterations to see "
      "reasonable recall.")