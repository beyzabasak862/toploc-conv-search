import faiss
import numpy as np
import pyarrow.parquet as pq
import os
import sys
import time
import logging

# ======================================================================
# DIAGNOSTIC: is the nprobe>=8 TopLoc collapse caused by the
# search_preassigned code path itself, rather than by TopLoc's logic?
#
# Method: for each nprobe, compute the coarse assignments with a full
# quantizer scan (identical to what fused search uses internally), then
# time (a) fused index.search() and (b) index.search_preassigned() fed
# those exact assignments. Same queries, same centroids, same k, same
# posting lists scanned. Any latency gap = pure API-path overhead,
# independent of TopLoc.
#
# Also sweeps index.parallel_mode (0..3), which controls how OpenMP
# threads are distributed (over queries vs probes) and is known to
# affect the preassigned path differently from the fused path.
#
# Interpretation:
#   preassigned ~= fused  -> the TopLoc selection loop is the bottleneck
#                            (fixable in Python by batching)
#   preassigned >> fused  -> the Python-exposed preassigned path is the
#                            bottleneck; matching the paper's high-nprobe
#                            numbers requires their low-level C++ route
#                            (as the paper itself states it used)
# ======================================================================

INDEX_PATH     = os.environ.get("INDEX_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf.index")
QUERY_EMB_PATH = os.environ.get("QUERY_EMB_PATH",
    "/home/toploc1/Datasets/conversational/CAST2019/topics/topics_snowflake_embeddings.parquet")
USE_MMAP       = os.environ.get("MMAP", "1") == "1"

NPROBE_VALUES  = [1, 8, 64, 256]
PARALLEL_MODES = [0, 1, 2, 3]
K              = int(os.environ.get("K", 10))
WARMUP_RUNS    = 1
TIMED_RUNS     = 3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def timeit(fn):
    for _ in range(WARMUP_RUNS):
        fn()
    ts = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


log.info(f"Loading index (mmap={USE_MMAP})")
index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP if USE_MMAP else 0)
quantizer = faiss.downcast_index(index.quantizer)
log.info(f"Index: {index.ntotal:,} vectors, nlist={index.nlist}, "
         f"omp_threads={faiss.omp_get_max_threads()}")

q_table = pq.read_table(QUERY_EMB_PATH)
q_emb = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
faiss.normalize_L2(q_emb)
n = q_emb.shape[0]
log.info(f"Queries: {n} | dim={q_emb.shape[1]} | k={K}")

default_pm = index.parallel_mode
log.info(f"Default parallel_mode = {default_pm}")

for nprobe in NPROBE_VALUES:
    index.nprobe = nprobe

    # Coarse assignments identical to fused search's internal step.
    # NOT timed — we only compare the scan phase.
    Dq, Iq = quantizer.search(q_emb, nprobe)
    Iq = Iq.astype(np.int64)
    Dq = Dq.astype(np.float32)

    log.info(f"--- nprobe={nprobe} ---")
    for pm in PARALLEL_MODES:
        index.parallel_mode = pm
        fused_ms = timeit(lambda: index.search(q_emb, K)) / n
        pre_ms   = timeit(
            lambda: index.search_preassigned(q_emb, K, Iq, Dq)) / n
        ratio = pre_ms / fused_ms if fused_ms > 0 else float("nan")
        log.info(f"  parallel_mode={pm} | fused={fused_ms:7.3f} ms/q | "
                 f"preassigned={pre_ms:7.3f} ms/q | "
                 f"preassigned/fused = {ratio:5.2f}x")

index.parallel_mode = default_pm
log.info("Done. If preassigned/fused >> 1 at high nprobe across all "
         "parallel modes, the collapse is the API path, not TopLoc.")