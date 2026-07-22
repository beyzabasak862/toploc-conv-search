"""
Lightweight Tier-2 encoder-match check that does NOT load the full index.
 
Idea: for a few judged queries, pull ONLY their qrel-relevant passage
vectors plus a sample of random distractor passages by streaming the
passage-embedding parquet files, then check that relevant passages
outrank the distractors. If query & passage encoders are aligned,
relevant docs land at the top; if pooling/prefix are wrong, they don't.
 
Reads far less than the ~150 GB index load and keeps RAM tiny.
"""
 
import glob
import numpy as np
import pyarrow.parquet as pq
import ir_datasets
from collections import defaultdict
 
# ---- set these ----
QUERY_EMB_PATH = "cast2020_query_embeddings.parquet"
# the parquet files holding the SNOWFLAKE v2.0 PASSAGE embeddings for 2020:
PASSAGE_GLOB   = "/home/toploc1/Datasets/conversational/CAST2019/snowflake_embeddings/**/*.parquet"
IR_DATASET     = "trec-cast/v1/2020/judged"
 
REL_THRESHOLD  = 1
N_QUERIES      = 5        # how many judged queries to probe
MIN_RELEVANT   = 150      # stop scanning once we have this many relevant vecs
N_DISTRACTORS  = 20000    # random passages as a background pool
TOP_K          = 10
PASSAGE_ID_COL = "id"
PASSAGE_EMB_COL = "embedding"
 
# -------------------------
# queries + qrels
# -------------------------
qt  = pq.read_table(QUERY_EMB_PATH)
qids = [str(x) for x in qt["id"].to_pylist()]
qemb = np.array(qt["embedding"].to_pylist(), dtype=np.float32)
qmap = {qid: qemb[i] for i, qid in enumerate(qids)}
 
qrels = defaultdict(dict)
for q in ir_datasets.load(IR_DATASET).qrels_iter():
    qrels[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
 
# pick judged queries that actually have relevant docs
probe = []
for qid in qids:
    rel = {d for d, g in qrels.get(qid, {}).items() if g >= REL_THRESHOLD}
    if rel:
        probe.append((qid, rel))
    if len(probe) >= N_QUERIES:
        break
if not probe:
    raise SystemExit("No judged queries with relevant docs found.")
 
wanted_rel = set().union(*[rel for _, rel in probe])
print(f"Probing {len(probe)} queries; {len(wanted_rel)} distinct relevant doc ids to find")
 
# -------------------------
# stream passages: collect relevant vecs + distractor sample
# -------------------------
files = sorted(glob.glob(PASSAGE_GLOB, recursive=True))
if not files:
    raise SystemExit(f"No passage parquet files at {PASSAGE_GLOB} -- fix PASSAGE_GLOB.")
 
rel_vecs, rel_ids = {}, []
dis_vecs = []
found_rel = 0
 
for fi, f in enumerate(files):
    tbl = pq.read_table(f, columns=[PASSAGE_ID_COL, PASSAGE_EMB_COL])
    ids = [str(x) for x in tbl[PASSAGE_ID_COL].to_pylist()]
    emb = np.array(tbl[PASSAGE_EMB_COL].to_pylist(), dtype=np.float32)
 
    for j, did in enumerate(ids):
        if did in wanted_rel and did not in rel_vecs:
            rel_vecs[did] = emb[j]
            found_rel += 1
        elif len(dis_vecs) < N_DISTRACTORS and (j % 97 == 0):
            dis_vecs.append(emb[j])
 
    print(f"  [{fi+1}/{len(files)}] relevant found={found_rel}/{len(wanted_rel)} "
          f"distractors={len(dis_vecs)}")
    if found_rel >= MIN_RELEVANT and len(dis_vecs) >= N_DISTRACTORS:
        break
 
if not rel_vecs:
    raise SystemExit("Found ZERO relevant passages -- doc-id format mismatch "
                     "between qrels and passage parquet? Check the ids.")
 
# -------------------------
# build candidate pool: relevant + distractors
# -------------------------
rel_id_list = list(rel_vecs.keys())
R = np.array([rel_vecs[d] for d in rel_id_list], dtype=np.float32)
Dst = np.array(dis_vecs, dtype=np.float32) if dis_vecs else np.empty((0, R.shape[1]), np.float32)
 
pool_vecs = np.vstack([R, Dst])
pool_ids  = rel_id_list + [f"__distractor_{i}" for i in range(len(Dst))]
# cosine: normalize both sides
pool_vecs /= np.maximum(np.linalg.norm(pool_vecs, axis=1, keepdims=True), 1e-12)
 
print(f"\nCandidate pool: {len(rel_id_list)} relevant + {len(Dst)} distractors "
      f"= {len(pool_ids)}")
print("=" * 55)
 
hit = 0
for qid, rel in probe:
    v = qmap[qid].astype(np.float32)
    v = v / max(np.linalg.norm(v), 1e-12)
    scores = pool_vecs @ v
    order = np.argsort(-scores)[:TOP_K]
    topk_ids = [pool_ids[i] for i in order]
    n_rel = sum(1 for d in topk_ids if d in rel)
    hit += (n_rel > 0)
    print(f"  {qid:>8}  relevant-in-top{TOP_K}: {n_rel}/{TOP_K}"
          f"   (pool has {sum(1 for d in rel_id_list if d in rel)} of its relevant docs)")
 
print("=" * 55)
print(f"queries with >=1 relevant hit: {hit}/{len(probe)}")
if hit >= max(1, len(probe) // 2):
    print("PASS -- relevant passages outrank random ones. Encoders aligned.")
else:
    print("FAIL -- relevant passages do NOT outrank random. "
          "Suspect encoder mismatch (pooling/prefix/model) or doc-id mismatch.")