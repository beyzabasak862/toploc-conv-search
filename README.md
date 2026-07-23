# Efficient Conversational Search via Topical Locality in Dense Retrieval — A Replication Study

This repository contains our replication of:

> Muntean, C. I., Nardini, F. M., Perego, R., Rocchietti, G., & Rulli, C. (2025).
> *Efficient Conversational Search via Topical Locality in Dense Retrieval.*
> Proceedings of the 48th International ACM SIGIR Conference on Research and
> Development in Information Retrieval (SIGIR '25), 2749–2753.

The paper introduces **TopLoc**, a method that exploits the topical locality of
conversational queries to accelerate Approximate Nearest Neighbor (ANN) search
in dense retrieval. It proposes two integrations: **TopLoc IVF** (with an
optional refresh variant, **TopLoc IVF+**) built on FAISS's Inverted File
index, and **TopLoc HNSW** built on the Hierarchical Navigable Small World
graph. Our goal is to reproduce the paper's efficiency and effectiveness
claims on the TREC CAsT 2019 dataset, using the Snowflake and Dragon
embedding models specified in the original work.

This document summarizes the methodology, hardware setup, repository
structure, and known limitations of our replication.

---

## 1. Environment and Hardware

All experiments were run on **Pegasus**, accessed via a jump server, with a
secondary machine (**big-dama-3**) used for parallel/overflow workloads
(e.g. index construction while search experiments were running elsewhere).

| | Paper's server | Pegasus (primary) |
|---|---|---|
| CPU | 4× Intel Xeon Gold 6252N | 1× AMD EPYC 7702P |
| Cores / threads | 96 physical / 192 logical | 64 physical / 128 logical (SMT) |
| NUMA nodes | 4 | **1** |
| Isolation method | `numactl --cpunodebind` (single socket, local memory) | not directly applicable — see below |

**Important divergence from the paper's protocol.** The paper explicitly runs
retrieval under `numactl`, confining execution to a single CPU socket and its
local memory, to avoid cross-socket memory-latency artifacts during search.
Pegasus has only **one NUMA node**, so this specific isolation technique does
not apply here — there is no second socket to isolate against. During the
replication we found that FAISS's two internal search code paths (the fused
`index.search()` call and the split `quantizer.search()` +
`index.search_preassigned()` call used by TopLoc) respond very differently to
thread count and scheduling. We ran a dedicated micro-benchmark
(`diag_preassigned_overhead.py`) sweeping FAISS's `parallel_mode` (0–3) for
both code paths and adopted the empirically fastest setting for each
(`parallel_mode=0` for fused search, `parallel_mode=2` for the preassigned
path) throughout all experiments. We further decomposed every reported
speedup into a **mechanism** component (the caching idea itself, isolated via
a preassigned-path control arm that performs the full coarse scan without any
cache) and an **implementation** component (the raw code-path/threading
effect), since naively reporting only the end-to-end speedup would conflate
the two. This decomposition is present throughout our IVF and IVF+ results.

**Practical note for reproducing a run:** always launch long-running search
scripts with `python3 -u` (unbuffered stdout) under `nohup`, and check
`uptime` / `htop` beforehand — several of our early runs were contaminated by
unrelated co-tenant load on the shared server, which is visible as physically
inconsistent latencies (e.g. a smaller `k` reporting higher latency than a
larger `k` for the same configuration).

---

## 2. Repository Structure

```
IVF/
├── dragon/                       # index-building outputs for Dragon (see §4)
├── grid_search_results_snowflake/  # TopLoc IVF results, Snowflake embeddings
├── grid_search_results_dragon/     # TopLoc IVF results, Dragon embeddings
└── grid_search_scripts/            # the grid-search scripts described below

IVF+/
├── grid_search_results_snowflake/  # TopLoc IVF+ (refresh variant) results
└── grid_search_results_dragon/     # same, Dragon embeddings

indexes/Dragon_indexes/          # Dragon IVF index build script + artifacts
```

---

## 3. Plain IVF and TopLoc IVF

**Index.** FAISS `IndexIVFFlat`, inner-product metric, `nlist = 32,768` for
Snowflake (matching the paper's reported best configuration for that
encoder). See §4 for the Dragon index, which uses a different `nlist`.

**Method being replicated.** TopLoc IVF caches the top-*h* coarse centroids
identified from a conversation's first utterance and restricts the coarse
step of every follow-up utterance to that cached set, instead of scanning all
`nlist` centroids. We measure this against a plain-IVF baseline and against a
**control arm** that performs the full, uncached coarse scan through the same
code path TopLoc uses (`search_preassigned`) — this isolates how much of the
speedup is the caching mechanism itself versus the underlying search
implementation (see §1).

**Grid.** `h ∈ {512, 1024, 4096, 8192}`, `nprobe ∈ {1, 2, 4, 8, 16, 32, 64,
128, 256}`, latency swept over `k ∈ {10, 100, 1000}`; effectiveness
(MRR@10, NDCG@3, NDCG@10) computed once per cell at `k = 1000`. Cache-build
cost is timed once per (h, conversation), outside every latency timer, and
reported separately (charged to first-turn total and to a break-even/
conversation-level analysis) rather than folded into follow-up latency.

**Running trick.** The grid-search script reads its index/query/qrels/output
paths from environment variables, so no code edits are needed to point it at
a different index or output directory — only the launch command changes,
e.g.:

```bash
OMP_NUM_THREADS=128 \
OUTPUT_DIR=~/Datasets/toploc1/IVF/grid_search_results_snowflake \
nohup python3 -u search_toploc_ivf_grid_v11.py > .../stdout.log 2>&1 &
```

`OMP_NUM_THREADS` is set **explicitly** rather than left at its default: on
Pegasus's single-socket, 128-logical-core layout, unconstrained threading
interacts with the fused-vs-preassigned code-path difference described in
§1, and controlling it explicitly is what makes the mechanism/implementation
decomposition meaningful and reproducible across runs.

---

## 4. TopLoc IVF+ (Centroid Cache Refresh)

**Method.** TopLoc IVF+ adds a topic-shift detector: at each follow-up turn,
the overlap `|I0|` between the current cache selection and the reference
selection is computed; if it falls below `α · nprobe`, the cache is
refreshed (a full coarse rescan is performed for that turn and the cache is
rebuilt around it).

**Grid.** Same `(h, nprobe, k)` grid as §3, plus `α ∈ {0.0, 0.05, 0.1, 0.2}`.
`α = 0.0` never triggers a refresh and therefore reproduces static TopLoc
IVF's recall exactly — this identity is used as a built-in consistency check
on every run.

**Methodological notes specific to IVF+:**
- Processing is **sequential per turn** within a conversation (a refresh at
  turn *j* changes the cache seen by turn *j+1*), so IVF+ latencies are
  comparable *across α values* but not directly against §3's batched TopLoc
  numbers.
- Refresh cost is **recurring**, not one-time: it is charged inside
  follow-up latency (unlike the initial turn-0 cache build), since it is
  real work paid by a real follow-up query whenever the topic-shift
  condition fires.
- Refresh rate, mean `|I0|` fraction, and per-conversation refresh counts
  are logged alongside effectiveness, to support an analysis of *when* and
  *how often* the mechanism intervenes.

---

## 5. Dragon Index Construction

The Dragon IVF index was built with the script located in
`indexes/Dragon_indexes/`, using `IndexIVFFlat` with **inner-product**
metric on the raw (non-normalized) Dragon embeddings — Dragon is a
dot-product model whose embedding norms carry information, so, unlike
Snowflake, query and document vectors are **not** L2-normalized before
indexing or search.

**Configuration.** `nlist = 2^18 = 262,144`, matching the paper's reported
best configuration for Dragon (as opposed to `2^15 = 32,768` for Snowflake).
Due to the time cost of training this many centroids, two deviations from
FAISS's recommended defaults were necessary:
- Training set size: `20 × nlist` (~5.2M vectors) rather than the
  recommended `39 × nlist` (~10.2M vectors).
- `k-means niter = 15` rather than FAISS's default of 25.

**Known limitation.** These reductions were necessary to make index
construction tractable under our time constraints. A dedicated diagnostic
(comparing exhaustive brute-force search against IVF search at high
`nprobe`, and comparing Dragon's exact/flat-index effectiveness against the
paper's reported Exact reference) confirmed that:
1. The Dragon embeddings themselves are correct — exact (flat, brute-force)
   search over them reaches NDCG@10 ≈ 0.49, in line with the paper's
   reported Exact-search ceiling for Dragon on CAsT 2019.
2. The IVF search mechanics are also correct — high-`nprobe` IVF search
   agrees exactly with low-`nprobe` results for the same query, and
   `nprobe` sweeps behave monotonically as expected.
3. The gap between (1) and our IVF grid's effectiveness is attributable to
   **undertrained k-means centroids**: 262,144 centroids trained on a
   reduced sample and fewer iterations do not partition the embedding space
   finely enough for low-`nprobe` search to reliably reach the correct
   posting lists. This is a training-budget limitation, not a bug in the
   retrieval or caching logic, and it is specific to the Dragon
   configuration — the Snowflake index (fewer centroids, adequately
   trained) does not exhibit this issue.

We report this explicitly rather than omitting the Dragon IVF results, since
the diagnostic process itself (isolating embedding quality from index
quality) is a relevant methodological contribution of this replication.

---

## 6. HNSW and TopLoc HNSW

*[To be completed — this section covers the HNSW baseline and TopLoc HNSW
replication.]*

### 6.1 HNSW baseline

*(index construction, parameters, effectiveness/latency results)*

### 6.2 TopLoc HNSW

*(privileged entry point mechanism, grid, results, comparison to plain
HNSW)*

---

## 7. Summary of Limitations

- Our AMD EPYC / single-NUMA-node hardware differs architecturally from the
  paper's 4-socket Intel Xeon setup; absolute latencies are not directly
  comparable across the two, though the qualitative speedup mechanism is
  reproducible (see §1).
- Effectiveness metrics show a small, consistent gap versus the paper even
  on our unmodified plain-IVF baseline, plausibly attributable to qrels
  coverage and/or k-means initialization randomness rather than any error
  in the TopLoc implementation.
- The Dragon IVF index is undertrained relative to FAISS's recommended
  configuration, for the reasons and with the diagnostic evidence detailed
  in §5.
- [Add any HNSW-specific limitations here.]

---

## 8. Authors

- IVF, TopLoc IVF, TopLoc IVF+, Dragon indexing, hardware/environment
  diagnostics: *[your name]*
- HNSW, TopLoc HNSW: *[teammate's name]*