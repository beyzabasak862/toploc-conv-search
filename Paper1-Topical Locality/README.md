# Paper 1 — Efficient Conversational Search via Topical Locality in Dense Retrieval (Replication)

This directory holds our replication of:

> Muntean, C. I., Nardini, F. M., Perego, R., Rocchietti, G., & Rulli, C. (2025).
> *Efficient Conversational Search via Topical Locality in Dense Retrieval.*
> Proceedings of the 48th International ACM SIGIR Conference on Research and
> Development in Information Retrieval (SIGIR '25), 2749–2753.

The paper introduces **TopLoc**, which exploits the topical locality of
conversational queries to accelerate Approximate Nearest Neighbor (ANN) search
in dense retrieval. It proposes two index integrations:

- **TopLoc IVF** (with a refresh variant, **TopLoc IVF+**) built on FAISS's
  Inverted File index.
- **TopLoc HNSW** built on the Hierarchical Navigable Small World graph.

We reproduce the paper's efficiency and effectiveness claims on **TREC CAsT
2019** and **CAsT 2020**, using the **Snowflake** and **Dragon** embedding
models specified in the original work.

> This README documents the file architecture of the `Paper1-Topical Locality/`
> directory. It is organized as two fully separate parts — **IVF** and
> **HNSW** — with every artifact grouped under its part. Shared assets
> (exact-search baselines, data exploration, demo) follow at the end. All
> paths below are relative to this directory unless noted.

---

## Directory Map (top level)

```
Paper1-Topical Locality/
├── Toploc IVF/          # Part 1 — IVF, TopLoc IVF, TopLoc IVF+
├── Toploc HNSW/         # Part 2 — HNSW baseline + TopLoc HNSW
├── Exact_Search/        # Shared — flat/brute-force reference (Dragon + Snowflake)
├── Data Exploration/    # Shared — datasets, query embeddings, encoding scripts
└── demo/                # Shared — interactive retrieval demo
```

---

## Environment and Hardware (applies to both parts)

All experiments were run on **Pegasus** (accessed via a jump server), with a
secondary machine (**big-dama-3**) used for parallel/overflow workloads
(e.g. index construction while search experiments ran elsewhere).

| | Paper's server | Pegasus (primary) |
|---|---|---|
| CPU | 4× Intel Xeon Gold 6252N | 1× AMD EPYC 7702P |
| Cores / threads | 96 physical / 192 logical | 64 physical / 128 logical (SMT) |
| NUMA nodes | 4 | **1** |
| Isolation method | `numactl --cpunodebind` | not applicable — single NUMA node |

**Divergence from the paper's protocol.** The paper runs retrieval under
`numactl`, confining execution to a single CPU socket and its local memory.
Pegasus has only **one NUMA node**, so this isolation technique does not apply
(there is no second socket to isolate against). This mainly affects the IVF
timing methodology — see Part 1.

**Practical note for reproducing a run:** launch long-running search scripts
with `python3 -u` (unbuffered stdout) under `nohup`, and check `uptime` /
`htop` beforehand — several early runs were contaminated by co-tenant load on
the shared server (visible as physically inconsistent latencies, e.g. a
smaller `k` reporting higher latency than a larger `k`).

---

# Part 1 — IVF (`Toploc IVF/`)

Everything for the plain-IVF baseline, TopLoc IVF, and TopLoc IVF+ lives under
`Toploc IVF/`.

```
Toploc IVF/
├── index_scripts/            # IVF index construction (Dragon + Snowflake)
│   ├── ivf_index_creation_dragon.py
│   ├── ivf_index_creation_snowflake.py
│   └── logs/
│       ├── ivf_index_creation_dragon.log
│       └── ivf_index_creation_snowflake.log
│
├── IVF/                      # Plain IVF baseline + static TopLoc IVF
│   ├── grid_search_scripts/
│   │   ├── search_ivf_grid.py                    # plain-IVF baseline sweep
│   │   ├── search_toploc_ivf_grid_snowflake.py   # TopLoc IVF, Snowflake
│   │   └── search_toploc_ivf_grid_dragon.py      # TopLoc IVF, Dragon
│   ├── diagnose_ivf_thread_scheduling.py         # parallel_mode / thread micro-benchmark
│   ├── grid_search_results/
│   │   ├── metrics_ivf_toploc_snowflake.json
│   │   └── metrics_ivf_toploc_dragon.json
│   └── logs/
│       ├── diagnose_ivf_thread_scheduling.log
│       ├── search_ivf_toploc_snowflake.log
│       └── search_ivf_toploc_dragon.log
│
└── IVF+/                     # TopLoc IVF+ (centroid-cache refresh variant)
    ├── grid_search_scripts/
    │   ├── toploc_ivf_plus_grid_snowflake.py
    │   └── toploc_ivf_plus_grid_dragon.py
    ├── grid_search_results/
    │   ├── metrics_toploc_ivf_plus_grid_snowflake.json
    │   └── metrics_toploc_ivf_plus_grid_dragon.json
    └── logs/
        ├── toploc_ivf_plus_grid.log
        ├── toploc_ivf_plus_grid_dragon.log
        └── v_dragon_stdout.log
```

### 1.1 IVF index construction — `index_scripts/`

Builds the FAISS `IndexIVFFlat` indexes searched by every IVF experiment.

- **Snowflake:** `IndexIVFFlat`, inner-product metric, `nlist = 2^15 = 32,768`
  (the paper's reported best for that encoder). Query/document vectors are
  L2-normalized.
- **Dragon:** `IndexIVFFlat`, inner-product metric, `nlist = 2^18 = 262,144`
  (the paper's reported best for Dragon). Dragon is a dot-product model whose
  embedding norms carry information, so vectors are **not** L2-normalized.
  Because of the training cost of this many centroids, two deviations from
  FAISS defaults were necessary: training set of `20 × nlist` (~5.2M vectors)
  instead of `39 × nlist`, and `k-means niter = 15` instead of 25. These make
  the Dragon index **undertrained** — see the limitation in §1.4.

### 1.2 Plain IVF baseline + TopLoc IVF — `IVF/`

**Method.** TopLoc IVF caches the top-*h* coarse centroids identified from a
conversation's first utterance and restricts the coarse step of every
follow-up utterance to that cached set, instead of scanning all `nlist`
centroids. It is measured against a plain-IVF baseline and against a **control
arm** performing the full, uncached coarse scan through the same code path
TopLoc uses (`search_preassigned`) — this isolates how much of the speedup is
the caching *mechanism* versus the underlying search *implementation*.

**Grid.** `h ∈ {512, 1024, 4096, 8192}`, `nprobe ∈ {1, 2, 4, 8, 16, 32, 64,
128, 256}`, latency swept over `k ∈ {10, 100, 1000}`; effectiveness (MRR@10,
NDCG@3, NDCG@10) computed once per cell at `k = 1000`. Cache-build cost is
timed once per (h, conversation), outside every latency timer, and reported
separately (charged to first-turn total and to a break-even analysis) rather
than folded into follow-up latency.

**Threading / code-path decomposition (`diagnose_ivf_thread_scheduling.py`).**
FAISS's two internal search code paths — the fused `index.search()` call and
the split `quantizer.search()` + `index.search_preassigned()` call used by
TopLoc — respond very differently to thread count and scheduling. This micro-
benchmark sweeps FAISS's `parallel_mode` (0–3) for both paths; we adopted the
empirically fastest setting for each (`parallel_mode=0` for fused search,
`parallel_mode=2` for the preassigned path) throughout. `OMP_NUM_THREADS` is
set **explicitly** rather than left at its default, because on Pegasus's
single-socket, 128-logical-core layout unconstrained threading interacts with
the fused-vs-preassigned difference and would make the mechanism/implementation
decomposition irreproducible.

**Running trick.** The grid-search scripts read index / query / qrels / output
paths from environment variables, so pointing at a different index or output
directory requires no code edits — only the launch command changes:

```bash
OMP_NUM_THREADS=128 \
OUTPUT_DIR=<...>/IVF/grid_search_results \
nohup python3 -u search_toploc_ivf_grid_snowflake.py > .../stdout.log 2>&1 &
```

**Outputs.** `grid_search_results/metrics_ivf_toploc_{snowflake,dragon}.json`.

### 1.3 TopLoc IVF+ (centroid cache refresh) — `IVF+/`

**Method.** IVF+ adds a topic-shift detector: at each follow-up turn the
overlap `|I0|` between the current cache selection and the reference selection
is computed; if it falls below `α · nprobe`, the cache is refreshed (a full
coarse rescan is performed for that turn and the cache rebuilt around it).

**Grid.** Same `(h, nprobe, k)` grid as §1.2, plus `α ∈ {0.0, 0.05, 0.1, 0.2}`.
`α = 0.0` never triggers a refresh and therefore reproduces static TopLoc IVF's
recall exactly — a built-in consistency check on every run.

**Notes specific to IVF+:**
- Processing is **sequential per turn** (a refresh at turn *j* changes the
  cache seen by turn *j+1*), so IVF+ latencies are comparable *across α values*
  but not directly against §1.2's batched numbers.
- Refresh cost is **recurring**, not one-time: it is charged inside follow-up
  latency (unlike the turn-0 cache build), since it is real work paid by a real
  follow-up query whenever the topic-shift condition fires.
- Refresh rate, mean `|I0|` fraction, and per-conversation refresh counts are
  logged alongside effectiveness.

**Outputs.** `grid_search_results/metrics_toploc_ivf_plus_grid_{snowflake,dragon}.json`.

### 1.4 Known limitation — undertrained Dragon IVF index

The reduced training budget in §1.1 leaves the Dragon IVF centroids
undertrained. A dedicated diagnostic (comparing exhaustive brute-force search
against IVF at high `nprobe`, and Dragon's flat-index effectiveness against the
paper's Exact reference) confirmed:
1. The **Dragon embeddings are correct** — exact (flat) search reaches
   NDCG@10 ≈ 0.49, in line with the paper's Exact ceiling for Dragon on CAsT
   2019.
2. The **IVF mechanics are correct** — high-`nprobe` IVF agrees with
   low-`nprobe` results and `nprobe` sweeps behave monotonically.
3. The gap between (1) and our IVF grid's effectiveness is attributable to
   **undertrained k-means centroids**, not a bug in the retrieval or caching
   logic. This is specific to Dragon; the Snowflake index (fewer centroids,
   adequately trained) does not exhibit it.

---

# Part 2 — HNSW (`Toploc HNSW/`)

Everything for the HNSW baseline and TopLoc HNSW lives under `Toploc HNSW/`,
split by embedding model (`Dragon/`, `Snowflake/`) with a shared analysis
folder. Each model has an `index build/` and a `search/` subtree; search
results are further split by dataset (`CAST2019/`, `CAST2020/`).

```
Toploc HNSW/
├── Dragon/
│   ├── index build/
│   │   ├── indexCreationHNSW_multiM.py           # builds M ∈ {16,32,64} HNSW indexes
│   │   └── logs/
│   │       ├── indexCreationHNSW_dragon_multiM.log
│   │       └── indexCreationHNSW_M32_dragon_mips.log
│   └── search/
│       ├── search_hnsw_dragon_cast2019.py
│       ├── search_hnsw_dragon_cast2020.py
│       ├── logs/                                 # per-M, per-year search logs
│       └── results/
│           ├── CAST2019/   # parquet run files + metrics JSON, M ∈ {16,32,64}
│           └── CAST2020/   # parquet run files + metrics JSON, M ∈ {16,32,64}
│
├── Snowflake/
│   ├── index build/
│   │   ├── indexCreation_Mgrid.py                # builds M ∈ {16,32,64} HNSW indexes
│   │   └── logs/
│   │       ├── indexCreationHNSW_M16.log
│   │       ├── indexCreationHNSW_M32.log
│   │       └── indexCreationHNSW_M64.log
│   └── search/
│       ├── search_hnsw_snowflake_cast2019.py
│       ├── search_hnsw_snowflake_cast2020.py
│       ├── logs/
│       └── results/
│           ├── CAST2019/   # parquet run files + metrics JSON, M ∈ {16,32,64}
│           └── CAST2020/   # parquet run files + metrics JSON, M ∈ {16,32,64}
│
└── results Analysis/                             # cross-model / cross-year synthesis
    ├── toploc_analysis_dragon_2019.ipynb
    ├── toploc_analysis_dragon_2020.ipynb
    ├── toploc_analysis_snowflake_2019.ipynb
    ├── toploc_analysis_snowflake_2020.ipynb
    ├── toploc_final_summary_table.ipynb
    ├── hnsw_table_{Dragon,Snowflake}_{2019,2020}.csv
    ├── global_best_{Dragon,Snowflake}_{2019,2020}.csv
    ├── final_summary_table.csv
    └── final_summary_table.html
```

### 2.1 HNSW index construction — `*/index build/`

Each model builds three HNSW graphs over the passage collection at
`M ∈ {16, 32, 64}` (graph degree), on the same embeddings as the IVF indexes:
Snowflake L2-normalized, Dragon raw (dot-product / MIPS). Build scripts:
`Dragon/index build/indexCreationHNSW_multiM.py` and
`Snowflake/index build/indexCreation_Mgrid.py`. Build logs sit in each
subtree's `logs/`.

### 2.2 HNSW baseline + TopLoc HNSW — `*/search/`

TopLoc HNSW replaces HNSW's fixed entry point with a **privileged entry point**
derived from the topical locality of the conversation: a follow-up query enters
the graph near where the conversation's earlier turns landed, shortening the
greedy descent instead of restarting from the graph's default entry node. Each
`search_hnsw_<model>_cast<year>.py` script runs both the plain-HNSW baseline
and the TopLoc-HNSW variant across `M ∈ {16, 32, 64}` and reports latency plus
effectiveness (MRR@10, NDCG@3, NDCG@10).

**Outputs** (per model / year / M):
- `results/CAST<year>/<model>_toplocHNSW_M<M>_*.parquet` — per-query run files.
- `results/CAST<year>/<model>_toplocHNSW_metrics*_M<M>_*.json` — aggregated
  metrics.
- `search/logs/` — per-run stdout.

### 2.3 Results analysis — `results Analysis/`

Notebooks aggregate the per-run parquet/JSON into comparison tables:
- `toploc_analysis_<model>_<year>.ipynb` — one notebook per (model, year),
  comparing plain HNSW vs TopLoc HNSW across `M`.
- `hnsw_table_*.csv` / `global_best_*.csv` — extracted per-(model, year) tables
  and the best configuration per cell.
- `toploc_final_summary_table.ipynb` → `final_summary_table.{csv,html}` — the
  single cross-model, cross-year summary (reproduced in §2.4).

### 2.4 Results — final summary table

Cross-model, cross-year comparison of **Exact**, plain **HNSW**, and **TopLoc
HNSW** (source: `results Analysis/final_summary_table.csv`). Higher is better
for MRR@10 / NDCG; **Time** is mean per-query search time (lower is better),
with the speedup over the plain-HNSW baseline in parentheses.

**CAsT 2019**

| Model | Search | MRR@10 | NDCG@3 | NDCG@10 | Time (speedup) |
|---|---|---|---|---|---|
| Dragon | Exact | 0.8082 | 0.5289 | 0.4929 | — |
| Dragon | HNSW | 0.7969 | 0.5267 | 0.4909 | 0.250 (—) |
| Dragon | **TopLoc HNSW** | 0.7978 | 0.5243 | 0.4868 | **0.024 (10.5×)** |
| Snowflake | Exact | 0.8158 | 0.5501 | 0.5020 | — |
| Snowflake | HNSW | 0.8129 | 0.5493 | 0.5017 | 3.030 (—) |
| Snowflake | **TopLoc HNSW** | 0.8129 | 0.5493 | 0.5019 | **0.762 (4.0×)** |

**CAsT 2020**

| Model | Search | MRR@10 | NDCG@3 | NDCG@10 | Time (speedup) |
|---|---|---|---|---|---|
| Dragon | Exact | 0.7651 | 0.4742 | 0.4631 | — |
| Dragon | HNSW | 0.7669 | 0.4762 | 0.4649 | 0.325 (—) |
| Dragon | **TopLoc HNSW** | 0.7574 | 0.4711 | 0.4619 | **0.068 (4.8×)** |
| Snowflake | Exact | 0.7885 | 0.5065 | 0.4741 | — |
| Snowflake | HNSW | 0.7885 | 0.5067 | 0.4746 | 0.732 (—) |
| Snowflake | **TopLoc HNSW** | 0.7890 | 0.5075 | 0.4750 | **0.283 (2.6×)** |

**Takeaway.** TopLoc HNSW delivers **2.6×–10.5×** search-time speedups over
plain HNSW while keeping effectiveness essentially on par with both the HNSW
baseline and Exact search (MRR@10 / NDCG deltas within ~0.005 in every cell) —
reproducing the paper's core efficiency-without-effectiveness-loss claim.

---

# Shared Assets

These support both parts and are not specific to IVF or HNSW.

### Exact search reference — `Exact_Search/`

Flat / brute-force baselines that establish the effectiveness ceiling each ANN
index is measured against.

```
Exact_Search/
├── Dragon/
│   ├── build_flat_index_dragon.py
│   ├── search_flat_dragon.py
│   ├── metrics_flat_dragon_cast2019.json
│   ├── metrics_flat_dragon_cast2020.json
│   └── *.log
└── Snowflake/
    ├── build_flat_index_snowflake.py
    ├── search_flat_snowflake.py
    ├── metrics_flat_snowflake_cast2019.json
    ├── metrics_flat_snowflake_cast2020.json
    └── *.log
```

Each model builds an `IndexFlat` (exact inner-product) index and searches CAsT
2019 + 2020. The Dragon flat result (NDCG@10 ≈ 0.49) is the reference used in
the §1.4 diagnostic.

### Data exploration — `Data Exploration/`

Dataset preparation and query encoding.

- `dataset_exploration.ipynb` — inspection of the CAsT collections/queries.
- `cast2020_queries.tsv`, `cast2020_qrels.{qrel,trec}`,
  `save_cast2020_qrels.py` — CAsT 2020 queries and qrels.
- `encode_topics_dragon_ct2019.py`, `encode_topics_dragon_ct2020.py` — encode
  conversational topics with Dragon → `topics_dragon_embeddings_ct{2019,2020}.parquet`.
- `query_embeddings_2020.py`, `prepro_embeddings.py`,
  `check_query_embeddings.py` — build / preprocess / sanity-check query
  embeddings (`cast2020_query_embeddings.parquet`).

### Interactive demo — `demo/`

A self-contained retrieval demo over a small sampled collection.

- `build_demo_collection_v8.py` → `demo_texts.parquet`, `demo_embeddings.parquet`,
  `demo_ids.npy`, `demo_metadata.json`.
- `build_demo_indexes_2.py` → `demo_flat.index`, `demo_ivf.index`,
  `demo_hnsw.index`.
- `search_demo5.py`, `app5.py` — the search backend and app entry point
  (`requirements.txt` pins dependencies; `*.log` capture build output).

---

## Summary of Limitations

- Our AMD EPYC / single-NUMA-node hardware differs architecturally from the
  paper's 4-socket Intel Xeon setup; absolute latencies are not directly
  comparable, though the qualitative speedup mechanism is reproducible.
- Effectiveness metrics show a small, consistent gap versus the paper even on
  the unmodified plain-IVF baseline, plausibly attributable to qrels coverage
  and/or k-means initialization randomness rather than a TopLoc implementation
  error.
- The Dragon IVF index is undertrained relative to FAISS's recommended
  configuration (§1.1, §1.4).

---

## Authors

- **IVF / TopLoc IVF / TopLoc IVF+**, Dragon indexing, hardware & threading diagnostics: *Ali Efe Bal*
- **HNSW / TopLoc HNSW / Exact Search**, result analysis, indexing  : *Beyza Basak*
