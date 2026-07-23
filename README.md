# Accelerating Dense Retrieval for Conversational Search — Two Replication Studies

This repository contains our replications of **two** SIGIR papers on making
Approximate Nearest Neighbor (ANN) search faster for dense conversational
retrieval, without sacrificing effectiveness. Both are evaluated on the same
family of datasets (**TREC CAsT 2019 / 2020**, MS MARCO) and the same two
embedding models (**Snowflake** and **Dragon**), which lets the two studies
share indexes, embeddings, and baselines.

| | Paper 1 — Topical Locality | Paper 2 — HNSW QLR |
|---|---|---|
| **Idea** | Cache the "hot" region of the index found by a conversation's first query, then restrict follow-up queries to it | Route a query to a good HNSW entry point using a query log, instead of the graph's default entry node |
| **Indexes** | FAISS IVF + HNSW | HNSW (FAISS "hybrid" + native C++) |
| **Where** | [`Paper1-Topical Locality/`](Paper1-Topical%20Locality/README.md) | [`Paper2-HNSW_QLR/`](Paper2-HNSW_QLR/README.md) |

Each paper directory has its **own detailed README** documenting methodology,
file architecture, and results. This top-level document is the entry point and
describes the shared setup.

---

## The two studies

### Paper 1 — Efficient Conversational Search via Topical Locality in Dense Retrieval

> Muntean, C. I., Nardini, F. M., Perego, R., Rocchietti, G., & Rulli, C. (2025).
> *Efficient Conversational Search via Topical Locality in Dense Retrieval.*
> SIGIR '25, 2749–2753.

Introduces **TopLoc**, which exploits the topical locality of conversational
queries to accelerate ANN search. The follow-up utterances of a conversation
tend to land in the same region of the index as the first utterance, so TopLoc
caches that region and reuses it. Two integrations are replicated:

- **TopLoc IVF** (+ a cache-refresh variant, **TopLoc IVF+**) on FAISS's
  Inverted File index.
- **TopLoc HNSW** on the HNSW graph, via a privileged entry point.

Our replication reproduces the paper's efficiency and effectiveness claims and
reports **2.6×–10.5×** HNSW search-time speedups at essentially unchanged
effectiveness. Full details, file map, and the results table:
**[`Paper1-Topical Locality/README.md`](Paper1-Topical%20Locality/README.md)**.

### Paper 2 — HNSW QLR (Query-Log Routing)

Evaluates a **query-log-routed (QLR)** HNSW search: instead of starting the
greedy graph descent from HNSW's fixed entry point, a query is seeded from a
good entry point learned from a query log, then searched with a reduced `ef`.
The replication is packaged as **eight benchmark workflows** across three
implementation tracks — hybrid FAISS on the full TREC-CAsT index, a native C++
implementation on a 500k MS MARCO-v1 export, and a faithful adaptive
search-depth variant (Paper 2, Algorithm 1). Full details, run scripts, and
per-benchmark reports:
**[`Paper2-HNSW_QLR/README.md`](Paper2-HNSW_QLR/README.md)**.

---

## Repository layout (top level)

```
toploc1/
├── Paper1-Topical Locality/   # Paper 1 replication — TopLoc IVF / IVF+ / HNSW  (has its own README)
├── Paper2-HNSW_QLR/           # Paper 2 replication — HNSW query-log routing     (has its own README)
│
├── indexes/                   # Shared prebuilt indexes (Dragon / Snowflake IVF, HNSW, Flat)
├── models/                    # Local embedding models (Snowflake Arctic-Embed L / L-v2.0)
├── HNSW/                      # Working area for the Paper 2 QLR development (MS MARCO, PCA, artifacts)
│
├── faiss/                     # FAISS source checkout
└── faiss_install/             # Locally built FAISS install (include / lib / share)
```

Because both papers use the same encoders and datasets, `indexes/` and
`models/` are shared across them. The `Dragon` and `Snowflake` embedding
models and the CAsT collections are the common substrate; each paper directory
then holds only its own scripts, results, and analysis.

---

## Shared environment and hardware

All experiments were run on **Pegasus** (accessed via a jump server), with a
secondary machine (**big-dama-3**) used for parallel/overflow workloads
(e.g. building an index while search experiments ran elsewhere).

| | Paper's server | Pegasus (primary) |
|---|---|---|
| CPU | 4× Intel Xeon Gold 6252N | 1× AMD EPYC 7702P |
| Cores / threads | 96 physical / 192 logical | 64 physical / 128 logical (SMT) |
| NUMA nodes | 4 | **1** |
| Isolation method | `numactl --cpunodebind` | not applicable — single NUMA node |

**Divergence from the papers' protocol.** The original work runs retrieval
under `numactl`, confining execution to a single CPU socket and its local
memory. Pegasus has only **one NUMA node**, so this isolation technique does
not apply (there is no second socket to isolate against). This most affects the
IVF timing methodology in Paper 1 — see its README.

**Practical note for reproducing a run:** launch long-running search scripts
with `python3 -u` (unbuffered stdout) under `nohup`, and check `uptime` /
`htop` beforehand — several early runs were contaminated by co-tenant load on
the shared server (visible as physically inconsistent latencies, e.g. a smaller
`k` reporting higher latency than a larger `k`).

---

## Where to go next

- **Paper 1 methodology, architecture, and results:**
  [`Paper1-Topical Locality/README.md`](Paper1-Topical%20Locality/README.md)
- **Paper 2 benchmarks, run scripts, and reports:**
  [`Paper2-HNSW_QLR/README.md`](Paper2-HNSW_QLR/README.md)

---

## Authors

- **Paper 1** — IVF / TopLoc IVF / TopLoc IVF+, Dragon indexing, hardware &
  threading diagnostics: *[your name]*; HNSW / TopLoc HNSW, results analysis:
  *[teammate's name]*
- **Paper 2** — HNSW QLR replication: *[author name(s)]*
