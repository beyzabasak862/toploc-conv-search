# Accelerating Dense Retrieval for Conversational Search — Two Replication Studies

This repository contains our replications of **two** SIGIR papers on making
Approximate Nearest Neighbor (ANN) search faster for dense conversational
retrieval, without sacrificing effectiveness. Both are evaluated on the same
family of datasets (**TREC CAsT 2019 / 2020**, MS MARCO) and the same two
embedding models (**Snowflake** and **Dragon**), which lets the two studies
share indexes, embeddings, and baselines.

| | Paper 1 — Topical Locality | Paper 2 — HNSW QLR |
|---|---|---|
| **Idea** | Cache the "hot" region of the index found by a conversation's first query, then restrict follow-up queries to it | Seed HNSW's ground-level beam search from the cached results of similar *historical* queries, instead of the graph's fixed entry point |
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

> Paper: *HNSW Graph Meets Query Logs: Accelerating Dense Retrieval with
> Historical Information.
* Introduces the **Query Log Router (QLR)**, a
lightweight auxiliary ANN index built over a sample of historical query vectors
plus a lookup table mapping each to its precomputed nearest neighbors. At query
time, QLR finds past queries similar to the incoming one and uses their cached
neighbors to seed HNSW's ground-level beam search — where most of the cost lies
— from an already-close position; when no sufficiently similar query is found
(similarity `< th`) it falls back to standard HNSW with minimal overhead. An
**adaptive** mechanism lowers `ef_search` when the routing decision is
confident, and **PCA** shrinks the query-log index so its lookup adds little
latency. The paper reports speedups of up to **1.8× (MS MARCO-v1)**, **2.6×
(MS MARCO-v2)**, and **2.3×** with a real-world MSN query log, at equal
Accuracy@10.

Our replication is packaged as **eight benchmark workflows** across three
implementation tracks — hybrid FAISS on the full TREC-CAsT index, a native C++
implementation on a 500k MS MARCO-v1 export, and a faithful adaptive
search-depth variant (Algorithm 1). Full details, run scripts, and
per-benchmark reports:
**[`Paper2-HNSW_QLR/README.md`](Paper2-HNSW_QLR/README.md)**.

---

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


---

## Where to go next

- **Paper 1 methodology, architecture, and results:**
  [`Paper1-Topical Locality/README.md`](Paper1-Topical%20Locality/README.md)
- **Paper 2 benchmarks, run scripts, and reports:**
  [`Paper2-HNSW_QLR/README.md`](Paper2-HNSW_QLR/README.md)
