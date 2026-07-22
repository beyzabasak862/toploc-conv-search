"""
search_demo3.py
===============
Search module for the CAST2019 demo collection.
Model: Snowflake/snowflake-arctic-embed-l-v2.0
"""

import os
import time
import csv
import logging
import numpy as np
import pyarrow.parquet as pq
import faiss
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DEMO_DIR         = Path("/home/toploc1/Datasets/toploc1/demo")
DEMO_TEXTS_PATH  = DEMO_DIR / "demo_texts.parquet"
DEMO_IDS_PATH    = DEMO_DIR / "demo_ids.npy"
FLAT_INDEX_PATH  = DEMO_DIR / "demo_flat.index"
IVF_INDEX_PATH   = DEMO_DIR / "demo_ivf.index"
HNSW_INDEX_PATH  = DEMO_DIR / "demo_hnsw.index"

TOPICS_TSV_PATH = Path("/home/toploc1/Datasets/conversational/CAST2019/topics/topics.tsv")
QUERY_EMB_PATH  = Path("/home/toploc1/Datasets/conversational/CAST2019/topics/topics_snowflake_embeddings.parquet")
QRELS_PATH      = Path("/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel")

MODEL_LOCAL_PATH = "/home/toploc1/Datasets/toploc1/models/snowflake-arctic-embed-l-v2.0"
MODEL_HF_NAME    = "Snowflake/snowflake-arctic-embed-l-v2.0"

QUERY_PREFIX = "query: "

# ── Allowed hyperparameter values ─────────────────────────────────────────────
NPROBE_VALUES    = [2, 4, 8, 16, 32, 64]
H_VALUES         = [32, 64, 128]
EF_SEARCH_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

NPROBE_DEFAULT    = 16
H_DEFAULT         = 64
EF_SEARCH_DEFAULT = 64


@dataclass
class SearchResult:
    rank:   int
    doc_id: str
    text:   str
    score:  float


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def load_topics() -> dict[str, str]:
    topics = {}
    with open(TOPICS_TSV_PATH, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2:
                topics[row[0].strip()] = row[1].strip()
    log.info(f"Loaded {len(topics)} topics")
    return topics


def load_query_embeddings() -> dict[str, np.ndarray]:
    tbl  = pq.read_table(QUERY_EMB_PATH)
    ids  = tbl["id"].to_pylist()
    embs = tbl["embedding"].to_pylist()
    return {str(qid): np.array(emb, dtype=np.float32) for qid, emb in zip(ids, embs)}


def load_qrel_qids() -> set:
    qids = set()
    with open(QRELS_PATH, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row:
                qids.add(row[0].strip())
    log.info(f"Loaded {len(qids)} unique qrel query IDs")
    return qids


# ══════════════════════════════════════════════
# Main searcher
# ══════════════════════════════════════════════

class DemoSearcher:

    def __init__(self):
        log.info("Initialising DemoSearcher ...")

        log.info("Loading demo_texts.parquet ...")
        tbl = pq.read_table(DEMO_TEXTS_PATH, columns=["id", "text"])
        self.text_map: dict[str, str] = dict(
            zip(tbl["id"].to_pylist(), tbl["text"].to_pylist())
        )
        log.info(f"  {len(self.text_map):,} texts loaded")

        self.id_map: list[str] = np.load(DEMO_IDS_PATH, allow_pickle=True).tolist()

        log.info("Loading FAISS indexes ...")
        self.flat_index = faiss.read_index(str(FLAT_INDEX_PATH))
        self.ivf_index  = faiss.read_index(str(IVF_INDEX_PATH))
        self.hnsw_index = faiss.read_index(str(HNSW_INDEX_PATH))
        log.info(
            f"  Flat={self.flat_index.ntotal:,}  "
            f"IVF={self.ivf_index.ntotal:,} (nlist={self.ivf_index.nlist})  "
            f"HNSW={self.hnsw_index.ntotal:,}"
        )

        self.topics     = load_topics()
        self.query_embs = load_query_embeddings()
        self.qrel_qids  = load_qrel_qids()

        # TopLoc state
        self._centroid_matrix: np.ndarray | None = None
        self._conv_cache: dict[str, tuple] = {}

        log.info("Loading embedding model ...")
        self.model_available = False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            source = MODEL_LOCAL_PATH if os.path.exists(MODEL_LOCAL_PATH) else MODEL_HF_NAME
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model     = AutoModel.from_pretrained(source, add_pooling_layer=False)
            self._device    = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()
            self.model_available = True
            log.info(f"  Model ready on {self._device}.")
        except Exception as e:
            self._tokenizer = self._model = self._device = None
            log.warning(f"  Embedding model not available: {e}")

        log.info("DemoSearcher ready.")

    # ── Embedding ──────────────────────────────────────────────────────────
    def embed(self, text: str) -> np.ndarray:
        if not self.model_available:
            raise RuntimeError("Free-text search unavailable.")
        import torch
        import torch.nn.functional as F
        inputs = self._tokenizer(
            [QUERY_PREFIX + text], padding=True, truncation=True,
            return_tensors="pt", max_length=8192,
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        emb = out[0][:, 0]
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    # ── Standard search ────────────────────────────────────────────────────
    def _pick_index(self, index_type: str):
        t = index_type.lower()
        if t in ("flat", "exact"): return self.flat_index
        if t == "ivf":             return self.ivf_index
        if t == "hnsw":            return self.hnsw_index
        raise ValueError(f"Unknown index_type '{index_type}'")

    def search(self, query_vec: np.ndarray, index_type: str = "hnsw",
               top_k: int = 5, nprobe: int = NPROBE_DEFAULT,
               ef_search: int = EF_SEARCH_DEFAULT):
        vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        index = self._pick_index(index_type)
        if index_type == "ivf":
            index.nprobe = nprobe
        if index_type == "hnsw":
            index.hnsw.efSearch = ef_search
        t0 = time.perf_counter()
        scores, positions = index.search(vec, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        results = []
        for rank, (pos, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if pos == -1:
                continue
            doc_id = self.id_map[pos]
            text   = self.text_map.get(doc_id, "[text not found]")
            results.append(SearchResult(rank=rank, doc_id=doc_id, text=text, score=float(score)))
        return results, latency_ms

    def search_by_topic(self, query_id: str, index_type: str = "hnsw",
                        top_k: int = 5, nprobe: int = NPROBE_DEFAULT,
                        ef_search: int = EF_SEARCH_DEFAULT):
        if query_id not in self.query_embs:
            raise KeyError(f"Query ID '{query_id}' not found.")
        vec        = self.query_embs[query_id]
        query_text = self.topics.get(query_id, "[unknown query]")
        results, latency_ms = self.search(
            vec, index_type=index_type, top_k=top_k,
            nprobe=nprobe, ef_search=ef_search
        )
        return results, latency_ms, query_text

    def search_free_text(self, text: str, index_type: str = "hnsw",
                         top_k: int = 5, nprobe: int = NPROBE_DEFAULT,
                         ef_search: int = EF_SEARCH_DEFAULT):
        vec = self.embed(text)
        return self.search(vec, index_type=index_type, top_k=top_k,
                           nprobe=nprobe, ef_search=ef_search)

    # ── TopLoc IVF — single query ──────────────────────────────────────────
    def _load_centroid_matrix(self) -> None:
        quant = faiss.downcast_index(self.ivf_index.quantizer)
        self._centroid_matrix = quant.reconstruct_n(0, self.ivf_index.nlist)
        log.info(f"  Centroid matrix: {self._centroid_matrix.shape}")

    def clear_conv_cache(self) -> None:
        self._conv_cache.clear()

    def search_toploc_by_topic(
        self,
        query_id: str,
        h:        int = H_DEFAULT,
        nprobe:   int = NPROBE_DEFAULT,
        top_k:    int = 5,
    ) -> tuple[list, float, str, bool, int, int]:
        """Returns: results, latency_ms, query_text, is_first_turn, nprobe_used, h_used"""
        if query_id not in self.query_embs:
            raise KeyError(f"Query ID '{query_id}' not found.")
        if self._centroid_matrix is None:
            self._load_centroid_matrix()

        vec = self.query_embs[query_id].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        query_text = self.topics.get(query_id, "[unknown]")

        parts   = query_id.rsplit("_", 1)
        conv_id = parts[0]
        turn    = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1

        h_used      = min(h, self.ivf_index.nlist)
        nprobe_used = min(nprobe, h_used)
        quant       = faiss.downcast_index(self.ivf_index.quantizer)

        if turn == 1 or conv_id not in self._conv_cache:
            D_c, I_c    = quant.search(vec, h_used)
            cached_ids  = I_c[0].copy()
            cached_vecs = self._centroid_matrix[cached_ids].copy()
            small_idx   = faiss.IndexFlatIP(self.ivf_index.d)
            small_idx.add(cached_vecs)
            self._conv_cache[conv_id] = (cached_ids, small_idx)
            assigned_ids   = cached_ids[:nprobe_used].reshape(1, -1).astype(np.int64)
            assigned_dists = D_c[0, :nprobe_used].reshape(1, -1).astype(np.float32)
            is_first_turn  = True
        else:
            cached_ids, small_idx = self._conv_cache[conv_id]
            nprobe_used   = min(nprobe_used, len(cached_ids))
            D_local, I_local = small_idx.search(vec, nprobe_used)
            assigned_ids   = cached_ids[I_local[0]].reshape(1, -1).astype(np.int64)
            assigned_dists = D_local[0].reshape(1, -1).astype(np.float32)
            is_first_turn  = False

        self.ivf_index.nprobe = nprobe_used
        t0 = time.perf_counter()
        D_res, I_res = self.ivf_index.search_preassigned(vec, top_k, assigned_ids, assigned_dists)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (pos, score) in enumerate(zip(I_res[0], D_res[0]), start=1):
            if pos == -1:
                continue
            doc_id = self.id_map[pos]
            text   = self.text_map.get(doc_id, "[text not found]")
            results.append(SearchResult(rank=rank, doc_id=doc_id, text=text, score=float(score)))

        return results, latency_ms, query_text, is_first_turn, nprobe_used, h_used

    # ── Full conversation comparison — all four methods ────────────────────
    def get_conversation_turns(self, conv_id: str) -> list[str]:
        turn_ids = [
            qid for qid in self.qrel_qids
            if qid.rsplit("_", 1)[0] == conv_id
            and len(qid.rsplit("_", 1)) == 2
            and qid.rsplit("_", 1)[1].isdigit()
            and qid in self.query_embs
        ]
        return sorted(turn_ids, key=lambda x: int(x.rsplit("_", 1)[1]))

    def run_conversation_comparison(
        self,
        conv_id:   str,
        nprobe:    int = NPROBE_DEFAULT,
        h:         int = H_DEFAULT,
        ef_search: int = EF_SEARCH_DEFAULT,
        top_k:     int = 5,
        n_runs:    int = 5,
    ) -> list[dict]:
        """
        Run all turns of a conversation n_runs times through all four methods:
        Flat, HNSW (with ef_search), Plain IVF (with nprobe), TopLoc IVF (with h, nprobe).
        Returns per-turn average latencies.
        """
        turn_ids = self.get_conversation_turns(conv_id)
        if not turn_ids:
            return []

        if self._centroid_matrix is None:
            self._load_centroid_matrix()

        h_use      = min(h, self.ivf_index.nlist)
        nprobe_use = min(nprobe, h_use)
        quant      = faiss.downcast_index(self.ivf_index.quantizer)

        flat_times   = {qid: [] for qid in turn_ids}
        hnsw_times   = {qid: [] for qid in turn_ids}
        ivf_times    = {qid: [] for qid in turn_ids}
        toploc_times = {qid: [] for qid in turn_ids}

        for _ in range(n_runs):

            # ── Flat ──────────────────────────────────────────────────────
            for qid in turn_ids:
                vec = self.query_embs[qid].reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vec)
                t0 = time.perf_counter()
                self.flat_index.search(vec, top_k)
                flat_times[qid].append((time.perf_counter() - t0) * 1000)

            # ── HNSW ──────────────────────────────────────────────────────
            self.hnsw_index.hnsw.efSearch = ef_search
            for qid in turn_ids:
                vec = self.query_embs[qid].reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vec)
                t0 = time.perf_counter()
                self.hnsw_index.search(vec, top_k)
                hnsw_times[qid].append((time.perf_counter() - t0) * 1000)

            # ── Plain IVF ─────────────────────────────────────────────────
            self.ivf_index.nprobe = nprobe_use
            for qid in turn_ids:
                vec = self.query_embs[qid].reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vec)
                t0 = time.perf_counter()
                self.ivf_index.search(vec, top_k)
                ivf_times[qid].append((time.perf_counter() - t0) * 1000)

            # ── TopLoc IVF ────────────────────────────────────────────────
            self._conv_cache.pop(conv_id, None)

            for qid in turn_ids:
                vec  = self.query_embs[qid].reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vec)
                turn = int(qid.rsplit("_", 1)[1])

                if turn == 1 or conv_id not in self._conv_cache:
                    t0 = time.perf_counter()
                    D_c, I_c    = quant.search(vec, h_use)
                    cached_ids  = I_c[0].copy()
                    cached_vecs = self._centroid_matrix[cached_ids].copy()
                    small_idx   = faiss.IndexFlatIP(self.ivf_index.d)
                    small_idx.add(cached_vecs)
                    self._conv_cache[conv_id] = (cached_ids, small_idx)
                    assigned_ids   = cached_ids[:nprobe_use].reshape(1, -1).astype(np.int64)
                    assigned_dists = D_c[0, :nprobe_use].reshape(1, -1).astype(np.float32)
                    self.ivf_index.nprobe = nprobe_use
                    self.ivf_index.search_preassigned(vec, top_k, assigned_ids, assigned_dists)
                    toploc_times[qid].append((time.perf_counter() - t0) * 1000)
                else:
                    cached_ids, small_idx = self._conv_cache[conv_id]
                    np_use = min(nprobe_use, len(cached_ids))
                    t0 = time.perf_counter()
                    D_local, I_local = small_idx.search(vec, np_use)
                    assigned_ids   = cached_ids[I_local[0]].reshape(1, -1).astype(np.int64)
                    assigned_dists = D_local[0].reshape(1, -1).astype(np.float32)
                    self.ivf_index.nprobe = np_use
                    self.ivf_index.search_preassigned(vec, top_k, assigned_ids, assigned_dists)
                    toploc_times[qid].append((time.perf_counter() - t0) * 1000)

        rows = []
        for qid in turn_ids:
            turn       = int(qid.rsplit("_", 1)[1])
            query_text = self.topics.get(qid, "")
            flat_avg   = float(np.mean(flat_times[qid]))
            hnsw_avg   = float(np.mean(hnsw_times[qid]))
            ivf_avg    = float(np.mean(ivf_times[qid]))
            toploc_avg = float(np.mean(toploc_times[qid]))
            rows.append({
                "turn":       turn,
                "query_id":   qid,
                "query_text": query_text[:70] + "..." if len(query_text) > 70 else query_text,
                "flat_ms":    flat_avg,
                "hnsw_ms":    hnsw_avg,
                "ivf_ms":     ivf_avg,
                "toploc_ms":  toploc_avg,
                "speedup":    ivf_avg / toploc_avg if toploc_avg > 0 else 1.0,
                "is_first":   turn == 1,
            })
        return rows