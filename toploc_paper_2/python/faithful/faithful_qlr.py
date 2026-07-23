# ==================== CLAUDE IMPROVEMENT START ====================
# Faithful Paper 2 (QLR) Algorithm 1 implementation.
#
# Every step follows the paper's Algorithm 1:
#   1: {(q_1,s_1)..(q_k',s_k')} <- Search(I_Q, q, k')
#   2: s <- s_1
#   3-4: if s < th: return HnswSearch(I_D, q, k, ef)
#   5: C <- union_{i=1..k'} EP(q_i)          [dedup]
#   6-9: adaptive ef' = ef_min + (ef-ef_min)*(s_max-s)/(s_max-th) clipped
#   10: return BeamSearch(I_D, q, k, ef', C)  [FAISS search_level_0 search_type=2]
#
# Gaps vs frozen (k'=1, k_ep=3, no adaptive, search_type=1):
#   * k' extended from 1 to {5,10,20}
#   * k_ep extended from 3 to {5,10}
#   * union C across all k' historical queries, deduped
#   * adaptive ef' per paper formula
#   * search_type=2 (pooled beam) instead of search_type=1 (per-seed)
#
# All timed FAISS single-thread; PCA under threadpool_limits(1) inside caller.
# ==================== CLAUDE IMPROVEMENT END ====================
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import faiss

_F32 = np.float32
_I32 = np.int32
_I64 = np.int64


@dataclass
class QLRConfig:
    kp: int              # k' = number of historical queries retrieved
    kep: int             # k_ep used from each row (<=10, EP width)
    th: float            # fallback threshold (router IP top-1 score)
    ef_min: int          # min beam width (adaptive lower)
    ef_default: int      # baseline/fallback beam width, and adaptive upper bound
    s_max: float         # 75th pct of q_l top-1 similarity (paper)
    router_ef: int       # I_Q efSearch during router call
    search_type: int     # FAISS search_level_0 search_type (1=per-seed, 2=pooled beam)
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        return f"kp{self.kp}_kep{self.kep}_th{self.th:.2f}_efmin{self.ef_min}_ef{self.ef_default}_st{self.search_type}"


@dataclass
class QLRHandles:
    doc_index: object            # faiss IndexHNSWFlat, L2 metric
    query_index: object          # faiss IndexHNSWFlat, IP metric (PCA space)
    ep_indices: np.ndarray       # int32 [nQL, 10]
    ep_distances: np.ndarray     # float32 [nQL, 10]  (native metric of doc index; L2)
    pca_mean: np.ndarray         # float32 [1024]
    pca_components: np.ndarray   # float32 [256, 1024]
    topk: int = 10


class FaithfulQLR:
    """Paper 2 Alg 1 in an object. Not thread-safe. Reuses preallocated buffers."""

    def __init__(self, h: QLRHandles, max_c: int = 200):
        self.h = h
        self.topk = h.topk
        self.dim = h.pca_mean.shape[0]           # 1024
        self.pca_dim = h.pca_components.shape[0]  # 256
        self.max_c = max_c
        # Preallocated per-query buffers (reused every query)
        self._qp = np.empty((1, self.pca_dim), _F32)
        self._q_diff = np.empty(self.dim, _F32)  # not used but reserved
        self._Dh = np.empty((1, max(20, 32)), _F32)
        self._Ih = np.empty((1, max(20, 32)), _I64)
        self._sv = np.empty((max_c, self.dim), _F32)
        self._diff = np.empty((max_c, self.dim), _F32)
        self._dc = np.empty(max_c, _F32)
        self._Ic = np.empty(max_c, _I32)
        self._Dq = np.zeros((1, self.topk), _F32)
        self._Iq = np.zeros((1, self.topk), _I64)

    # ---- Step 1a: PCA (paper says amortized <20us at batch 32) ----
    def pca_transform(self, q_row: np.ndarray) -> np.ndarray:
        """q_row: [1, 1024] f32 (already l2-normalized). Returns [1, 256] f32 buffer view.

        Implements: qp = (q - mean) @ C.T   using preallocated output buffer _qp.
        """
        diff = q_row - self.h.pca_mean            # allocates (1,1024), cheap
        np.matmul(diff, self.h.pca_components.T, out=self._qp)
        return self._qp

    # ---- Step 5: union of EP over top-k' historical queries, deduped ----
    def union_ep(self, Ih_row: np.ndarray, kep: int) -> np.ndarray:
        """
        Returns dedup int32 array of unique candidate doc IDs, order-preserving
        by first-occurrence (row-major flatten of EP[Ih_row, :kep]).
        """
        # EP[Ih_row, :kep] flattens to length kp*kep
        # Use fancy indexing; Ih_row is int64
        block = self.h.ep_indices[Ih_row, :kep]  # int32 [kp, kep]
        flat = block.reshape(-1)                  # int32 [kp*kep]
        # dedup preserving first-occurrence order
        _, first_idx = np.unique(flat, return_index=True)
        first_idx.sort()
        uniq = flat[first_idx]
        return uniq

    # ---- Step 5b: seed L2 distances for current query (Paper says pooled beam init) ----
    def compute_seed_dists(self, ids: np.ndarray, q_row: np.ndarray) -> tuple:
        """
        Reconstruct seed vectors from doc index; compute squared-L2 to q_row.
        ids: int32 [n]. q_row: [1024] f32.
        Returns (Ic_int32, Dc_float32) both [n], sorted by increasing distance
        (recommended for search_type=2 beam initialization).
        """
        n = len(ids)
        # cap to buffer size
        n = min(n, self.max_c)
        sv = self._sv[:n]
        # reconstruct (uses reconstruct which returns f32 [dim])
        dx = self.h.doc_index
        for k in range(n):
            sv[k] = dx.reconstruct(int(ids[k]))
        # squared-L2: ||sv - q||^2 via einsum
        diff = self._diff[:n]
        np.subtract(sv, q_row[None, :], out=diff)
        dc = self._dc[:n]
        np.einsum('ij,ij->i', diff, diff, out=dc)
        # sort by ascending distance (closest first)
        order = np.argsort(dc, kind="stable")
        Ic_sorted = np.ascontiguousarray(ids[order[:n]].astype(_I32, copy=False))
        Dc_sorted = np.ascontiguousarray(dc[order[:n]])
        return Ic_sorted, Dc_sorted

    # ---- Step 6-9: adaptive ef' ----
    @staticmethod
    def adaptive_ef(s: float, s_max: float, th: float,
                    ef_min: int, ef_default: int) -> int:
        if s > s_max:
            return int(ef_min)
        denom = s_max - th
        if denom <= 0:
            return int(ef_default)
        ef_ = ef_min + (ef_default - ef_min) * (s_max - s) / denom
        if ef_ < ef_min:
            return int(ef_min)
        if ef_ > ef_default:
            return int(ef_default)
        return int(round(ef_))

    # ---- Step 10: seeded ground-level beam search ----
    def seeded_beam(self, q_row: np.ndarray, Ic: np.ndarray, Dc: np.ndarray,
                    ef_prime: int, search_type: int):
        """
        Fresh beam initialized from candidate set Ic with distances Dc.
        Uses FAISS search_level_0 with the requested search_type.
        """
        dx = self.h.doc_index
        dx.hnsw.efSearch = ef_prime
        n = len(Ic)
        Dq = self._Dq
        Iq = self._Iq
        # 9-arg call to include search_type
        dx.search_level_0(
            1,
            faiss.swig_ptr(q_row),
            self.topk,
            faiss.swig_ptr(Ic),
            faiss.swig_ptr(Dc),
            faiss.swig_ptr(Dq),
            faiss.swig_ptr(Iq),
            n,                # nprobe = number of pooled seeds (search_type=2 uses all)
            search_type,
        )
        return Iq[0].copy(), Dq[0].copy()

    # ---- Fallback: full HNSW at ef_default ----
    def fallback_hnsw(self, q_row: np.ndarray, ef_default: int):
        dx = self.h.doc_index
        dx.hnsw.efSearch = ef_default
        # standard search: q_row is [1, 1024]
        D, I = dx.search(q_row.reshape(1, -1), self.topk)
        return I[0].copy(), D[0].copy()


# ==================== end faithful_qlr.py ====================
