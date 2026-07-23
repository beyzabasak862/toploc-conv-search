// ==================== CLAUDE IMPROVEMENT START ====================
// native_qlr_v2 — engineering-improved candidate over native_qlr.
// Preserves the verified pipeline semantics (Paper 2 Algorithm 1):
//  - PCA(q) -> qp
//  - Router: seeded (from EP) HNSW beam on I_Q, top-kp historical queries
//  - If s_top1 < th: baseline HNSW fallback on doc index at ef_default
//  - Else: union C from k' × k_ep EP hits, score against q, seeded beam on doc index at ef'
//
// Backend improvements (all preserve the seed set and its scoring exactly):
//   (i)   Zero per-query allocation.  All working buffers live in QueryContext
//         and are reused via preallocated arrays; the frontier and result set
//         are fixed-capacity binary heaps built on those arrays.
//   (ii)  Bounded top-K extraction — partial_sort only the top-K rather than
//         drain+sort the whole ef-sized min-heap.
//   (iii) 4-accumulator AVX2/FMA dot product with 32-lane unroll to hide
//         Zen2 FMA latency (4c latency, 2/cycle throughput).
//   (iv)  Deeper software prefetch — 8 neighbors ahead + PCA row prefetch.
//   (v)   PCA transform: fused (q-mean) subtraction and stream-through matmul,
//         thread-local diff buffer stays warm across queries.
//   (vi)  I_Q router beam uses the same zero-alloc frontier machinery.
//
// The module name is `native_qlr_v2` so it coexists with the verified backend
// side by side for interleaved measurement.
// v3 addition: fp16 iq_vecs and F16C-based dot product for the router. Doc
// side stays fp32.
// ==================== CLAUDE IMPROVEMENT END ====================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <memory>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;

// ============================ F16C fp16 -> fp32 router kernel ============================
// 256-dim (iq_dim) dot product: 4-accumulator, 32-lane unroll, load fp16 and
// on-the-fly convert to fp32 using F16C (_mm256_cvtph_ps).
#if defined(__AVX2__) && defined(__F16C__)
static inline float dot_f16_f32_avx2(const uint16_t* __restrict a_f16,
                                     const float* __restrict b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a_f16 + i +  0)));
        __m256 a1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a_f16 + i +  8)));
        __m256 a2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a_f16 + i + 16)));
        __m256 a3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a_f16 + i + 24)));
        __m256 y0 = _mm256_loadu_ps(b + i +  0);
        __m256 y1 = _mm256_loadu_ps(b + i +  8);
        __m256 y2 = _mm256_loadu_ps(b + i + 16);
        __m256 y3 = _mm256_loadu_ps(b + i + 24);
        s0 = _mm256_fmadd_ps(a0, y0, s0);
        s1 = _mm256_fmadd_ps(a1, y1, s1);
        s2 = _mm256_fmadd_ps(a2, y2, s2);
        s3 = _mm256_fmadd_ps(a3, y3, s3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a_f16 + i)));
        __m256 y = _mm256_loadu_ps(b + i);
        s0 = _mm256_fmadd_ps(a, y, s0);
    }
    __m256 s01 = _mm256_add_ps(s0, s1);
    __m256 s23 = _mm256_add_ps(s2, s3);
    __m256 s   = _mm256_add_ps(s01, s23);
    __m128 lo  = _mm256_castps256_ps128(s);
    __m128 hi  = _mm256_extractf128_ps(s, 1);
    __m128 t   = _mm_add_ps(lo, hi);
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x1));
    float out = _mm_cvtss_f32(t);
    for (; i < n; ++i) {
        float av = _cvtsh_ss(a_f16[i]);
        out += av * b[i];
    }
    return out;
}
#else
static inline float dot_f16_f32_avx2(const uint16_t* a_f16, const float* b, int n) {
    // Portable fallback: scalar half->float via bit ops.
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        uint16_t h = a_f16[i];
        uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
        uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
        uint32_t mant = (uint32_t)h & 0x3ffu;
        uint32_t f;
        if (exp == 0) f = (sign << 31);
        else if (exp == 31) f = (sign << 31) | 0x7f800000u;
        else { uint32_t e = exp - 15 + 127; f = (sign << 31) | (e << 23) | (mant << 13); }
        float av; std::memcpy(&av, &f, 4);
        s += av * b[i];
    }
    return s;
}
#endif

// ============================ AVX2 dot product — 4-acc / 32-lane ============================
#if defined(__AVX2__)
static inline float dot_f32_avx2_4acc(const float* __restrict a,
                                      const float* __restrict b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    // Unroll by 32: 4 independent FMA chains keeps Zen2 FMA units saturated.
    for (; i + 32 <= n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(a + i +  0);
        __m256 y0 = _mm256_loadu_ps(b + i +  0);
        __m256 x1 = _mm256_loadu_ps(a + i +  8);
        __m256 y1 = _mm256_loadu_ps(b + i +  8);
        __m256 x2 = _mm256_loadu_ps(a + i + 16);
        __m256 y2 = _mm256_loadu_ps(b + i + 16);
        __m256 x3 = _mm256_loadu_ps(a + i + 24);
        __m256 y3 = _mm256_loadu_ps(b + i + 24);
        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        s3 = _mm256_fmadd_ps(x3, y3, s3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(a + i);
        __m256 y = _mm256_loadu_ps(b + i);
        s0 = _mm256_fmadd_ps(x, y, s0);
    }
    __m256 s01 = _mm256_add_ps(s0, s1);
    __m256 s23 = _mm256_add_ps(s2, s3);
    __m256 s   = _mm256_add_ps(s01, s23);
    __m128 lo  = _mm256_castps256_ps128(s);
    __m128 hi  = _mm256_extractf128_ps(s, 1);
    __m128 t   = _mm_add_ps(lo, hi);
    t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x1));
    float out = _mm_cvtss_f32(t);
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}
#else
static inline float dot_f32_avx2_4acc(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

// ============================ mmap loader ============================
struct MmapBlob {
    void*  ptr   = nullptr;
    size_t bytes = 0;
    int    fd    = -1;
    MmapBlob() = default;
    MmapBlob(const std::string& path) { open(path); }
    ~MmapBlob() { close(); }
    MmapBlob(const MmapBlob&) = delete;
    MmapBlob& operator=(const MmapBlob&) = delete;
    MmapBlob(MmapBlob&& o) noexcept : ptr(o.ptr), bytes(o.bytes), fd(o.fd) {
        o.ptr = nullptr; o.bytes = 0; o.fd = -1;
    }
    void open(const std::string& path) {
        close();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed: " + path);
        struct stat st;
        if (fstat(fd, &st) < 0) throw std::runtime_error("fstat failed: " + path);
        bytes = st.st_size;
        ptr = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (ptr == MAP_FAILED) { ptr = nullptr;
            throw std::runtime_error("mmap failed: " + path); }
        madvise(ptr, bytes, MADV_WILLNEED);
    }
    void close() {
        if (ptr) { munmap(ptr, bytes); ptr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    template <class T> T* data() { return reinterpret_cast<T*>(ptr); }
    template <class T> const T* data() const { return reinterpret_cast<const T*>(ptr); }
    size_t size_bytes() const { return bytes; }
};

// ============================ NativeIndex ============================
struct NativeIndex {
    MmapBlob doc_vecs, doc_l0_neighbors, doc_l0_offsets, doc_levels;
    int32_t  doc_entry_point = 0;
    int64_t  doc_ntotal = 0;
    int32_t  dim = 0;

    MmapBlob iq_vecs, iq_l0_neighbors, iq_l0_offsets, iq_levels;
    MmapBlob iq_vecs_f16;                     // v3: fp16 copy used by router kernel
    int32_t  iq_entry_point = 0;
    int64_t  iq_ntotal = 0;
    int32_t  iq_dim = 0;

    MmapBlob pca_mean, pca_components_T;
    int32_t  pca_dim = 0;

    MmapBlob ep_ids;
    int64_t  ep_n = 0;
    int32_t  ep_width = 0;

    NativeIndex(const std::string& export_dir) {
        std::string d = export_dir + "/";
        doc_vecs.open(d + "doc_vecs.f32");
        doc_l0_neighbors.open(d + "doc_level0_neighbors.i32");
        doc_l0_offsets.open(d + "doc_level0_offsets.u64");
        doc_levels.open(d + "doc_levels.i32");
        { MmapBlob ep(d + "doc_entry_point.i32"); doc_entry_point = *ep.data<int32_t>(); }
        dim = 1024;
        doc_ntotal = doc_vecs.size_bytes() / (sizeof(float) * dim);

        iq_vecs.open(d + "iq_vecs.f32");
        iq_vecs_f16.open(d + "iq_vecs.f16");   // v3: half-precision router source
        iq_l0_neighbors.open(d + "iq_level0_neighbors.i32");
        iq_l0_offsets.open(d + "iq_level0_offsets.u64");
        iq_levels.open(d + "iq_levels.i32");
        { MmapBlob ep(d + "iq_entry_point.i32"); iq_entry_point = *ep.data<int32_t>(); }
        pca_dim = 256;
        iq_dim = 256;
        iq_ntotal = iq_vecs.size_bytes() / (sizeof(float) * iq_dim);

        pca_mean.open(d + "pca_mean.f32");
        pca_components_T.open(d + "pca_components_T.f32");

        ep_ids.open(d + "ep_ids.i32");
        ep_width = 10;
        ep_n = ep_ids.size_bytes() / (sizeof(int32_t) * ep_width);
    }

    const float* doc_vec(int64_t i) const { return doc_vecs.data<float>() + i * dim; }
    const int32_t* doc_neighbors(int64_t i) const {
        const uint64_t* offs = doc_l0_offsets.data<uint64_t>();
        return doc_l0_neighbors.data<int32_t>() + offs[i];
    }
    size_t doc_ndeg(int64_t i) const {
        const uint64_t* offs = doc_l0_offsets.data<uint64_t>();
        return static_cast<size_t>(offs[i + 1] - offs[i]);
    }
    const float* iq_vec(int64_t i) const { return iq_vecs.data<float>() + i * iq_dim; }
    const uint16_t* iq_vec_f16(int64_t i) const { return iq_vecs_f16.data<uint16_t>() + i * iq_dim; }
    const int32_t* iq_neighbors(int64_t i) const {
        const uint64_t* offs = iq_l0_offsets.data<uint64_t>();
        return iq_l0_neighbors.data<int32_t>() + offs[i];
    }
    size_t iq_ndeg(int64_t i) const {
        const uint64_t* offs = iq_l0_offsets.data<uint64_t>();
        return static_cast<size_t>(offs[i + 1] - offs[i]);
    }
    const int32_t* ep_row(int64_t i) const { return ep_ids.data<int32_t>() + i * ep_width; }
};

// ============================ Fixed-capacity heap over preallocated arrays ============================
// Score IDs stored in Struct-of-Arrays form so heap ops on primitives are cheap.
// max_heap: parent >= children (top = best score, for the candidate frontier)
// min_heap: parent <= children (top = worst-in-topef, for the results set)
struct SoAHeap {
    float*   score = nullptr;
    int32_t* id    = nullptr;
    int      n     = 0;
    int      cap   = 0;

    inline void reset() { n = 0; }
    inline bool empty() const { return n == 0; }
    inline int  size()  const { return n; }
    inline float top_score() const { return score[0]; }
    inline int32_t top_id() const { return id[0]; }

    // Max-heap: sift up on push, sift down on pop.
    inline void push_max(float s, int32_t i) {
        int k = n++;
        score[k] = s; id[k] = i;
        while (k > 0) {
            int p = (k - 1) >> 1;
            if (score[p] >= score[k]) break;
            std::swap(score[p], score[k]);
            std::swap(id[p], id[k]);
            k = p;
        }
    }
    inline void pop_max() {
        --n;
        if (n == 0) return;
        score[0] = score[n]; id[0] = id[n];
        int k = 0;
        while (true) {
            int l = 2*k + 1, r = 2*k + 2, m = k;
            if (l < n && score[l] > score[m]) m = l;
            if (r < n && score[r] > score[m]) m = r;
            if (m == k) break;
            std::swap(score[k], score[m]);
            std::swap(id[k], id[m]);
            k = m;
        }
    }

    // Min-heap: sift up on push, sift down on pop.
    inline void push_min(float s, int32_t i) {
        int k = n++;
        score[k] = s; id[k] = i;
        while (k > 0) {
            int p = (k - 1) >> 1;
            if (score[p] <= score[k]) break;
            std::swap(score[p], score[k]);
            std::swap(id[p], id[k]);
            k = p;
        }
    }
    inline void pop_min() {
        --n;
        if (n == 0) return;
        score[0] = score[n]; id[0] = id[n];
        int k = 0;
        while (true) {
            int l = 2*k + 1, r = 2*k + 2, m = k;
            if (l < n && score[l] < score[m]) m = l;
            if (r < n && score[r] < score[m]) m = r;
            if (m == k) break;
            std::swap(score[k], score[m]);
            std::swap(id[k], id[m]);
            k = m;
        }
    }
    // Replace top of min-heap with (s, i) and sift down.
    inline void replace_top_min(float s, int32_t i) {
        score[0] = s; id[0] = i;
        int k = 0;
        while (true) {
            int l = 2*k + 1, r = 2*k + 2, m = k;
            if (l < n && score[l] < score[m]) m = l;
            if (r < n && score[r] < score[m]) m = r;
            if (m == k) break;
            std::swap(score[k], score[m]);
            std::swap(id[k], id[m]);
            k = m;
        }
    }
};

// ============================ QueryContext (thread-local) ============================
struct QueryContext {
    // Generation-stamped visited arrays
    std::vector<uint32_t> visited_doc;   uint32_t gen_doc = 0;
    std::vector<uint32_t> visited_iq;    uint32_t gen_iq  = 0;

    // PCA scratch
    std::vector<float>   pca_out;        // pca_dim
    std::vector<float>   pca_diff;       // dim

    // Router output buffers
    std::vector<int32_t> hist_ids_buf;   // capacity kp_max
    std::vector<float>   hist_scores_buf;

    // Union seed buffers
    std::vector<int32_t> seed_buf;       // capacity kp_max * kep_max
    std::vector<float>   seed_score_buf;

    // Frontier and result heaps (SoA).
    // The `res` heap is bounded by ef (<=512).  The `cand` frontier is only bounded by
    // the number of unique nodes visited, which can reach several thousand for larger
    // ef values on the 500k / 800k graphs.  We size the arena at 65536 so the cand
    // heap cannot overflow even in pathological cases.
    static constexpr int EF_MAX = 512;
    static constexpr int CAND_CAP = 65536;
    static constexpr int RES_CAP  = EF_MAX + 128;
    static constexpr int HEAP_CAP = CAND_CAP; // legacy alias used at heap-construction sites

    std::vector<float>   frontier_score_doc;
    std::vector<int32_t> frontier_id_doc;
    std::vector<float>   results_score_doc;
    std::vector<int32_t> results_id_doc;

    std::vector<float>   frontier_score_iq;
    std::vector<int32_t> frontier_id_iq;
    std::vector<float>   results_score_iq;
    std::vector<int32_t> results_id_iq;

    // Extract scratch (drain min-heap for partial_sort)
    std::vector<float>   extract_score;
    std::vector<int32_t> extract_id;

    void init(int64_t doc_ntotal, int64_t iq_ntotal, int32_t pca_dim, int32_t dim) {
        visited_doc.assign(doc_ntotal, 0);
        visited_iq .assign(iq_ntotal,  0);
        pca_out .assign(pca_dim, 0.f);
        pca_diff.assign(dim,     0.f);
        // Provision for kp<=32, kep<=16 which are far above documented settings.
        hist_ids_buf   .assign(64,      0);
        hist_scores_buf.assign(64,      0.f);
        seed_buf       .assign(64 * 32, 0);
        seed_score_buf .assign(64 * 32, 0.f);
        frontier_score_doc.assign(CAND_CAP, 0.f); frontier_id_doc.assign(CAND_CAP, 0);
        results_score_doc .assign(RES_CAP,  0.f); results_id_doc .assign(RES_CAP,  0);
        frontier_score_iq .assign(CAND_CAP, 0.f); frontier_id_iq .assign(CAND_CAP, 0);
        results_score_iq  .assign(RES_CAP,  0.f); results_id_iq  .assign(RES_CAP,  0);
        extract_score.assign(RES_CAP, 0.f); extract_id.assign(RES_CAP, 0);
    }
    inline uint32_t bump_doc() {
        if (++gen_doc == 0) { std::fill(visited_doc.begin(), visited_doc.end(), 0); gen_doc = 1; }
        return gen_doc;
    }
    inline uint32_t bump_iq() {
        if (++gen_iq == 0) { std::fill(visited_iq.begin(), visited_iq.end(), 0); gen_iq = 1; }
        return gen_iq;
    }
};

// ============================ Bounded top-K extract ============================
static inline void extract_topk_min_heap(SoAHeap& results,
                                          std::vector<float>& tmp_score,
                                          std::vector<int32_t>& tmp_id,
                                          int32_t* out_ids, float* out_scores, int topk)
{
    int n = results.size();
    for (int i = 0; i < n; ++i) { tmp_score[i] = results.score[i]; tmp_id[i] = results.id[i]; }
    // We want the top-`topk` by descending score.  Use partial_sort of indices.
    int m = std::min(n, topk);
    // Build index array 0..n-1, partial-sort by score desc.
    static thread_local std::vector<int> idx;
    if ((int)idx.size() < n) idx.resize(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + m, idx.begin() + n,
                       [&](int a, int b) { return tmp_score[a] > tmp_score[b]; });
    for (int i = 0; i < m; ++i) { out_ids[i] = tmp_id[idx[i]]; out_scores[i] = tmp_score[idx[i]]; }
    for (int i = m; i < topk; ++i) { out_ids[i] = -1; out_scores[i] = -1e9f; }
}

// ============================ Seeded doc beam (zero-alloc) ============================
static void seeded_beam_search_doc_v2(const NativeIndex& idx, QueryContext& qctx,
                                       const float* q,
                                       const int32_t* seeds, const float* seed_scores, int n_seeds,
                                       int ef, int topk,
                                       int32_t* out_ids, float* out_scores)
{
    uint32_t gen = qctx.bump_doc();
    uint32_t* vis = qctx.visited_doc.data();

    SoAHeap cand{qctx.frontier_score_doc.data(), qctx.frontier_id_doc.data(), 0, QueryContext::CAND_CAP};
    SoAHeap res {qctx.results_score_doc.data(),  qctx.results_id_doc.data(),  0, QueryContext::RES_CAP};

    // Seed insertion — keep only unique seeds (caller already deduped in union)
    for (int i = 0; i < n_seeds; ++i) {
        int32_t s = seeds[i];
        if (s < 0 || s >= idx.doc_ntotal) continue;
        if (vis[s] == gen) continue;
        vis[s] = gen;
        float sc = seed_scores[i];
        cand.push_max(sc, s);
        if (res.size() < ef) {
            res.push_min(sc, s);
        } else if (sc > res.top_score()) {
            res.replace_top_min(sc, s);
        }
    }
    float lower_bound = res.empty() ? -1e30f : res.top_score();

    while (!cand.empty()) {
        float cs = cand.top_score();
        int32_t cid = cand.top_id();
        cand.pop_max();
        if (cs < lower_bound) break;

        const int32_t* neigh = idx.doc_neighbors(cid);
        size_t nn = idx.doc_ndeg(cid);
        // Deep prefetch — 8 neighbors ahead
        for (size_t k = 0; k < nn && k < 8; ++k)
            __builtin_prefetch(idx.doc_vec(neigh[k]), 0, 1);

        for (size_t k = 0; k < nn; ++k) {
            int32_t nid = neigh[k];
            if (nid < 0) continue;
            if (vis[nid] == gen) continue;
            vis[nid] = gen;
            if (k + 8 < nn) __builtin_prefetch(idx.doc_vec(neigh[k + 8]), 0, 1);
            float sc = dot_f32_avx2_4acc(q, idx.doc_vec(nid), idx.dim);
            if (res.size() < ef) {
                cand.push_max(sc, nid);
                res.push_min(sc, nid);
                lower_bound = res.top_score();
            } else if (sc > lower_bound) {
                cand.push_max(sc, nid);
                res.replace_top_min(sc, nid);
                lower_bound = res.top_score();
            }
        }
    }
    extract_topk_min_heap(res, qctx.extract_score, qctx.extract_id, out_ids, out_scores, topk);
}

// Level-0 baseline HNSW search on doc index (single-source from entry point).
static void baseline_search_doc_v2(const NativeIndex& idx, QueryContext& qctx,
                                    const float* q, int ef, int topk,
                                    int32_t* out_ids, float* out_scores)
{
    int32_t ep = idx.doc_entry_point;
    float ep_score = dot_f32_avx2_4acc(q, idx.doc_vec(ep), idx.dim);
    int32_t seed[1] = { ep };
    float   scr [1] = { ep_score };
    seeded_beam_search_doc_v2(idx, qctx, q, seed, scr, 1, ef, topk, out_ids, out_scores);
}

// I_Q router: seeded beam from entry point, small ef.  Zero-alloc.
static void seeded_beam_search_iq_v2(const NativeIndex& idx, QueryContext& qctx,
                                      const float* qp,
                                      int ef, int topk,
                                      int32_t* out_ids, float* out_scores)
{
    uint32_t gen = qctx.bump_iq();
    uint32_t* vis = qctx.visited_iq.data();

    SoAHeap cand{qctx.frontier_score_iq.data(), qctx.frontier_id_iq.data(), 0, QueryContext::CAND_CAP};
    SoAHeap res {qctx.results_score_iq.data(),  qctx.results_id_iq.data(),  0, QueryContext::RES_CAP};

    int32_t ep = idx.iq_entry_point;
    float ep_score = dot_f16_f32_avx2(idx.iq_vec_f16(ep), qp, idx.iq_dim);
    vis[ep] = gen;
    cand.push_max(ep_score, ep);
    res.push_min(ep_score, ep);
    float lower_bound = ep_score;

    while (!cand.empty()) {
        float cs = cand.top_score();
        int32_t cid = cand.top_id();
        cand.pop_max();
        if (cs < lower_bound && res.size() >= ef) break;

        const int32_t* neigh = idx.iq_neighbors(cid);
        size_t nn = idx.iq_ndeg(cid);
        // Prefetch fp16 vector rows (half the bytes = better line utilization).
        for (size_t k = 0; k < nn && k < 8; ++k)
            __builtin_prefetch(idx.iq_vec_f16(neigh[k]), 0, 1);

        for (size_t k = 0; k < nn; ++k) {
            int32_t nid = neigh[k];
            if (nid < 0) continue;
            if (vis[nid] == gen) continue;
            vis[nid] = gen;
            if (k + 8 < nn) __builtin_prefetch(idx.iq_vec_f16(neigh[k + 8]), 0, 1);
            float sc = dot_f16_f32_avx2(idx.iq_vec_f16(nid), qp, idx.iq_dim);
            if (res.size() < ef) {
                cand.push_max(sc, nid);
                res.push_min(sc, nid);
                lower_bound = res.top_score();
            } else if (sc > lower_bound) {
                cand.push_max(sc, nid);
                res.replace_top_min(sc, nid);
                lower_bound = res.top_score();
            }
        }
    }
    // Use the doc-side extract scratch buffers — sizes fit either heap.
    extract_topk_min_heap(res, qctx.extract_score, qctx.extract_id, out_ids, out_scores, topk);
}

// ============================ PCA (fused, zero-alloc) ============================
static void pca_transform_v2(const NativeIndex& idx, QueryContext& qctx,
                              const float* q, float* qp_out)
{
    const float* mean = idx.pca_mean.data<float>();
    const float* pcT  = idx.pca_components_T.data<float>();
    const int dim     = idx.dim;
    const int pca_dim = idx.pca_dim;
    float*     diff   = qctx.pca_diff.data();

#if defined(__AVX2__)
    // Fused (q - mean) using AVX2, then outer-product accumulate.
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(q + i);
        __m256 m = _mm256_loadu_ps(mean + i);
        _mm256_storeu_ps(diff + i, _mm256_sub_ps(x, m));
    }
    for (; i < dim; ++i) diff[i] = q[i] - mean[i];
#else
    for (int i = 0; i < dim; ++i) diff[i] = q[i] - mean[i];
#endif

    std::memset(qp_out, 0, sizeof(float) * pca_dim);

#if defined(__AVX2__)
    // pcT is row-major (dim, pca_dim).  Broadcast diff[i], multiply into pcT row, accumulate.
    for (int i = 0; i < dim; ++i) {
        __m256 dv  = _mm256_set1_ps(diff[i]);
        const float* row = pcT + (size_t)i * pca_dim;
        // Prefetch next row into L1
        if (i + 2 < dim) __builtin_prefetch(pcT + (size_t)(i + 2) * pca_dim, 0, 3);
        int j = 0;
        for (; j + 32 <= pca_dim; j += 32) {
            __m256 r0 = _mm256_loadu_ps(row + j +  0);
            __m256 o0 = _mm256_loadu_ps(qp_out + j +  0);
            o0 = _mm256_fmadd_ps(dv, r0, o0);
            _mm256_storeu_ps(qp_out + j +  0, o0);
            __m256 r1 = _mm256_loadu_ps(row + j +  8);
            __m256 o1 = _mm256_loadu_ps(qp_out + j +  8);
            o1 = _mm256_fmadd_ps(dv, r1, o1);
            _mm256_storeu_ps(qp_out + j +  8, o1);
            __m256 r2 = _mm256_loadu_ps(row + j + 16);
            __m256 o2 = _mm256_loadu_ps(qp_out + j + 16);
            o2 = _mm256_fmadd_ps(dv, r2, o2);
            _mm256_storeu_ps(qp_out + j + 16, o2);
            __m256 r3 = _mm256_loadu_ps(row + j + 24);
            __m256 o3 = _mm256_loadu_ps(qp_out + j + 24);
            o3 = _mm256_fmadd_ps(dv, r3, o3);
            _mm256_storeu_ps(qp_out + j + 24, o3);
        }
        for (; j + 8 <= pca_dim; j += 8) {
            __m256 r = _mm256_loadu_ps(row + j);
            __m256 o = _mm256_loadu_ps(qp_out + j);
            o = _mm256_fmadd_ps(dv, r, o);
            _mm256_storeu_ps(qp_out + j, o);
        }
        for (; j < pca_dim; ++j) qp_out[j] += diff[i] * row[j];
    }
#else
    for (int i = 0; i < dim; ++i) {
        float d = diff[i];
        const float* row = pcT + (size_t)i * pca_dim;
        for (int j = 0; j < pca_dim; ++j) qp_out[j] += d * row[j];
    }
#endif
}

// ============================ Union C + scoring (zero-alloc) ============================
static void build_and_score_union_v2(const NativeIndex& idx, QueryContext& qctx,
                                      const float* q,
                                      const int32_t* hist_row_ids, int n_hist, int kep,
                                      int32_t* seed_out, float* score_out, int& n_unique)
{
    uint32_t gen = qctx.bump_doc();
    uint32_t* vis = qctx.visited_doc.data();
    n_unique = 0;
    for (int i = 0; i < n_hist; ++i) {
        int32_t hi = hist_row_ids[i];
        if (hi < 0 || hi >= idx.ep_n) continue;
        const int32_t* row = idx.ep_row(hi);
        // Prefetch next hist row's vectors (12 = kep for typical settings)
        if (i + 1 < n_hist) {
            int32_t nh = hist_row_ids[i + 1];
            if (nh >= 0 && nh < idx.ep_n) {
                const int32_t* nrow = idx.ep_row(nh);
                for (int j = 0; j < kep && j < 4; ++j) {
                    int32_t d = nrow[j];
                    if (d >= 0 && d < idx.doc_ntotal)
                        __builtin_prefetch(idx.doc_vec(d), 0, 1);
                }
            }
        }
        for (int j = 0; j < kep; ++j) {
            int32_t did = row[j];
            if (did < 0 || did >= idx.doc_ntotal) continue;
            if (vis[did] == gen) continue;
            vis[did] = gen;
            seed_out[n_unique] = did;
            score_out[n_unique] = dot_f32_avx2_4acc(q, idx.doc_vec(did), idx.dim);
            n_unique++;
        }
    }
}

// ============================ Adaptive ef ============================
static inline int adaptive_ef(float s, float s_max, float th, int ef_min, int ef_default) {
    if (s > s_max) return ef_min;
    float denom = s_max - th;
    if (denom <= 0.f) return ef_default;
    float ef_p = ef_min + (float)(ef_default - ef_min) * (s_max - s) / denom;
    if (ef_p < (float)ef_min) return ef_min;
    if (ef_p > (float)ef_default) return ef_default;
    return (int)std::lround(ef_p);
}

// ============================ Class ============================
class NativeQLR_V3 {
public:
    NativeQLR_V3(const std::string& export_dir)
      : idx_(export_dir)
    {
        qctx_.init(idx_.doc_ntotal, idx_.iq_ntotal, idx_.pca_dim, idx_.dim);
    }

    py::dict qlr(py::array_t<float, py::array::c_style | py::array::forcecast> q_arr,
                 int kp, int kep, float th, int ef_default, int ef_min, int router_ef,
                 float s_max, int topk)
    {
        auto qbuf = q_arr.request();
        if (qbuf.ndim != 1 && !(qbuf.ndim == 2 && qbuf.shape[0] == 1))
            throw std::runtime_error("q must be 1D or (1, dim)");
        const float* q = static_cast<const float*>(qbuf.ptr);

        auto t0 = std::chrono::high_resolution_clock::now();

        pca_transform_v2(idx_, qctx_, q, qctx_.pca_out.data());
        auto t_pca = std::chrono::high_resolution_clock::now();

        // Router uses preallocated hist buffers
        int32_t* hist_ids = qctx_.hist_ids_buf.data();
        float*   hist_scores = qctx_.hist_scores_buf.data();
        seeded_beam_search_iq_v2(idx_, qctx_, qctx_.pca_out.data(),
                                  router_ef, kp, hist_ids, hist_scores);
        auto t_router = std::chrono::high_resolution_clock::now();

        float s = hist_scores[0];
        int32_t out_ids[16];
        float   out_scores[16];
        int ef_used = 0;
        int n_unique = 0;
        int routed_flag = 0;
        auto t_union = t_router, t_beam = t_router, t_fb = t_router;

        if (s < th) {
            baseline_search_doc_v2(idx_, qctx_, q, ef_default, topk, out_ids, out_scores);
            t_fb = std::chrono::high_resolution_clock::now();
            ef_used = ef_default;
        } else {
            build_and_score_union_v2(idx_, qctx_, q, hist_ids, kp, kep,
                                      qctx_.seed_buf.data(), qctx_.seed_score_buf.data(), n_unique);
            t_union = std::chrono::high_resolution_clock::now();

            ef_used = adaptive_ef(s, s_max, th, ef_min, ef_default);

            seeded_beam_search_doc_v2(idx_, qctx_, q,
                                       qctx_.seed_buf.data(), qctx_.seed_score_buf.data(), n_unique,
                                       ef_used, topk, out_ids, out_scores);
            t_beam = std::chrono::high_resolution_clock::now();
            routed_flag = 1;
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        py::dict d;
        py::array_t<int32_t> ids_arr(topk);
        py::array_t<float>   scores_arr(topk);
        std::memcpy(ids_arr.mutable_data(), out_ids, sizeof(int32_t) * topk);
        std::memcpy(scores_arr.mutable_data(), out_scores, sizeof(float) * topk);
        d["ids"] = ids_arr;
        d["scores"] = scores_arr;
        d["total_us"] = std::chrono::duration<double, std::micro>(t_end - t0).count();
        d["pca_us"]   = std::chrono::duration<double, std::micro>(t_pca - t0).count();
        d["router_us"] = std::chrono::duration<double, std::micro>(t_router - t_pca).count();
        if (routed_flag) {
            d["union_us"] = std::chrono::duration<double, std::micro>(t_union - t_router).count();
            d["beam_us"]  = std::chrono::duration<double, std::micro>(t_beam - t_union).count();
            d["fallback_us"] = 0.0;
        } else {
            d["union_us"] = 0.0;
            d["beam_us"] = 0.0;
            d["fallback_us"] = std::chrono::duration<double, std::micro>(t_fb - t_router).count();
        }
        d["routed"] = routed_flag;
        d["n_seeds"] = n_unique;
        d["ef_used"] = ef_used;
        d["s_top1"] = s;
        return d;
    }

    py::dict baseline(py::array_t<float, py::array::c_style | py::array::forcecast> q_arr, int ef, int topk) {
        auto qbuf = q_arr.request();
        const float* q = static_cast<const float*>(qbuf.ptr);
        int32_t out_ids[16];
        float   out_scores[16];
        auto t0 = std::chrono::high_resolution_clock::now();
        baseline_search_doc_v2(idx_, qctx_, q, ef, topk, out_ids, out_scores);
        auto t1 = std::chrono::high_resolution_clock::now();
        py::array_t<int32_t> ids_arr(topk);
        py::array_t<float>   scores_arr(topk);
        std::memcpy(ids_arr.mutable_data(), out_ids, sizeof(int32_t) * topk);
        std::memcpy(scores_arr.mutable_data(), out_scores, sizeof(float) * topk);
        py::dict d;
        d["ids"] = ids_arr;
        d["scores"] = scores_arr;
        d["total_us"] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return d;
    }

    int64_t doc_ntotal() const { return idx_.doc_ntotal; }
    int64_t iq_ntotal()  const { return idx_.iq_ntotal;  }
    int32_t dim()        const { return idx_.dim; }
    int32_t doc_entry_point() const { return idx_.doc_entry_point; }

private:
    NativeIndex  idx_;
    QueryContext qctx_;
};

PYBIND11_MODULE(native_qlr_v3, m) {
    m.doc() = "native_qlr v3 — v2 + fp16 iq_vecs (F16C router kernel)";
    py::class_<NativeQLR_V3>(m, "NativeQLR")
      .def(py::init<const std::string&>(), py::arg("export_dir"))
      .def("qlr", &NativeQLR_V3::qlr, py::arg("q"), py::arg("kp"), py::arg("kep"),
                                       py::arg("th"), py::arg("ef_default"), py::arg("ef_min"),
                                       py::arg("router_ef"), py::arg("s_max"), py::arg("topk"))
      .def("baseline", &NativeQLR_V3::baseline, py::arg("q"), py::arg("ef"), py::arg("topk"))
      .def("doc_ntotal", &NativeQLR_V3::doc_ntotal)
      .def("iq_ntotal",  &NativeQLR_V3::iq_ntotal)
      .def("dim",        &NativeQLR_V3::dim)
      .def("doc_entry_point", &NativeQLR_V3::doc_entry_point);
}
