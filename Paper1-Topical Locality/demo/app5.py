"""
app2.py  - CAST2019 Demo Search
Run: streamlit run app2.py
"""

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import altair as alt
import streamlit as st
from search_demo5 import (
    DemoSearcher,
    NPROBE_VALUES,
    H_VALUES,
    EF_SEARCH_VALUES,
    NPROBE_DEFAULT,
    H_DEFAULT,
    EF_SEARCH_DEFAULT,
)

st.set_page_config(page_title="CAST2019 Demo Search", page_icon="🔍", layout="wide")
st.title("🔍 CAST2019 Demo Search")
st.caption("Conversational search over the CAST2019 demo collection")

@st.cache_resource(show_spinner="Loading indexes and text collection...")
def get_searcher():
    return DemoSearcher()

searcher = get_searcher()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    index_type = st.radio(
        "Index type",
        options=["hnsw", "ivf", "flat", "toploc"],
        format_func=lambda x: {
            "hnsw":   "HNSW (approximate, graph-based)",
            "ivf":    "IVF  (approximate, cluster-based)",
            "flat":   "Flat (exact search)",
            "toploc": "TopLoc IVF (conversational)",
        }[x],
    )

    top_k = st.slider("Top-K results", min_value=1, max_value=20, value=5)

    st.divider()

    # ef_search — for HNSW (and always shown since comparison tab uses HNSW)
    ef_search = st.select_slider(
        "ef-search — HNSW beam width",
        options=EF_SEARCH_VALUES,
        value=EF_SEARCH_DEFAULT,
        help="Controls beam width during HNSW graph traversal. "
             "Higher = more accurate but slower. Powers of 2 from 1 to 4096."
    )

    # nprobe — for IVF and TopLoc
    if index_type in ("ivf", "toploc"):
        nprobe = st.select_slider(
            "nprobe — IVF lists probed per query",
            options=NPROBE_VALUES,
            value=NPROBE_DEFAULT,
        )
    else:
        nprobe = NPROBE_DEFAULT

    # h — TopLoc only
    if index_type == "toploc":
        h = st.select_slider(
            "h — cached centroids per conversation",
            options=H_VALUES,
            value=H_DEFAULT,
        )
        st.divider()
        if st.button("Clear conversation cache"):
            searcher.clear_conv_cache()
            st.success("Cache cleared.")
        st.markdown(f"**Cached conversations:** {len(searcher._conv_cache)}")
        for cid in list(searcher._conv_cache.keys()):
            st.markdown(f"- `{cid}`")
    else:
        h = H_DEFAULT

    st.divider()
    st.markdown(f"**Collection:** {len(searcher.text_map):,} docs")
    st.markdown(f"**nlist (IVF):** {searcher.ivf_index.nlist}")
    st.markdown(f"**Topics with qrels:** {len(searcher.qrel_qids):,}")
    model_ok = searcher.model_available
    st.markdown(f"**Free-text:** {'available' if model_ok else 'model not installed'}")


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_topic, tab_conv, tab_free = st.tabs([
    "📋 Topic query",
    "📊 Conversation comparison",
    "✏️ Free-text query",
])

# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Topic query
# ══════════════════════════════════════════════════════════════════════════
with tab_topic:

    topic_options = {
        qid: f"{qid} — {text}"
        for qid, text in sorted(searcher.topics.items())
        if qid in searcher.qrel_qids
    }
    selected_qid = st.selectbox(
        "Select a topic query",
        options=list(topic_options.keys()),
        format_func=lambda qid: topic_options[qid],
    )

    # Conversation context for TopLoc
    if index_type == "toploc" and selected_qid:
        parts    = selected_qid.rsplit("_", 1)
        conv_id  = parts[0]
        turn     = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
        in_cache = conv_id in searcher._conv_cache
        h_eff    = min(h, searcher.ivf_index.nlist)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Conversation", conv_id)
        col_b.metric("Turn", turn)

        if turn == 1 or not in_cache:
            col_c.metric("Mode", "First turn")
            st.info(
                f"**First turn** — searches all {searcher.ivf_index.nlist} centroids, "
                f"caches top **{h_eff}** for conversation `{conv_id}`, "
                f"probes **{min(nprobe, h_eff)}** lists."
            )
        else:
            col_c.metric("Mode", "Sub-turn")
            st.success(
                f"**Sub-turn** — centroid selection over **{h_eff}** cached centroids "
                f"instead of all {searcher.ivf_index.nlist} "
                f"(**{searcher.ivf_index.nlist / h_eff:.1f}x fewer**), "
                f"probes **{min(nprobe, h_eff)}** lists."
            )

    if st.button("Search", key="btn_topic", type="primary"):
        if selected_qid not in searcher.query_embs:
            st.error(f"No embedding found for **{selected_qid}**.")

        elif index_type == "toploc":
            results, latency_ms, query_text, is_first_turn, nprobe_used, h_used = \
                searcher.search_toploc_by_topic(selected_qid, h=h, nprobe=nprobe, top_k=top_k)
            turn_label = "First turn" if is_first_turn else "Sub-turn"
            cap_note   = f" (h capped to {h_used} = nlist)" if h_used < h else ""
            st.success(
                f"**{len(results)} results** for *\"{query_text}\"* | "
                f"{latency_ms:.2f} ms | {turn_label} | nprobe: {nprobe_used} | h: {h_used}{cap_note}"
            )
            for r in results:
                with st.expander(f"#{r.rank}  {r.doc_id}  score: {r.score:.4f}", expanded=True):
                    st.markdown(r.text)

        else:
            results, latency_ms, query_text = searcher.search_by_topic(
                selected_qid, index_type=index_type, top_k=top_k,
                nprobe=nprobe, ef_search=ef_search
            )
            param_note = ""
            if index_type == "ivf":
                param_note = f", nprobe={nprobe}"
            elif index_type == "hnsw":
                param_note = f", ef_search={ef_search}"
            st.success(
                f"**{len(results)} results** for *\"{query_text}\"* | "
                f"{latency_ms:.2f} ms ({index_type.upper()}{param_note})"
            )
            for r in results:
                with st.expander(f"#{r.rank}  {r.doc_id}  score: {r.score:.4f}", expanded=True):
                    st.markdown(r.text)

# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Conversation comparison
# ══════════════════════════════════════════════════════════════════════════
with tab_conv:
    st.subheader("Flat vs HNSW vs Plain IVF vs TopLoc IVF — per-turn latency")
    st.caption(
        "Runs all turns of a conversation through all four methods. "
        "Turn 1 is slightly slower for TopLoc (cache build overhead). "
        "Sub-turns should be faster than plain IVF (centroid selection over h << nlist)."
    )

    conv_ids = sorted(set(
        qid.rsplit("_", 1)[0]
        for qid in searcher.qrel_qids
        if "_" in qid and qid in searcher.query_embs
    ))

    col_left, col_right = st.columns([2, 1])
    with col_left:
        selected_conv = st.selectbox("Select conversation", conv_ids, key="conv_select")
    with col_right:
        n_runs = st.slider("Runs to average over", min_value=1, max_value=20, value=5)

    preview_turns = searcher.get_conversation_turns(selected_conv)
    if preview_turns:
        st.markdown(f"**{len(preview_turns)} turns** in conversation `{selected_conv}`:")
        for qid in preview_turns:
            turn = qid.rsplit("_", 1)[1]
            text = searcher.topics.get(qid, "")
            st.markdown(f"- Turn {turn}: *{text}*")
    else:
        st.warning("No turns found for this conversation.")

    h_eff = min(h, searcher.ivf_index.nlist)
    st.markdown(
        f"Settings: nprobe = **{nprobe}**, h = **{h_eff}**, "
        f"ef-search = **{ef_search}**, nlist = **{searcher.ivf_index.nlist}**"
    )

    if st.button("Run comparison", key="btn_conv", type="primary"):
        with st.spinner(f"Running {n_runs} x {len(preview_turns)} turns x 4 methods..."):
            rows = searcher.run_conversation_comparison(
                selected_conv, nprobe=nprobe, h=h,
                ef_search=ef_search, top_k=top_k, n_runs=n_runs
            )

        if not rows:
            st.warning("No results.")
        else:
            total_flat   = sum(r["flat_ms"]   for r in rows)
            total_hnsw   = sum(r["hnsw_ms"]   for r in rows)
            total_ivf    = sum(r["ivf_ms"]    for r in rows)
            total_toploc = sum(r["toploc_ms"] for r in rows)

            sub_rows   = [r for r in rows if not r["is_first"]]
            sub_ivf    = sum(r["ivf_ms"]    for r in sub_rows)
            sub_toploc = sum(r["toploc_ms"] for r in sub_rows)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Flat (ms)",   f"{total_flat:.2f}")
            m2.metric("Total HNSW (ms)",   f"{total_hnsw:.2f}",
                      help=f"ef-search={ef_search}")
            m3.metric("Total IVF (ms)",    f"{total_ivf:.2f}",
                      help=f"nprobe={nprobe}")

            toploc_delta = None
            if sub_rows and sub_toploc > 0:
                pct = (sub_ivf - sub_toploc) / sub_ivf * 100
                toploc_delta = (
                    f"Sub-turns {pct:.1f}% faster than IVF"
                    if pct > 0
                    else f"Sub-turns {abs(pct):.1f}% slower than IVF"
                )
            m4.metric("Total TopLoc (ms)", f"{total_toploc:.2f}", delta=toploc_delta,
                      help=f"nprobe={nprobe}, h={h_eff}")

            # ── Grouped bar chart ─────────────────────────────────────────
            METHOD_ORDER  = ["Flat", "HNSW", "Plain IVF", "TopLoc"]
            METHOD_COLORS = ["#54A24B", "#72B7B2", "#4C78A8", "#F58518"]

            chart_data = []
            for r in rows:
                chart_data.append({"Turn": str(r["turn"]), "Method": "Flat",      "Latency (ms)": r["flat_ms"]})
                chart_data.append({"Turn": str(r["turn"]), "Method": "HNSW",      "Latency (ms)": r["hnsw_ms"]})
                chart_data.append({"Turn": str(r["turn"]), "Method": "Plain IVF", "Latency (ms)": r["ivf_ms"]})
                chart_data.append({"Turn": str(r["turn"]), "Method": "TopLoc",    "Latency (ms)": r["toploc_ms"]})

            chart = (
                alt.Chart(pd.DataFrame(chart_data))
                .mark_bar()
                .encode(
                    x       = alt.X("Turn:O",
                                    sort=[str(r["turn"]) for r in rows],
                                    axis=alt.Axis(labelAngle=0, title="Turn")),
                    y       = alt.Y("Latency (ms):Q", title="Latency (ms)"),
                    color   = alt.Color("Method:N",
                                        sort=METHOD_ORDER,
                                        scale=alt.Scale(domain=METHOD_ORDER, range=METHOD_COLORS),
                                        legend=alt.Legend(orient="bottom")),
                    xOffset = alt.XOffset("Method:N", sort=METHOD_ORDER),
                    tooltip = ["Turn:O", "Method:N",
                               alt.Tooltip("Latency (ms):Q", format=".3f")],
                )
                .properties(
                    height=400,
                    title=f"Latency per turn (ms) — ef-search={ef_search}, nprobe={nprobe}, h={h_eff}"
                )
            )

            st.altair_chart(chart, use_container_width=True)

            # ── Per-turn table ────────────────────────────────────────────
            st.subheader("Per-turn breakdown")
            table_df = pd.DataFrame([{
                "Turn":           r["turn"],
                "Query":          r["query_text"],
                "Flat (ms)":      f"{r['flat_ms']:.3f}",
                "HNSW (ms)":      f"{r['hnsw_ms']:.3f}",
                "Plain IVF (ms)": f"{r['ivf_ms']:.3f}",
                "TopLoc (ms)":    f"{r['toploc_ms']:.3f}",
                "IVF/TopLoc":     f"{r['speedup']:.2f}x",
                "Type":           "First turn" if r["is_first"] else "Sub-turn",
            } for r in rows])
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.info(
                f"**What you're seeing:** Sub-turn centroid selection searches **{h_eff}** cached "
                f"centroids instead of all **{searcher.ivf_index.nlist}** "
                f"({searcher.ivf_index.nlist / h_eff:.1f}x fewer comparisons). "
                f"On the full 34M index (nlist=32768), h={h_eff} gives a "
                f"{32768 // h_eff}x reduction in centroid comparisons per sub-turn."
            )

# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Free-text query
# ══════════════════════════════════════════════════════════════════════════
with tab_free:
    if not searcher.model_available:
        st.warning(
            "Free-text search unavailable — sentence-transformers not installed.\n\n"
            "Run: pip install sentence-transformers"
        )
    else:
        if index_type == "toploc":
            st.info("Free-text mode uses standard IVF — TopLoc requires a conversation query ID.")

        free_text = st.text_input("Enter your query",
                                  placeholder="e.g. What are the symptoms of lung cancer?")
        if st.button("Search", key="btn_free", type="primary"):
            if not free_text.strip():
                st.warning("Please enter a query.")
            else:
                idx_for_free = "ivf" if index_type == "toploc" else index_type
                with st.spinner("Embedding and searching..."):
                    results, latency_ms = searcher.search_free_text(
                        free_text.strip(), index_type=idx_for_free,
                        top_k=top_k, nprobe=nprobe, ef_search=ef_search
                    )
                param_note = ""
                if idx_for_free == "ivf":
                    param_note = f", nprobe={nprobe}"
                elif idx_for_free == "hnsw":
                    param_note = f", ef_search={ef_search}"
                st.success(
                    f"**{len(results)} results** for *\"{free_text.strip()}\"* | "
                    f"{latency_ms:.2f} ms ({idx_for_free.upper()}{param_note})"
                )
                for r in results:
                    with st.expander(f"#{r.rank}  {r.doc_id}  score: {r.score:.4f}", expanded=True):
                        st.markdown(r.text)