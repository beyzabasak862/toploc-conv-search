"""
Diagnose which DRAGON+ encoder produced the corpus index.

Logic:
  DRAGON+ is asymmetric. A corpus index SHOULD be built with the
  context-encoder. If it was mistakenly built with the query-encoder,
  query/passage spaces are mismatched and retrieval collapses.

  We can't read the encoder name off the vectors, but we can test
  empirically: encode the SAME queries with BOTH encoders, run each
  against the index, and see which one retrieves sensibly. The encoder
  whose queries land on-topic is the SAME family the index was built
  with -> tells us what the index actually is.

  Additionally we do a self-similarity check: reconstruct a few passage
  vectors from the index and check their norm (normalized ~1.0) and the
  score distribution, to catch normalization/dim issues.

Run on Pegasus inside mein_env.
"""

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INDEX_PATH = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_dragon_ivf.index"

# A few unambiguous test queries. Each should retrieve clearly on-topic
# passages if the encoder matches the index.
TEST_QUERIES = [
    "What is throat cancer?",
    "What are the different types of sharks?",
    "Tell me about the Bronze Age collapse.",
]

# Candidate QUERY encoders across the DRAGON family. The index was built
# with SOME context encoder; the matching query encoder is the one whose
# queries retrieve sensibly. RoBERTa vs plus are mutually incompatible,
# so only the correct family will produce high, on-topic scores.
CANDIDATE_QUERY_ENCODERS = [
    "facebook/dragon-plus-query-encoder",      # DRAGON+ (RetroMAE init)
    "facebook/dragon-roberta-query-encoder",   # DRAGON-RoBERTa (RoBERTa init)
]


@torch.no_grad()
def encode(texts, model_name, normalize):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    inputs = tok(texts, padding=True, truncation=True, return_tensors="pt")
    emb = model(**inputs).last_hidden_state[:, 0, :].numpy().astype("float32")
    if normalize:
        faiss.normalize_L2(emb)
    return emb


def main():
    print(f"Loading index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    index.nprobe = 64
    print(f"  ntotal={index.ntotal:,} nlist={index.nlist} d={index.d}")

    # --- passage norm sanity: reconstruct a few vectors from the index ---
    try:
        index.make_direct_map()
        sample = np.vstack([index.reconstruct(int(i)) for i in range(5)])
        norms = np.linalg.norm(sample, axis=1)
        print(f"\nSample passage-vector norms (first 5): {norms.round(4)}")
        print("  -> ~1.0 means index stores L2-normalized vectors (cosine).")
    except Exception as e:
        print(f"  (could not reconstruct sample vectors: {e})")

    # --- the main test: sweep candidate query encoders ---
    # Index is normalized (confirmed), so normalize queries to match.
    normalize = True
    summary = {}
    for model_name in CANDIDATE_QUERY_ENCODERS:
        print(f"\n{'='*60}\nqueries encoded with: {model_name}\n{'='*60}")
        try:
            q = encode(TEST_QUERIES, model_name, normalize)
        except Exception as e:
            print(f"  could not load/encode ({e}); skipping")
            continue
        D, I = index.search(q, 5)
        for qi, query in enumerate(TEST_QUERIES):
            print(f"  Q: {query}")
            print(f"     top-5 scores: {D[qi].round(3)}")
            print(f"     top-5 docids: {I[qi]}")
        mean_top1 = float(D[:, 0].mean())
        summary[model_name] = mean_top1
        print(f"  --> mean top-1 score: {mean_top1:.4f}")

    print("\n" + "="*60)
    print("SUMMARY (mean top-1 cosine, higher = better match to index):")
    for m, s in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {s:.4f}   {m}")
    print("\nINTERPRETATION:")
    print("  The query encoder with the CLEARLY HIGHEST mean top-1 score")
    print("  (and on-topic docids) is the family the index was built with.")
    print("  A good match usually lands well above ~0.5 cosine; a mismatch")
    print("  sits low and flat with scrambled docids across all 3 queries.")
    print("  Whichever wins here is the query encoder you must use in the")
    print("  encode step -- if it's NOT dragon-plus, that mismatch is why")
    print("  your metrics were low.")
    print("="*60)


if __name__ == "__main__":
    main()
