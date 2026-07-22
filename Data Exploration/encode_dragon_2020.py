"""
Encode TREC CAsT 2020 queries with the DRAGON+ query encoder.

Matches the CAsT 2019 Dragon recipe exactly:
  * model   : facebook/dragon-plus-query-encoder
  * pooling : CLS token  ->  last_hidden_state[:, 0, :]
  * prefix  : NONE (Dragon takes the raw query text)
  * norm    : NONE -- raw embeddings (Dragon is a dot-product model; the
              flat-IP / MIPS-HNSW pipeline ranks by raw <q, x>)
Queries come from ir_datasets, using the MANUAL rewrites (as the paper does).

Output columns: id (str), query (str), embedding (list[float])  -- 768-dim.
"""

import ir_datasets
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME  = "facebook/dragon-plus-query-encoder"
IR_DATASET  = "trec-cast/v1/2020/judged"
OUTPUT_PARQUET = "topics_dragon_embeddings_2020.parquet"
BATCH_SIZE  = 64
MAX_LENGTH  = 512

# -------------------------
# Load queries (manual rewrites preferred, as in the paper)
# -------------------------
dataset = ir_datasets.load(IR_DATASET)

def pick_text(q):
    return (
        q.manual_rewritten_utterance
        or q.automatic_rewritten_utterance
        or q.raw_utterance
    )

rows = []
for q in dataset.queries_iter():
    text = pick_text(q).replace("\n", " ").strip()
    rows.append((q.query_id, text))
print(f"Loaded {len(rows)} queries from {IR_DATASET}")

# -------------------------
# Load model (CLS pooling, no pooling layer)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------
# Batch embedding
# -------------------------
results = []
with torch.no_grad():
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ids   = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        inputs = tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=MAX_LENGTH,
        ).to(device)

        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]     # CLS token; NO normalization
        emb = emb.cpu().numpy().astype(np.float32)

        for j in range(len(ids)):
            results.append({
                "id":        ids[j],                  # column matches the search loader
                "query":     texts[j],
                "embedding": emb[j].tolist(),
            })
        print(f"  encoded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

# -------------------------
# Save
# -------------------------
df = pd.DataFrame(results)
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Wrote {len(df)} query embeddings (dim={len(results[0]['embedding'])}) "
      f"to {OUTPUT_PARQUET}")