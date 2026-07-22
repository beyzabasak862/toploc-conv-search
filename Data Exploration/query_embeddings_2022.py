"""
Encode TREC CAsT 2020 queries with Snowflake arctic-embed-l-v2.0,
matching the embedding recipe used in search_demo3.py exactly:
  * model   : Snowflake/snowflake-arctic-embed-l-v2.0 (add_pooling_layer=False)
  * prefix  : "query: "
  * pooling : CLS token  ->  out[0][:, 0]      (NOT mean pooling)
  * norm    : L2-normalized
  * max_len : 8192
This is the SAME model that embedded the passages, so query and document
vectors share one space. (arctic-embed-m + mean pooling would be a mismatch.)
"""

import ir_datasets
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config (mirror the demo)
# -------------------------
MODEL_LOCAL_PATH = "/home/toploc1/Datasets/toploc1/models/snowflake-arctic-embed-l-v2.0"
MODEL_HF_NAME    = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_PREFIX     = "query: "
MAX_LENGTH       = 8192
BATCH_SIZE       = 32
OUTPUT_PARQUET   = "cast2020_query_embeddings.parquet"
TOPICS_TSV       = "cast2020_queries.tsv"

# -------------------------
# Load dataset
# -------------------------
dataset = ir_datasets.load("trec-cast/v1/2020/judged")

def pick_text(q):
    # paper uses the MANUAL rewrites; fall back only if missing
    return (
        q.manual_rewritten_utterance
        or q.automatic_rewritten_utterance
        or q.raw_utterance
    )

# write a plain topics tsv for reference / the demo loader
with open(TOPICS_TSV, "w", encoding="utf-8") as f:
    f.write("query_id\tquery_text\n")
    for q in dataset.queries_iter():
        text = pick_text(q).replace("\n", " ").strip()
        f.write(f"{q.query_id}\t{text}\n")

# -------------------------
# Load Snowflake v2.0 model (local mirror if present, else HF)
# -------------------------
import os
source = MODEL_LOCAL_PATH if os.path.exists(MODEL_LOCAL_PATH) else MODEL_HF_NAME

tokenizer = AutoTokenizer.from_pretrained(source)
model     = AutoModel.from_pretrained(source, add_pooling_layer=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------
# Collect queries
# -------------------------
rows = []
for q in dataset.queries_iter():
    text = pick_text(q).replace("\n", " ").strip()
    rows.append((q.query_id, text))

# -------------------------
# Batch embedding (CLS pooling, exactly like the demo's embed())
# -------------------------
results = []
with torch.no_grad():
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ids   = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        inputs = tokenizer(
            [QUERY_PREFIX + t for t in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        ).to(device)

        out = model(**inputs)
        emb = out[0][:, 0]                    # CLS token  <-- v2.0 pooling
        emb = F.normalize(emb, p=2, dim=1)
        emb = emb.cpu().numpy()

        for j in range(len(ids)):
            results.append({
                "id":        ids[j],           # column name matches the search loader
                "text":      texts[j],
                "embedding": emb[j].tolist(),
            })

        print(f"  encoded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

# -------------------------
# Save to parquet
# -------------------------
df = pd.DataFrame(results)
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Wrote {len(df)} query embeddings (dim={len(results[0]['embedding'])}) "
      f"to {OUTPUT_PARQUET}")