"""
Encode CAST2019 topics.tsv queries using the DRAGON+ query encoder.
 
Input:  conversational/CAST2019/topics/topics.tsv
        format per line: turn_id,query   (e.g. "31_1,What is throat cancer?")
        turn_id format:  {conversation_id}_{turn_number}
 
Output: conversational/CAST2019/dragon_embeddings/topics_dragon_embeddings.parquet
        columns: conversation_id (str), turn_id (str), query (str), embedding (list[float])
 
Usage:
    python encode_topics_dragon.py \
        --input /path/to/topics.tsv \
        --output /path/to/topics_dragon_embeddings.parquet \
        --batch-size 64
"""
 
import argparse
import csv
import sys
 
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
 
 
def load_topics(path: str) -> pd.DataFrame:
    """Load turn_id,query pairs from a comma-delimited topics file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            turn_id, query = line[0], ",".join(line[1:]).strip()
            conversation_id = turn_id.split("_")[0]
            rows.append((conversation_id, turn_id, query))
    df = pd.DataFrame(rows, columns=["conversation_id", "id", "query"])
    return df
 
 
@torch.no_grad()
def encode_queries(
    queries: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode a list of query strings with the DRAGON+ query encoder (CLS pooling)."""
    model.eval()
    all_embeddings = []
 
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
 
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        all_embeddings.append(cls_emb.cpu().numpy())
 
        done = min(start + batch_size, len(queries))
        print(f"  encoded {done}/{len(queries)}", file=sys.stderr)
 
    return np.concatenate(all_embeddings, axis=0)
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/home/toploc1/Datasets/conversational/CAST2019/topics/topics.tsv",
        help="Path to topics.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/topics_dragon_embeddingsv2.parquet",
        help="Path to output .parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="facebook/dragon-plus-query-encoder",
        help="HF model id for the query encoder",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run encoding on",
    )
    args = parser.parse_args()
 
    print(f"Loading topics from {args.input}", file=sys.stderr)
    df = load_topics(args.input)
    print(f"  loaded {len(df)} query rows across "
          f"{df['conversation_id'].nunique()} conversations", file=sys.stderr)
 
    print(f"Loading tokenizer/model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
 
    print(f"Encoding {len(df)} queries on {args.device} "
          f"(batch_size={args.batch_size})", file=sys.stderr)
    embeddings = encode_queries(
        df["query"].tolist(), tokenizer, model, args.device, args.batch_size
    )
 
    df["embedding"] = list(embeddings.astype(np.float32))
 
    print(f"Writing output to {args.output}", file=sys.stderr)
    df.to_parquet(args.output, index=False)
    print("Done.", file=sys.stderr)
 
 
if __name__ == "__main__":
    main()