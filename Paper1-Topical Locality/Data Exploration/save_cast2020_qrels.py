"""
Save the TREC CAsT 2020 relevance judgments (qrels) to disk from ir_datasets.

Writes TWO files so you have whichever format you need:
  * cast2020_qrels.qrel        comma-separated  qid,iteration,docid,grade
                               (matches your grid scripts' parser:
                                parts=split(','); qid,docid,grade = 0,2,3)
  * cast2020_qrels.trec        standard TREC     qid iteration docid grade
                               (space-separated; for trec_eval)
"""

import ir_datasets

IR_DATASET   = "trec-cast/v1/2020/judged"
OUT_COMMA    = "cast2020_qrels.qrel"     # your scripts read this one
OUT_TREC     = "cast2020_qrels.trec"     # standard trec_eval format

dataset = ir_datasets.load(IR_DATASET)

n = 0
with open(OUT_COMMA, "w", encoding="utf-8") as fc, \
     open(OUT_TREC, "w", encoding="utf-8") as ft:
    for q in dataset.qrels_iter():
        it = getattr(q, "iteration", "0") or "0"
        fc.write(f"{q.query_id},{it},{q.doc_id},{q.relevance}\n")
        ft.write(f"{q.query_id} {it} {q.doc_id} {q.relevance}\n")
        n += 1

print(f"Wrote {n} judgments")
print(f"  comma format (your scripts): {OUT_COMMA}")
print(f"  TREC format  (trec_eval)   : {OUT_TREC}")