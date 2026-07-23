import pyarrow.parquet as pq
import numpy as np
import glob

import faiss

PATH_SNOWFLAKE = "snowflake_embeddings/cast2019_snowflake_v2.rank0.part00000.parquet"
PATH_DRAGON = "../../conversational/CAST2019/dragon_embeddings/cast2019_dragon.rank0.part00000.parquet"

#table = pq.read_table("PATH_DRAGON")
#print(table.schema)


#schema check


table = pq.read_table(PATH_DRAGON)

print("Columns:", table.column_names)
print("Schema:", table.schema)



ids = table["id"].to_pylist()
emb_list = table["embedding"].to_pylist()

print("Number of rows:", len(ids))
print("First embedding length:", len(emb_list[0]))


#checking dimensions
dims = set(len(e) for e in emb_list[:1000])  # check first 1000
print("Unique dimensions:", dims)


#converting to numpy array
embeddings = np.array(emb_list, dtype="float32")

print("Shape:", embeddings.shape)


#norms = np.linalg.norm(embeddings, axis=1)

#print("Min norm:", norms.min())
#print("Max norm:", norms.max())
#print("Mean norm:", norms.mean())

d = embeddings.shape[1]

index = faiss.IndexFlatIP(d)  # or IndexFlatIP if cosine
index.add(embeddings)

print("Indexed vectors:", index.ntotal)

query = embeddings[0:1]

k = 5
D, I = index.search(query, k)

print("Indices:", I)
print("Distances:", D)

# Map back to IDs
results = [ids[i] for i in I[0]]
print("Retrieved IDs:", results)


'''
import pyarrow.parquet as pq

table = pq.read_table("../conversational/CAST2019/topics/topics_snowflake_embeddings.parquet")
print("Columns:", table.schema.names)
print("Num rows:", len(table))
print()
for col in table.schema.names:
    print(f"{col}:", table[col].to_pylist()[:3])
    '''