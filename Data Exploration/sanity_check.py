import pandas as pd

# 1. load mapping
mapping = pd.read_csv('../conversational/CAST2019/CAST2019_ID_Mapping.tsv', sep='\t')
print(f"Total passages in mapping: {len(mapping)}")

# 2. check one parquet chunk
df = pd.read_parquet('../conversational/CAST2019/snowflake_embeddings/cast2019_snowflake_v2.rank0.part00000.parquet')
print(f"Parquet chunk shape: {df.shape}")
print(f"First few ids: {df['id'].head().tolist()}")
print(f"Embedding dimension: {len(df['embedding'][0])}")

# 3. check mapping aligns with parquet ids
first_parquet_id = df['id'][0]
mapping_match = mapping[mapping['id'] == first_parquet_id]
print(f"First parquet id '{first_parquet_id}' found in mapping at index: {mapping_match['index'].values}")

# 4. load topics and qrels
topics = pd.read_csv('topics/topics.tsv', header=None, 
                     names=['query_id', 'query'], sep=',')
qrels = pd.read_csv('topics/qrels.qrel', header=None,
                    names=['query_id', 'iteration', 'passage_id', 'relevance'], sep=',')

print(f"\nUnique queries in topics: {topics['query_id'].nunique()}")
print(f"Unique queries in qrels: {qrels['query_id'].nunique()}")

# 5. check qrel passage ids exist in mapping
qrel_passages = set(qrels['passage_id'].unique())
mapping_passages = set(mapping['id'].unique())
missing = qrel_passages - mapping_passages
print(f"Passages in qrels but not in mapping: {len(missing)}")
if len(missing) > 0:
    print(f"Example missing: {list(missing)[:5]}")

# 6. conversation structure
topics['conv_id'] = topics['query_id'].apply(lambda x: x.split('_')[0])
topics['turn_id'] = topics['query_id'].apply(lambda x: x.split('_')[1])
print(f"\nNumber of conversations: {topics['conv_id'].nunique()}")
print(f"Average turns per conversation: {topics.groupby('conv_id').size().mean():.1f}")
print(f"Max turns in one conversation: {topics.groupby('conv_id').size().max()}")
print(topics.head(10))