"""
STEP 4: ITEM2VEC
Assumes: events df trong memory + data/processed/ files from step 2.

Item2Vec = Word2Vec on user view sequences.
Items co-occurring in same user's history → similar embeddings.

Output: data/processed/embeddings/item2vec_embeddings.npy

Cần: pip install gensim
"""

import pandas as pd
import numpy as np
import json
import os
from gensim.models import Word2Vec

OUT_DIR = "data/processed"
EMB_DIR = "data/processed/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

if 'timestamp_dt' not in events.columns:
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], unit='ms')


# ============================================================
# 1. BUILD SEQUENCES (training data only)
# ============================================================
section("1. BUILD VIEW SEQUENCES")

"""
Dùng USER-LEVEL sequences (không phải session-level).

Tại sao: EDA Part 10 cho thấy:
  - User-level: median=7 items → đủ dài cho Word2Vec
  - Session-level: median=1 item → quá ngắn, Word2Vec cần ≥2 items

Sequence = list items user viewed, ordered by time.
Chỉ dùng training events (trước train_cutoff).
"""

train_pairs = pd.read_parquet(os.path.join(OUT_DIR, "train_pairs.parquet"))
# Dùng cùng cutoff với Step 2: 70th percentile first_ts
user_item_all = pd.read_parquet(os.path.join(OUT_DIR, "user_item_labeled.parquet"))
train_cutoff = user_item_all['first_ts'].quantile(0.70)
del user_item_all

train_views = events[
    (events['event'] == 'view') & (events['timestamp_dt'] <= train_cutoff)
].sort_values(['visitorid', 'timestamp_dt'])

# Build sequences: user → list of item IDs (strings)
sequences = []
for user, group in train_views.groupby('visitorid'):
    seq = [str(x) for x in group['itemid'].tolist()]
    if len(seq) >= 2:
        sequences.append(seq)

# Unique items in sequences
all_items = set()
for s in sequences:
    all_items.update(s)

print(f"Sequences: {len(sequences):,}")
print(f"Unique items: {len(all_items):,}")

seq_lens = [len(s) for s in sequences]
print(f"Sequence length: median={np.median(seq_lens):.0f}, "
      f"P75={np.percentile(seq_lens, 75):.0f}, "
      f"P90={np.percentile(seq_lens, 90):.0f}")


# ============================================================
# 2. TRAIN ITEM2VEC
# ============================================================
section("2. TRAIN")

"""
Hyperparameters:
  vector_size=128: item embedding dimension
  window=3: context window
    EDA session median=2 items → nearby items in sequence are within same session.
    Window=3 captures "items viewed close together".
  min_count=3: ignore very rare items
  sg=1: Skip-gram (better than CBOW for sparse data)
  negative=10: negative samples per positive pair
  ns_exponent=0.75: frequency smoothing (reduce popular item dominance)
  epochs=15: more than NLP default because sequences shorter
  workers=1: must be 1 in Jupyter notebooks
"""

model = Word2Vec(
    sentences=sequences,
    vector_size=128,
    window=3,
    min_count=3,
    sg=1,
    negative=10,
    ns_exponent=0.75,
    epochs=15,
    workers=1,
    seed=42,
)

vocab_size = len(model.wv)
print(f"Vocabulary: {vocab_size:,} items")

# Quick test
test_items = list(model.wv.index_to_key[:3])
for item in test_items:
    similar = model.wv.most_similar(item, topn=3)
    print(f"\n  Item {item} most similar:")
    for s_item, score in similar:
        print(f"    → {s_item} ({score:.3f})")


# ============================================================
# 3. SAVE
# ============================================================
section("3. SAVE")

item_ids = list(model.wv.index_to_key)
embeddings = np.array([model.wv[item] for item in item_ids])
item2emb_idx = {item: i for i, item in enumerate(item_ids)}

np.save(os.path.join(EMB_DIR, "item2vec_embeddings.npy"), embeddings)
with open(os.path.join(EMB_DIR, "item2vec_item2idx.json"), 'w') as f:
    json.dump(item2emb_idx, f)
model.save(os.path.join(EMB_DIR, "item2vec_model.bin"))

print(f"✅ item2vec_embeddings.npy: {embeddings.shape}")
print(f"✅ item2vec_item2idx.json: {len(item2emb_idx):,} items")
print(f"\n✅ STEP 4 COMPLETE")# initial build
# initial build
# initial build
