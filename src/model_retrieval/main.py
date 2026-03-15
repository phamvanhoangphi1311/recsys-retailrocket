"""
STEP 7: TWO-TOWER + MULTI-CHANNEL RETRIEVAL
Input: embeddings from Steps 4/5/6, features from Step 3
Output: candidates per user + FAISS indices

4 retrieval channels:
  A. Two-Tower (features → embeddings → FAISS)  ← NEW
  B. LightGCN (graph collaborative → FAISS)
  C. Item2Vec (item similarity)
  D. Popular items (cold user fallback)

Cần: pip install faiss-cpu torch
"""

import pandas as pd
import numpy as np
import json
import os
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

OUT_DIR = "data/processed"
EMB_DIR = "data/processed/embeddings"
FEAT_DIR = "data/processed/features"
RETR_DIR = "data/processed/retrieval"
os.makedirs(RETR_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 0. LOAD DATA
# ============================================================
section("0. LOAD DATA")

# Embeddings
lightgcn_user_emb = np.load(os.path.join(EMB_DIR, "lightgcn_user_emb.npy"))
lightgcn_item_emb = np.load(os.path.join(EMB_DIR, "lightgcn_item_emb.npy"))
item2vec_emb = np.load(os.path.join(EMB_DIR, "item2vec_embeddings.npy"))

with open(os.path.join(OUT_DIR, "user2id.json")) as f:
    user2id = json.load(f)
with open(os.path.join(OUT_DIR, "item2id.json")) as f:
    item2id = json.load(f)
with open(os.path.join(EMB_DIR, "item2vec_item2idx.json")) as f:
    item2vec_item2idx = json.load(f)

id2item = {v: k for k, v in item2id.items()}
item2vec_idx2item = {v: k for k, v in item2vec_item2idx.items()}

# Features
user_feat = pd.read_parquet(os.path.join(FEAT_DIR, "user_features.parquet"))
item_feat = pd.read_parquet(os.path.join(FEAT_DIR, "item_features.parquet"))

# Pairs
train_pairs = pd.read_parquet(os.path.join(OUT_DIR, "train_pairs.parquet"))
val_pairs = pd.read_parquet(os.path.join(OUT_DIR, "val_pairs.parquet"))
test_pairs = pd.read_parquet(os.path.join(OUT_DIR, "test_pairs.parquet"))

# Events for recent items + popularity
if 'timestamp_dt' not in events.columns:
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], unit='ms')

user_item_all = pd.read_parquet(os.path.join(OUT_DIR, "user_item_labeled.parquet"))
train_cutoff = user_item_all['first_ts'].quantile(0.70)
del user_item_all
train_events = events[events['timestamp_dt'] <= train_cutoff]

print(f"LightGCN: user={lightgcn_user_emb.shape}, item={lightgcn_item_emb.shape}")
print(f"Item2Vec: {item2vec_emb.shape}")
print(f"Users: {len(user2id):,}, Items: {len(item2id):,}")


# ============================================================
# A. TWO-TOWER MODEL
# ============================================================
section("A. TWO-TOWER MODEL")

"""
Two-Tower:
  User Tower: user features → MLP → user vector (64-dim)
  Item Tower: item features → MLP → item vector (64-dim)
  Score = dot product(user_vec, item_vec)
  
Train: positive pairs score > negative pairs score (BPR loss)

Tại sao Two-Tower bổ sung LightGCN:
  - LightGCN: learn từ graph structure (indirect CF)
  - Two-Tower: learn từ features (behavioral patterns)
  - LightGCN tốt cho warm users, Two-Tower có thể handle warm users 
    với features KHÁC nhau (session behavior, category preference...)
"""

# Prepare features as tensors
user_feature_cols = [c for c in user_feat.columns if c != 'visitorid']
item_feature_cols = [c for c in item_feat.columns 
                     if c not in ['itemid', 'categoryid'] and item_feat[c].dtype in [np.float64, np.int64, np.float32]]

# User feature matrix (indexed by user2id)
user_feat_matrix = np.zeros((len(user2id), len(user_feature_cols)), dtype=np.float32)
user_feat_indexed = user_feat.set_index('visitorid')
for uid, idx in user2id.items():
    if uid in user_feat_indexed.index:
        user_feat_matrix[idx] = user_feat_indexed.loc[uid, user_feature_cols].values

# Item feature matrix (indexed by item2id)
item_feat_matrix = np.zeros((len(item2id), len(item_feature_cols)), dtype=np.float32)
item_feat_indexed = item_feat.set_index('itemid')
for iid, idx in item2id.items():
    if iid in item_feat_indexed.index:
        row = item_feat_indexed.loc[iid]
        item_feat_matrix[idx] = row[item_feature_cols].values

# Normalize features (zero mean, unit variance)
user_mean = user_feat_matrix.mean(axis=0, keepdims=True)
user_std = user_feat_matrix.std(axis=0, keepdims=True) + 1e-8
user_feat_matrix = (user_feat_matrix - user_mean) / user_std

item_mean = item_feat_matrix.mean(axis=0, keepdims=True)
item_std = item_feat_matrix.std(axis=0, keepdims=True) + 1e-8
item_feat_matrix = (item_feat_matrix - item_mean) / item_std

user_feat_tensor = torch.FloatTensor(user_feat_matrix).to(DEVICE)
item_feat_tensor = torch.FloatTensor(item_feat_matrix).to(DEVICE)

print(f"User features: {user_feat_matrix.shape}")
print(f"Item features: {item_feat_matrix.shape}")

# Two-Tower model
TOWER_DIM = 64

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, tower_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, tower_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, tower_dim),
        )
    
    def forward(self, user_feats, item_feats):
        u = self.user_tower(user_feats)
        i = self.item_tower(item_feats)
        return u, i
    
    def score(self, u, i):
        return (u * i).sum(dim=1)

tt_model = TwoTower(len(user_feature_cols), len(item_feature_cols), TOWER_DIM).to(DEVICE)

# Train with BPR on positive pairs
pos_df = train_pairs[train_pairs['label'] == 1].copy()
pos_df['user_idx'] = pos_df['visitorid'].map(user2id)
pos_df['item_idx'] = pos_df['itemid'].map(item2id)
pos_df = pos_df.dropna(subset=['user_idx', 'item_idx']).astype({'user_idx': int, 'item_idx': int})
user_pos_items = pos_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

n_items_tt = len(item2id)
optimizer = optim.Adam(tt_model.parameters(), lr=0.001)

print(f"Training Two-Tower... ({len(pos_df):,} positive pairs)")

for epoch in range(30):
    tt_model.train()
    total_loss = 0
    
    # Mini-batch BPR
    indices = np.random.permutation(len(pos_df))
    batch_size = 2048
    n_batches = 0
    
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start+batch_size]
        users = pos_df.iloc[batch_idx]['user_idx'].values
        pos_items = pos_df.iloc[batch_idx]['item_idx'].values
        neg_items = np.random.randint(0, n_items_tt, size=len(users))
        
        u_feat = user_feat_tensor[users]
        pi_feat = item_feat_tensor[pos_items]
        ni_feat = item_feat_tensor[neg_items]
        
        u_vec, pi_vec = tt_model(u_feat, pi_feat)
        _, ni_vec = tt_model(u_feat, ni_feat)
        
        pos_scores = tt_model.score(u_vec, pi_vec)
        neg_scores = tt_model.score(u_vec, ni_vec)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/30: loss = {total_loss/max(n_batches,1):.4f}")

# Generate Two-Tower embeddings for all users and items
tt_model.eval()
with torch.no_grad():
    tt_user_emb = tt_model.user_tower(user_feat_tensor).cpu().numpy()
    tt_item_emb = tt_model.item_tower(item_feat_tensor).cpu().numpy()

print(f"Two-Tower user emb: {tt_user_emb.shape}")
print(f"Two-Tower item emb: {tt_item_emb.shape}")

np.save(os.path.join(EMB_DIR, "twotower_user_emb.npy"), tt_user_emb)
np.save(os.path.join(EMB_DIR, "twotower_item_emb.npy"), tt_item_emb)
torch.save(tt_model.state_dict(), os.path.join(EMB_DIR, "twotower_model.pt"))


# ============================================================
# B. BUILD FAISS INDICES
# ============================================================
section("B. BUILD FAISS INDICES")

def normalize(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)

# Index 1: Two-Tower
tt_item_norm = normalize(tt_item_emb).astype(np.float32)
tt_user_norm = normalize(tt_user_emb).astype(np.float32)
index_tt = faiss.IndexFlatIP(TOWER_DIM)
index_tt.add(tt_item_norm)
print(f"Two-Tower FAISS: {index_tt.ntotal} items")

# Index 2: LightGCN
gcn_item_norm = normalize(lightgcn_item_emb).astype(np.float32)
gcn_user_norm = normalize(lightgcn_user_emb).astype(np.float32)
index_gcn = faiss.IndexFlatIP(lightgcn_item_emb.shape[1])
index_gcn.add(gcn_item_norm)
print(f"LightGCN FAISS: {index_gcn.ntotal} items")

# Index 3: Item2Vec
i2v_norm = normalize(item2vec_emb).astype(np.float32)
index_i2v = faiss.IndexFlatIP(item2vec_emb.shape[1])
index_i2v.add(i2v_norm)
print(f"Item2Vec FAISS: {index_i2v.ntotal} items")


# ============================================================
# C. POPULARITY CHANNEL
# ============================================================
section("C. POPULARITY")

item_pop = train_events.groupby('itemid')['visitorid'].nunique().sort_values(ascending=False)
top_popular = item_pop.head(100).index.tolist()
print(f"Top 100 popular items ready")


# ============================================================
# D. RETRIEVE CANDIDATES
# ============================================================
section("D. RETRIEVE CANDIDATES")

K_TT = 30    # Two-Tower candidates
K_GCN = 30   # LightGCN candidates
K_I2V = 20   # Item2Vec candidates
K_POP = 20   # Popular fallback

# User → train items (for filtering already seen)
user_train_items = train_pairs.groupby('visitorid')['itemid'].apply(set).to_dict()

# User → recent items (for Item2Vec seeds)
recent_views = train_events[train_events['event'] == 'view'].sort_values('timestamp_dt')
user_recent_items = recent_views.groupby('visitorid')['itemid'].apply(
    lambda x: list(x.tail(3))
).to_dict()

def retrieve_for_user(visitor_id):
    candidates = set()
    seen = user_train_items.get(visitor_id, set())
    
    # Channel A: Two-Tower
    if visitor_id in user2id:
        uid = user2id[visitor_id]
        vec = tt_user_norm[uid:uid+1]
        D, I = index_tt.search(vec, K_TT + len(seen))
        for idx in I[0]:
            if idx < len(id2item):
                iid = id2item[idx]
                if iid not in seen:
                    candidates.add(('tt', iid))
                    if sum(1 for c in candidates if c[0] == 'tt') >= K_TT:
                        break
    
    # Channel B: LightGCN
    if visitor_id in user2id:
        uid = user2id[visitor_id]
        vec = gcn_user_norm[uid:uid+1]
        D, I = index_gcn.search(vec, K_GCN + len(seen))
        for idx in I[0]:
            if idx < len(id2item):
                iid = id2item[idx]
                if iid not in seen:
                    candidates.add(('gcn', iid))
                    if sum(1 for c in candidates if c[0] == 'gcn') >= K_GCN:
                        break
    
    # Channel C: Item2Vec
    recent = user_recent_items.get(visitor_id, [])
    for seed in recent:
        if seed in item2vec_item2idx:
            sidx = item2vec_item2idx[seed]
            vec = i2v_norm[sidx:sidx+1]
            D, I = index_i2v.search(vec, K_I2V)
            for idx in I[0]:
                if idx in item2vec_idx2item:
                    iid = item2vec_idx2item[idx]
                    if iid not in seen and iid != seed:
                        candidates.add(('i2v', iid))
    
    # Channel D: Popular
    for iid in top_popular:
        if iid not in seen:
            candidates.add(('pop', iid))
            if sum(1 for c in candidates if c[0] == 'pop') >= K_POP:
                break
    
    # Return deduplicated items + track which channel
    item_channels = {}
    for channel, iid in candidates:
        if iid not in item_channels:
            item_channels[iid] = channel
    
    return item_channels

# Retrieve for val + test
print(f"Retrieving candidates...")

val_candidates = {}
for i, uid in enumerate(val_pairs['visitorid'].unique()):
    val_candidates[uid] = retrieve_for_user(uid)
    if (i+1) % 5000 == 0:
        print(f"  Val: {i+1:,}")

test_candidates = {}
for i, uid in enumerate(test_pairs['visitorid'].unique()):
    test_candidates[uid] = retrieve_for_user(uid)
    if (i+1) % 5000 == 0:
        print(f"  Test: {i+1:,}")

val_sizes = [len(c) for c in val_candidates.values()]
test_sizes = [len(c) for c in test_candidates.values()]
print(f"\nVal candidates/user: mean={np.mean(val_sizes):.0f}, median={np.median(val_sizes):.0f}")
print(f"Test candidates/user: mean={np.mean(test_sizes):.0f}, median={np.median(test_sizes):.0f}")


# ============================================================
# E. EVALUATE RETRIEVAL PER CHANNEL
# ============================================================
section("E. EVALUATE RETRIEVAL")

def eval_recall(pairs_df, candidates_dict, k=50):
    positives = pairs_df[pairs_df['label'] == 1].groupby('visitorid')['itemid'].apply(set).to_dict()
    
    recalls = []
    channel_hits = {'tt': 0, 'gcn': 0, 'i2v': 0, 'pop': 0}
    channel_total = {'tt': 0, 'gcn': 0, 'i2v': 0, 'pop': 0}
    
    for user, pos_items in positives.items():
        if user not in candidates_dict:
            continue
        cands = candidates_dict[user]
        cand_items = set(cands.keys())
        
        hits = pos_items & cand_items
        recalls.append(len(hits) / len(pos_items) if pos_items else 0)
        
        # Track which channel found hits
        for item in hits:
            ch = cands[item]
            channel_hits[ch] = channel_hits.get(ch, 0) + 1
        for item, ch in cands.items():
            channel_total[ch] = channel_total.get(ch, 0) + 1
    
    print(f"  Recall@{k}: {np.mean(recalls):.4f} (evaluated on {len(recalls):,} users)")
    print(f"  Channel contribution to hits:")
    total_hits = sum(channel_hits.values())
    for ch in ['tt', 'gcn', 'i2v', 'pop']:
        h = channel_hits.get(ch, 0)
        t = channel_total.get(ch, 0)
        print(f"    {ch:4s}: {h:>5,} hits / {t:>8,} candidates ({h/max(total_hits,1)*100:.1f}% of hits)")
    
    return np.mean(recalls)

print("Val:")
val_recall = eval_recall(val_pairs, val_candidates)
print("\nTest:")
test_recall = eval_recall(test_pairs, test_candidates)


# ============================================================
# F. SAVE
# ============================================================
section("F. SAVE")

# Save candidates
def save_candidates(cands_dict, name):
    rows = []
    for user, item_channels in cands_dict.items():
        for item, channel in item_channels.items():
            rows.append({'visitorid': user, 'itemid': item, 'channel': channel})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(RETR_DIR, f"{name}_candidates.parquet"), index=False)
    return df

val_cand_df = save_candidates(val_candidates, 'val')
test_cand_df = save_candidates(test_candidates, 'test')

faiss.write_index(index_tt, os.path.join(RETR_DIR, "faiss_twotower.index"))
faiss.write_index(index_gcn, os.path.join(RETR_DIR, "faiss_lightgcn.index"))
faiss.write_index(index_i2v, os.path.join(RETR_DIR, "faiss_item2vec.index"))

with open(os.path.join(RETR_DIR, "popular_items.json"), 'w') as f:
    json.dump(top_popular, f)

print(f"✅ val_candidates.parquet: {len(val_cand_df):,}")
print(f"✅ test_candidates.parquet: {len(test_cand_df):,}")
print(f"✅ FAISS indices saved")
print(f"\n✅ STEP 7 COMPLETE")