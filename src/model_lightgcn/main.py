"""
STEP 5: LIGHTGCN
Assumes: data/processed/ files from step 2.

Build user-item bipartite graph → propagate embeddings → learn collaborative signal.
Edge weights: view=0.1, cart=0.5, transaction=1.0

Output: data/processed/embeddings/lightgcn_user_emb.npy, lightgcn_item_emb.npy

Cần: pip install torch
"""

import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

OUT_DIR = "data/processed"
EMB_DIR = "data/processed/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# 1. BUILD GRAPH FROM TRAINING PAIRS
# ============================================================
section("1. BUILD GRAPH")

train_pairs = pd.read_parquet(os.path.join(OUT_DIR, "train_pairs.parquet"))

# Edge weights from label_source
weight_map = {'view_only': 0.1, 'addtocart': 0.5, 'transaction': 1.0}
train_pairs['edge_weight'] = train_pairs['label_source'].map(weight_map)

# ID mappings
unique_users = sorted(train_pairs['visitorid'].unique())
unique_items = sorted(train_pairs['itemid'].unique())
user2id = {u: i for i, u in enumerate(unique_users)}
item2id = {it: i for i, it in enumerate(unique_items)}

train_pairs['user_idx'] = train_pairs['visitorid'].map(user2id)
train_pairs['item_idx'] = train_pairs['itemid'].map(item2id)

n_users = len(user2id)
n_items = len(item2id)

print(f"Users: {n_users:,}, Items: {n_items:,}")
print(f"Edges: {len(train_pairs):,}")
for src, w in weight_map.items():
    n = (train_pairs['label_source'] == src).sum()
    print(f"  {src} (w={w}): {n:,}")

# Save mappings
with open(os.path.join(OUT_DIR, "user2id.json"), 'w') as f:
    json.dump(user2id, f)
with open(os.path.join(OUT_DIR, "item2id.json"), 'w') as f:
    json.dump(item2id, f)


# ============================================================
# 2. ADJACENCY MATRIX
# ============================================================
section("2. ADJACENCY MATRIX")

user_indices = torch.LongTensor(train_pairs['user_idx'].values)
item_indices = torch.LongTensor(train_pairs['item_idx'].values) + n_users
edge_weights = torch.FloatTensor(train_pairs['edge_weight'].values)

row = torch.cat([user_indices, item_indices])
col = torch.cat([item_indices, user_indices])
weights = torch.cat([edge_weights, edge_weights])

n_nodes = n_users + n_items

degree = torch.zeros(n_nodes)
degree.index_add_(0, row, weights)
degree_inv_sqrt = torch.pow(degree.clamp(min=1), -0.5)
norm_weights = degree_inv_sqrt[row] * weights * degree_inv_sqrt[col]

adj = torch.sparse_coo_tensor(
    torch.stack([row, col]), norm_weights, size=(n_nodes, n_nodes)
).coalesce().to(DEVICE)

print(f"Adjacency: {n_nodes} × {n_nodes}")


# ============================================================
# 3. MODEL + TRAINING
# ============================================================
section("3. TRAIN LIGHTGCN")

EMB_DIM = 64
N_LAYERS = 3
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 50
REG_WEIGHT = 1e-4

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.adj = adj
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    
    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            emb_list.append(all_emb)
        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]
    
    def bpr_loss(self, user_final, item_final, users, pos_items, neg_items):
        u_e = user_final[users]
        p_e = item_final[pos_items]
        n_e = item_final[neg_items]
        pos_scores = (u_e * p_e).sum(dim=1)
        neg_scores = (u_e * n_e).sum(dim=1)
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        reg = REG_WEIGHT * (
            self.user_emb.weight[users].norm(2).pow(2) +
            self.item_emb.weight[pos_items].norm(2).pow(2) +
            self.item_emb.weight[neg_items].norm(2).pow(2)
        ) / len(users)
        return bpr + reg

# Positive pairs for BPR
pos_pairs = train_pairs[train_pairs['label'] == 1][['user_idx', 'item_idx']].dropna()
pos_pairs = pos_pairs.astype(int)
user_pos_items = pos_pairs.groupby('user_idx')['item_idx'].apply(set).to_dict()

class BPRDataset(Dataset):
    def __init__(self, pos_pairs, n_items, user_pos_items):
        self.users = pos_pairs['user_idx'].values
        self.pos_items = pos_pairs['item_idx'].values
        self.n_items = n_items
        self.user_pos_items = user_pos_items
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        user = self.users[idx]
        pos = self.pos_items[idx]
        neg = np.random.randint(0, self.n_items)
        while neg in self.user_pos_items.get(user, set()):
            neg = np.random.randint(0, self.n_items)
        return user, pos, neg

train_loader = DataLoader(
    BPRDataset(pos_pairs, n_items, user_pos_items),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS, adj).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"emb_dim={EMB_DIM}, layers={N_LAYERS}, epochs={EPOCHS}")
print(f"Positive pairs: {len(pos_pairs):,}, Batches: {len(train_loader):,}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0
    user_final, item_final = model()
    
    for users, pos_items, neg_items in train_loader:
        users = users.to(DEVICE)
        pos_items = pos_items.to(DEVICE)
        neg_items = neg_items.to(DEVICE)
        loss = model.bpr_loss(user_final, item_final, users, pos_items, neg_items)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: loss = {total_loss/max(n_batches,1):.4f}")


# ============================================================
# 4. SAVE EMBEDDINGS
# ============================================================
section("4. SAVE")

model.eval()
with torch.no_grad():
    u_emb, i_emb = model()
    u_emb = u_emb.cpu().numpy()
    i_emb = i_emb.cpu().numpy()

np.save(os.path.join(EMB_DIR, "lightgcn_user_emb.npy"), u_emb)
np.save(os.path.join(EMB_DIR, "lightgcn_item_emb.npy"), i_emb)
torch.save(model.state_dict(), os.path.join(EMB_DIR, "lightgcn_model.pt"))

print(f"✅ lightgcn_user_emb.npy: {u_emb.shape}")
print(f"✅ lightgcn_item_emb.npy: {i_emb.shape}")
print(f"\n✅ STEP 5 COMPLETE")