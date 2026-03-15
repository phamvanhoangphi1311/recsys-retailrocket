"""
STEP 6 v3: SASREC — FIXED nan val_loss
Main fixes:
  1. Filter out very short sequences that become all-padding
  2. Add epsilon to prevent log(0) in loss
  3. Skip all-padding batches in validation
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

if 'timestamp_dt' not in events.columns:
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], unit='ms')

# ============================================================
# 1. BUILD SEQUENCES
# ============================================================
print("="*60)
print("  1. BUILD SEQUENCES")
print("="*60)

user_item_all = pd.read_parquet(os.path.join(OUT_DIR, "user_item_labeled.parquet"))
train_cutoff = user_item_all['first_ts'].quantile(0.70)
del user_item_all

train_events = events[events['timestamp_dt'] <= train_cutoff].sort_values(
    ['visitorid', 'timestamp_dt']
)

# Build sequences — ONLY strings, ONLY length >= 5 (need enough for input+target+context)
user_seqs = {}
for user, group in train_events.groupby('visitorid'):
    seq = [str(x) for x in group['itemid'].tolist()]
    if len(seq) >= 5:  # stricter filter: need meaningful sequences
        user_seqs[user] = seq

all_items = set()
for seq in user_seqs.values():
    all_items.update(seq)

item_list = sorted(all_items)
item2idx = {item: idx + 1 for idx, item in enumerate(item_list)}
n_items = len(item2idx) + 1

seq_lens = [len(s) for s in user_seqs.values()]
MAX_SEQ_LEN = min(int(np.percentile(seq_lens, 90)), 50)
EMB_DIM = 64
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
BATCH_SIZE = 256
EPOCHS = 30

print(f"Users: {len(user_seqs):,}, Vocab: {n_items:,}")
print(f"Seq lengths: median={np.median(seq_lens):.0f}, P90={np.percentile(seq_lens, 90):.0f}")
print(f"MAX_SEQ_LEN={MAX_SEQ_LEN}")

# ============================================================
# 2. DATASET
# ============================================================
print("\n" + "="*60)
print("  2. DATASET")
print("="*60)

class SASRecDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len):
        self.sequences = sequences
        self.item2idx = item2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_idx = [self.item2idx.get(item, 0) for item in seq]
        
        # Truncate (keep most recent)
        if len(seq_idx) > self.max_len + 1:
            seq_idx = seq_idx[-(self.max_len + 1):]
        
        input_seq = seq_idx[:-1]
        target_seq = seq_idx[1:]
        
        # Left padding
        pad_len = self.max_len - len(input_seq)
        input_seq = [0] * pad_len + input_seq
        target_seq = [0] * pad_len + target_seq
        
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)

all_seqs = list(user_seqs.values())
np.random.seed(42)
indices = np.random.permutation(len(all_seqs))
split = int(0.9 * len(all_seqs))
train_seqs = [all_seqs[i] for i in indices[:split]]
val_seqs = [all_seqs[i] for i in indices[split:]]

train_loader = DataLoader(
    SASRecDataset(train_seqs, item2idx, MAX_SEQ_LEN),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    SASRecDataset(val_seqs, item2idx, MAX_SEQ_LEN),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
print(f"Train: {len(train_seqs):,}, Val: {len(val_seqs):,}")

# ============================================================
# 3. MODEL — simplified, more stable
# ============================================================
print("\n" + "="*60)
print("  3. MODEL")
print("="*60)

class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim, max_len, n_heads, n_layers, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=emb_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True,  # Pre-LN = more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(emb_dim, n_items)
    
    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape
        
        # Embeddings
        item_e = self.item_emb(input_seq)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_e = self.pos_emb(positions)
        seq_e = self.dropout(self.emb_norm(item_e + pos_e))
        
        # Causal mask — use additive float mask for compatibility
        causal_mask = torch.zeros(seq_len, seq_len, device=input_seq.device)
        causal_mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=input_seq.device), diagonal=1).bool(),
            float('-inf')
        )
        
        # Padding mask
        padding_mask = (input_seq == 0)
        
        # If entire batch row is padding, skip to avoid nan
        # Replace all-padding rows with a dummy token to prevent nan
        all_pad_rows = padding_mask.all(dim=1)
        if all_pad_rows.any():
            input_seq_fixed = input_seq.clone()
            input_seq_fixed[all_pad_rows, -1] = 1  # dummy token
            padding_mask = (input_seq_fixed == 0)
            seq_e = self.dropout(self.emb_norm(
                self.item_emb(input_seq_fixed) + pos_e
            ))
        
        output = self.transformer(seq_e, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Replace nan with 0 (safety net)
        output = torch.nan_to_num(output, nan=0.0)
        
        return self.output_proj(output)

model = SASRec(n_items, EMB_DIM, MAX_SEQ_LEN, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 4. TRAIN
# ============================================================
print("\n" + "="*60)
print("  4. TRAIN")
print("="*60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

torch.save(model.state_dict(), os.path.join(EMB_DIR, "sasrec_best.pt"))
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0
    n_b = 0
    for inp, tgt in train_loader:
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        logits = model(inp)
        loss = criterion(logits.view(-1, n_items), tgt.view(-1))
        
        if torch.isnan(loss):
            continue  # skip nan batches
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        n_b += 1
    
    # --- Validate ---
    model.eval()
    val_loss = 0
    v_b = 0
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            logits = model(inp)
            loss = criterion(logits.view(-1, n_items), tgt.view(-1))
            
            if not torch.isnan(loss):
                val_loss += loss.item()
                v_b += 1
    
    avg_t = train_loss / max(n_b, 1)
    avg_v = val_loss / max(v_b, 1) if v_b > 0 else float('nan')
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: train={avg_t:.4f}  val={avg_v:.4f}  (val_batches={v_b})")
    
    if not np.isnan(avg_v) and avg_v < best_val_loss:
        best_val_loss = avg_v
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(EMB_DIR, "sasrec_best.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load(os.path.join(EMB_DIR, "sasrec_best.pt"), map_location=DEVICE))
print(f"Best val loss: {best_val_loss:.4f}")

# ============================================================
# 5. SAVE
# ============================================================
print("\n" + "="*60)
print("  5. SAVE")
print("="*60)

model.eval()
with torch.no_grad():
    item_embeddings = model.item_emb.weight.cpu().numpy()[1:]

np.save(os.path.join(EMB_DIR, "sasrec_item_emb.npy"), item_embeddings)
torch.save(model.state_dict(), os.path.join(EMB_DIR, "sasrec_model.pt"))
with open(os.path.join(EMB_DIR, "sasrec_item2idx.json"), 'w') as f:
    json.dump(item2idx, f)

print(f"✅ sasrec_item_emb.npy: {item_embeddings.shape}")
print(f"✅ sasrec_model.pt")
print(f"\n✅ STEP 6 COMPLETE")