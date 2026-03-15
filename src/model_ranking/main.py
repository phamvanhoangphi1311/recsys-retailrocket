"""
STEP 8: RANKING MODEL — LightGBM
Input: feature_table from Step 3, embedding similarities from Steps 4/5/7
Output: scored + ranked recommendations

Cần: pip install lightgbm
"""

import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

OUT_DIR = "data/processed"
FEAT_DIR = "data/processed/features"
EMB_DIR = "data/processed/embeddings"
RANK_DIR = "data/processed/ranking"
os.makedirs(RANK_DIR, exist_ok=True)

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 1. LOAD + ADD EMBEDDING FEATURES
# ============================================================
section("1. LOAD DATA + EMBEDDING FEATURES")

feat_df = pd.read_parquet(os.path.join(FEAT_DIR, "feature_table.parquet"))
with open(os.path.join(FEAT_DIR, "feature_meta.json")) as f:
    feat_meta = json.load(f)

feature_cols = list(feat_meta['feature_columns'])

# Add embedding dot product as feature
try:
    gcn_u = np.load(os.path.join(EMB_DIR, "lightgcn_user_emb.npy"))
    gcn_i = np.load(os.path.join(EMB_DIR, "lightgcn_item_emb.npy"))
    tt_u = np.load(os.path.join(EMB_DIR, "twotower_user_emb.npy"))
    tt_i = np.load(os.path.join(EMB_DIR, "twotower_item_emb.npy"))
    
    with open(os.path.join(OUT_DIR, "user2id.json")) as f:
        user2id = json.load(f)
    with open(os.path.join(OUT_DIR, "item2id.json")) as f:
        item2id = json.load(f)
    
    # Compute dot products
    gcn_dots = []
    tt_dots = []
    for _, row in feat_df.iterrows():
        uid = user2id.get(row['visitorid'])
        iid = item2id.get(row['itemid'])
        if uid is not None and iid is not None:
            gcn_dots.append(np.dot(gcn_u[uid], gcn_i[iid]))
            tt_dots.append(np.dot(tt_u[uid], tt_i[iid]))
        else:
            gcn_dots.append(0.0)
            tt_dots.append(0.0)
    
    feat_df['emb_lightgcn_dot'] = gcn_dots
    feat_df['emb_twotower_dot'] = tt_dots
    feature_cols.extend(['emb_lightgcn_dot', 'emb_twotower_dot'])
    print(f"  Added emb_lightgcn_dot + emb_twotower_dot")
except Exception as e:
    print(f"  Could not add embedding features: {e}")

print(f"Total features: {len(feature_cols)}")


# ============================================================
# 2. PREPARE SPLITS
# ============================================================
section("2. PREPARE SPLITS")

# Only numeric features
numeric_features = [c for c in feature_cols 
                    if feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_, np.uint8]]

train_df = feat_df[feat_df['split'] == 'train']
val_df = feat_df[feat_df['split'] == 'val']
test_df = feat_df[feat_df['split'] == 'test']

X_train, y_train, w_train = train_df[numeric_features].values, train_df['label'].values, train_df['weight'].values
X_val, y_val, w_val = val_df[numeric_features].values, val_df['label'].values, val_df['weight'].values
X_test, y_test = test_df[numeric_features].values, test_df['label'].values

pos_neg_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"Features: {len(numeric_features)}")
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Pos:Neg = 1:{pos_neg_ratio:.0f}")


# ============================================================
# 3. TRAIN LIGHTGBM
# ============================================================
section("3. TRAIN")

params = {
    'objective': 'binary',
    'metric': 'auc',
    'scale_pos_weight': pos_neg_ratio,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
}

train_data = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=numeric_features)
val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, feature_name=numeric_features, reference=train_data)

model = lgb.train(
    params, train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)
print(f"Best iteration: {model.best_iteration}")


# ============================================================
# 4. EVALUATE
# ============================================================
section("4. EVALUATE")

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

val_auc = roc_auc_score(y_val, val_pred)
test_auc = roc_auc_score(y_test, test_pred)
print(f"Val  AUC: {val_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# NDCG@K per user
def ndcg_at_k(df, preds, k=20):
    df = df.copy()
    df['pred'] = preds
    scores = []
    for user, g in df.groupby('visitorid'):
        if g['label'].sum() == 0 or len(g) < 2:
            continue
        top_k = g.nlargest(k, 'pred')
        dcg = sum(top_k['label'].values[i] / np.log2(i+2) for i in range(len(top_k)))
        ideal = sorted(g['label'].values, reverse=True)[:k]
        idcg = sum(ideal[i] / np.log2(i+2) for i in range(len(ideal)))
        scores.append(dcg / max(idcg, 1e-8))
    return np.mean(scores) if scores else 0

val_ndcg = ndcg_at_k(val_df, val_pred)
test_ndcg = ndcg_at_k(test_df, test_pred)
print(f"Val  NDCG@20: {val_ndcg:.4f}")
print(f"Test NDCG@20: {test_ndcg:.4f}")

# Hit Rate@K
test_df_copy = test_df.copy()
test_df_copy['pred'] = test_pred
hit_rates = {}
for k in [5, 10, 20]:
    hits = []
    for user, g in test_df_copy.groupby('visitorid'):
        top = g.nlargest(k, 'pred')
        hits.append(1 if top['label'].sum() > 0 else 0)
    hit_rates[k] = np.mean(hits)
    print(f"Test Hit Rate@{k}: {hit_rates[k]:.4f}")

# Warm vs Cold breakdown
with open(os.path.join(OUT_DIR, "user2id.json")) as f:
    train_user_ids = set(json.load(f).keys())

print(f"\nWarm user metrics (users in training):")
warm_mask = test_df_copy['visitorid'].isin(train_user_ids)
if warm_mask.sum() > 0:
    warm_auc = roc_auc_score(y_test[warm_mask], test_pred[warm_mask])
    print(f"  AUC: {warm_auc:.4f}")
    warm_hits = []
    for user, g in test_df_copy[warm_mask].groupby('visitorid'):
        top = g.nlargest(20, 'pred')
        warm_hits.append(1 if top['label'].sum() > 0 else 0)
    print(f"  Hit Rate@20: {np.mean(warm_hits):.4f}")

print(f"\nCold user metrics:")
cold_mask = ~warm_mask
if cold_mask.sum() > 0 and y_test[cold_mask].sum() > 0:
    cold_auc = roc_auc_score(y_test[cold_mask], test_pred[cold_mask])
    print(f"  AUC: {cold_auc:.4f}")


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
section("5. FEATURE IMPORTANCE")

importance = model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': numeric_features, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False)

print("Top 15 features:")
for _, row in feat_imp.head(15).iterrows():
    bar = '█' * max(1, int(row['importance'] / feat_imp['importance'].max() * 30))
    print(f"  {row['feature']:35s} {row['importance']:>10.0f} {bar}")

feat_imp.to_csv(os.path.join(RANK_DIR, "feature_importance.csv"), index=False)


# ============================================================
# 6. SAVE
# ============================================================
section("6. SAVE")

model.save_model(os.path.join(RANK_DIR, "lgbm_ranker.txt"))

test_df_copy[['visitorid', 'itemid', 'label', 'pred']].to_parquet(
    os.path.join(RANK_DIR, "test_predictions.parquet"), index=False)

stats = {
    'val_auc': round(val_auc, 4), 'test_auc': round(test_auc, 4),
    'val_ndcg20': round(val_ndcg, 4), 'test_ndcg20': round(test_ndcg, 4),
    'hit_rates': {str(k): round(v, 4) for k, v in hit_rates.items()},
    'n_features': len(numeric_features),
    'best_iteration': model.best_iteration,
}
with open(os.path.join(RANK_DIR, "ranking_stats.json"), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✅ lgbm_ranker.txt")
print(f"✅ test_predictions.parquet")
print(f"✅ ranking_stats.json")
print(f"\n✅ STEP 8 COMPLETE")