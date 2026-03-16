"""
STEP 1E: APPLY SESSION (30min) + FULL EDA ON FINAL CLEAN DATA
Assumes: events df đã K-core filtered (K=5) trong memory.

Output:
  - Session columns gắn vào events
  - Full EDA: mọi thứ cần cho feature engineering + embedding decisions
  - Figures saved
  - events FINAL saved
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

FIG_DIR = "outputs/figures"
OUT_DIR = "data/processed"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

stats = {}

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ==============================================================
# PART 0: APPLY SESSION THRESHOLD = 30 MIN
# ==============================================================
section("0. APPLY SESSION THRESHOLD (30 min)")

SESSION_THRESHOLD_MIN = 20

if 'timestamp_dt' not in events.columns:
    events['timestamp_dt'] = pd.to_datetime(events['timestamp'], unit='ms')

events = events.sort_values(['visitorid', 'timestamp_dt']).reset_index(drop=True)

events['_prev_ts'] = events.groupby('visitorid')['timestamp_dt'].shift(1)
events['_gap_sec'] = (events['timestamp_dt'] - events['_prev_ts']).dt.total_seconds()

threshold_sec = SESSION_THRESHOLD_MIN * 60
events['_new_session'] = (events['_gap_sec'] > threshold_sec) | events['_gap_sec'].isna()
events['session_id'] = events.groupby('visitorid')['_new_session'].cumsum()

# Session columns
events['session_position'] = events.groupby(['visitorid', 'session_id']).cumcount()

session_length = events.groupby(['visitorid', 'session_id']).size().reset_index(name='session_length')
events = events.merge(session_length, on=['visitorid', 'session_id'], how='left')

events['session_unique_items'] = events.groupby(['visitorid', 'session_id'])['itemid'].transform('nunique')

events['session_has_purchase'] = events.groupby(['visitorid', 'session_id'])['event'].transform(
    lambda x: (x == 'transaction').any().astype(int)
)
events['session_has_cart'] = events.groupby(['visitorid', 'session_id'])['event'].transform(
    lambda x: (x == 'addtocart').any().astype(int)
)

session_start = events.groupby(['visitorid', 'session_id'])['timestamp_dt'].transform('min')
events['session_duration_sec'] = (
    events.groupby(['visitorid', 'session_id'])['timestamp_dt'].transform('max') - session_start
).dt.total_seconds()
events['time_in_session_sec'] = (events['timestamp_dt'] - session_start).dt.total_seconds()

# Time features
events['hour'] = events['timestamp_dt'].dt.hour
events['day_of_week'] = events['timestamp_dt'].dt.dayofweek
events['date'] = events['timestamp_dt'].dt.date

events = events.drop(columns=['_prev_ts', '_gap_sec', '_new_session'])

n_sessions = events.groupby(['visitorid', 'session_id']).ngroups
print(f"Session threshold: {SESSION_THRESHOLD_MIN} min")
print(f"Sessions: {n_sessions:,}")
print(f"Avg events/session: {len(events)/n_sessions:.1f}")
print(f"Columns: {list(events.columns)}")


# ==============================================================
# PART 1: BASIC STATS
# ==============================================================
section("1. BASIC STATS — FINAL CLEAN DATA")

N = len(events)
N_USERS = events['visitorid'].nunique()
N_ITEMS = events['itemid'].nunique()
DENSITY = N / (N_USERS * N_ITEMS) * 100

n_views = (events['event'] == 'view').sum()
n_carts = (events['event'] == 'addtocart').sum()
n_trans = (events['event'] == 'transaction').sum()
n_buyers = events[events['event'] == 'transaction']['visitorid'].nunique()

stats.update({
    'n_events': int(N), 'n_users': int(N_USERS), 'n_items': int(N_ITEMS),
    'density_pct': round(DENSITY, 4),
    'n_views': int(n_views), 'n_carts': int(n_carts), 'n_trans': int(n_trans),
    'n_buyers': int(n_buyers), 'n_sessions': int(n_sessions),
})

print(f"""
  Events:       {N:>10,}
  Users:        {N_USERS:>10,}
  Items:        {N_ITEMS:>10,}
  Density:      {DENSITY:>10.4f}%
  
  Views:        {n_views:>10,} ({n_views/N*100:.1f}%)
  AddToCart:    {n_carts:>10,} ({n_carts/N*100:.1f}%)
  Transactions: {n_trans:>10,} ({n_trans/N*100:.1f}%)
  Buyers:       {n_buyers:>10,} ({n_buyers/N_USERS*100:.1f}% of users)
  Sessions:     {n_sessions:>10,}
""")


# ==============================================================
# PART 2: CONVERSION FUNNEL
# ==============================================================
section("2. CONVERSION FUNNEL")

view_to_cart = n_carts / max(n_views, 1) * 100
cart_to_trans = n_trans / max(n_carts, 1) * 100
view_to_trans = n_trans / max(n_views, 1) * 100

print(f"  View → Cart:        {view_to_cart:.2f}%")
print(f"  Cart → Transaction: {cart_to_trans:.2f}%")
print(f"  View → Transaction: {view_to_trans:.2f}%")

stats.update({
    'funnel_view_cart': round(view_to_cart, 2),
    'funnel_cart_trans': round(cart_to_trans, 2),
    'funnel_view_trans': round(view_to_trans, 2),
})


# ==============================================================
# PART 3: USER-ITEM PAIR OUTCOMES (Label Design Input)
# ==============================================================
section("3. USER-ITEM PAIR OUTCOMES")

user_item = events.groupby(['visitorid', 'itemid']).agg(
    n_views=('event', lambda x: (x == 'view').sum()),
    n_carts=('event', lambda x: (x == 'addtocart').sum()),
    n_trans=('event', lambda x: (x == 'transaction').sum()),
    total_events=('event', 'count'),
    first_ts=('timestamp_dt', 'min'),
    last_ts=('timestamp_dt', 'max'),
    n_sessions_involved=('session_id', 'nunique'),
).reset_index()

purchased = (user_item['n_trans'] > 0).sum()
carted_only = ((user_item['n_carts'] > 0) & (user_item['n_trans'] == 0)).sum()
viewed_only = ((user_item['n_carts'] == 0) & (user_item['n_trans'] == 0)).sum()
total_pairs = len(user_item)

stats.update({
    'n_pairs': int(total_pairs),
    'n_pairs_purchased': int(purchased),
    'n_pairs_carted': int(carted_only),
    'n_pairs_viewed': int(viewed_only),
})

print(f"  Total user-item pairs: {total_pairs:,}")
print(f"  Purchased:       {purchased:>10,} ({purchased/total_pairs*100:.2f}%)")
print(f"  Carted only:     {carted_only:>10,} ({carted_only/total_pairs*100:.2f}%)")
print(f"  Viewed only:     {viewed_only:>10,} ({viewed_only/total_pairs*100:.2f}%)")

# Repeat views before action
purchased_pairs = user_item[user_item['n_trans'] > 0]
carted_pairs = user_item[(user_item['n_carts'] > 0) & (user_item['n_trans'] == 0)]
viewed_pairs = user_item[(user_item['n_carts'] == 0) & (user_item['n_trans'] == 0)]

print(f"\n  Avg views before purchase: {purchased_pairs['n_views'].mean():.1f}")
print(f"  Avg views for carted-only: {carted_pairs['n_views'].mean():.1f}")
print(f"  Avg views for view-only:   {viewed_pairs['n_views'].mean():.1f}")
print(f"  → Repeat view count có thể là strong feature cho ranking")

print(f"\n  Sessions involved per pair:")
print(f"    Purchased pairs: avg {purchased_pairs['n_sessions_involved'].mean():.1f} sessions")
print(f"    Carted pairs:    avg {carted_pairs['n_sessions_involved'].mean():.1f} sessions")
print(f"    Viewed pairs:    avg {viewed_pairs['n_sessions_involved'].mean():.1f} sessions")
print(f"  → Multi-session interest = stronger signal")


# ==============================================================
# PART 4: USER ANALYSIS
# ==============================================================
section("4. USER ANALYSIS")

user_stats_df = events.groupby('visitorid').agg(
    total_events=('event', 'count'),
    n_views=('event', lambda x: (x == 'view').sum()),
    n_carts=('event', lambda x: (x == 'addtocart').sum()),
    n_trans=('event', lambda x: (x == 'transaction').sum()),
    unique_items=('itemid', 'nunique'),
    n_sessions=('session_id', 'nunique'),
    first_event=('timestamp_dt', 'min'),
    last_event=('timestamp_dt', 'max'),
).reset_index()

user_stats_df['has_purchase'] = user_stats_df['n_trans'] > 0
user_stats_df['lifespan_days'] = (user_stats_df['last_event'] - user_stats_df['first_event']).dt.days
user_stats_df['purchase_rate'] = user_stats_df['n_trans'] / user_stats_df['total_events']
user_stats_df['events_per_session'] = user_stats_df['total_events'] / user_stats_df['n_sessions']

print("User activity percentiles:")
for p, q in [('P25', 0.25), ('P50', 0.5), ('P75', 0.75), ('P90', 0.9), ('P95', 0.95)]:
    val = user_stats_df['total_events'].quantile(q)
    print(f"  {p}: {val:.0f} events")

# User segments
n_browsers = (~user_stats_df['has_purchase'] & (user_stats_df['n_carts'] == 0)).sum()
n_carters = (~user_stats_df['has_purchase'] & (user_stats_df['n_carts'] > 0)).sum()

print(f"\nUser types:")
print(f"  Browsers (view only): {n_browsers:>8,} ({n_browsers/N_USERS*100:.1f}%)")
print(f"  Carters (cart, no buy):{n_carters:>8,} ({n_carters/N_USERS*100:.1f}%)")
print(f"  Buyers:               {n_buyers:>8,} ({n_buyers/N_USERS*100:.1f}%)")

# Returning vs one-time
returning = (user_stats_df['lifespan_days'] > 0).sum()
one_time = (user_stats_df['lifespan_days'] == 0).sum()
print(f"\n  Single-day users: {one_time:>8,} ({one_time/N_USERS*100:.1f}%)")
print(f"  Returning users:  {returning:>8,} ({returning/N_USERS*100:.1f}%)")

if returning > 0:
    ret = user_stats_df[user_stats_df['lifespan_days'] > 0]
    print(f"  Returning lifespan: median {ret['lifespan_days'].median():.0f} days")

stats.update({
    'pct_browsers': round(n_browsers/N_USERS*100, 1),
    'pct_carters': round(n_carters/N_USERS*100, 1),
    'pct_buyers': round(n_buyers/N_USERS*100, 1),
    'pct_returning': round(returning/N_USERS*100, 1),
    'median_events_per_user': float(user_stats_df['total_events'].median()),
    'median_sessions_per_user': float(user_stats_df['n_sessions'].median()),
})

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(user_stats_df['total_events'].clip(upper=50), bins=50,
             color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Events per User')
axes[0].set_title('User Activity Distribution', fontweight='bold')

axes[1].hist(user_stats_df['n_sessions'].clip(upper=20), bins=20,
             color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Sessions per User')
axes[1].set_title('Sessions per User', fontweight='bold')

# Buyer rate by activity quantile
user_stats_df['activity_q'] = pd.qcut(user_stats_df['total_events'], q=5, labels=False, duplicates='drop')
n_bins = user_stats_df['activity_q'].nunique()
bin_labels = [f'Q{i+1}' for i in range(n_bins)]
user_stats_df['activity_q'] = user_stats_df['activity_q'].map(dict(enumerate(bin_labels)))
buyer_by_q = user_stats_df.groupby('activity_q', observed=True)['has_purchase'].mean() * 100
axes[2].bar(range(len(buyer_by_q)), buyer_by_q.values, color='#2ecc71', edgecolor='white')
axes[2].set_xticks(range(len(buyer_by_q)))
axes[2].set_xticklabels(buyer_by_q.index)
axes[2].set_ylabel('% Buyers')
axes[2].set_title('Purchase Rate by Activity Quintile', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_01_users.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"💾 {FIG_DIR}/eda_01_users.png")


# ==============================================================
# PART 5: ITEM ANALYSIS
# ==============================================================
section("5. ITEM ANALYSIS")

item_stats_df = events.groupby('itemid').agg(
    total_events=('event', 'count'),
    n_views=('event', lambda x: (x == 'view').sum()),
    n_carts=('event', lambda x: (x == 'addtocart').sum()),
    n_trans=('event', lambda x: (x == 'transaction').sum()),
    unique_visitors=('visitorid', 'nunique'),
).reset_index()

item_stats_df['conversion_rate'] = item_stats_df['n_trans'] / item_stats_df['n_views'].clip(lower=1)
item_stats_df['cart_rate'] = item_stats_df['n_carts'] / item_stats_df['n_views'].clip(lower=1)
item_stats_df['has_transaction'] = item_stats_df['n_trans'] > 0

items_with_trans = item_stats_df['has_transaction'].sum()

print(f"  Items with ≥1 transaction: {items_with_trans:,} ({items_with_trans/N_ITEMS*100:.1f}%)")
print(f"  Items view-only: {N_ITEMS - items_with_trans:,}")

# Long tail
sorted_pop = item_stats_df['total_events'].sort_values(ascending=False)
cumsum = sorted_pop.cumsum()
total = sorted_pop.sum()

print(f"\n  Long tail:")
for p in [1, 5, 10, 20]:
    n_top = max(1, int(len(sorted_pop) * p / 100))
    pct_int = cumsum.iloc[n_top-1] / total * 100
    print(f"    Top {p:2d}% items → {pct_int:.1f}% events")

# Gini
n_arr = np.arange(1, len(sorted_pop)+1) / len(sorted_pop)
c_arr = cumsum.values / total
gini = 1 - 2 * np.trapz(c_arr, n_arr)
print(f"  Gini: {gini:.4f}")
stats['gini_item'] = round(gini, 4)

# Conversion rate for items with enough views
items_10v = item_stats_df[item_stats_df['n_views'] >= 10]
if len(items_10v) > 0:
    print(f"\n  Conversion rate (items ≥10 views, n={len(items_10v):,}):")
    print(f"    Mean:   {items_10v['conversion_rate'].mean():.4f}")
    print(f"    Median: {items_10v['conversion_rate'].median():.4f}")
    print(f"    P90:    {items_10v['conversion_rate'].quantile(0.9):.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ip_counts = item_stats_df['total_events'].value_counts().sort_index()
axes[0].scatter(ip_counts.index, ip_counts.values, s=3, alpha=0.5, color='steelblue')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('Events per Item')
axes[0].set_title('Item Popularity (log-log)', fontweight='bold')

axes[1].plot(n_arr, c_arr, color='coral', linewidth=2)
axes[1].plot([0,1],[0,1], 'k--', alpha=0.3)
axes[1].set_xlabel('Proportion of Items')
axes[1].set_ylabel('Proportion of Events')
axes[1].set_title(f'Lorenz Curve (Gini={gini:.3f})', fontweight='bold')

if len(items_10v) > 0:
    axes[2].hist(items_10v['conversion_rate'].clip(upper=0.2), bins=50,
                 color='#2ecc71', edgecolor='white', alpha=0.8)
    axes[2].set_xlabel('Conversion Rate')
    axes[2].set_title('Item Conversion Rate (≥10 views)', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_02_items.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"💾 {FIG_DIR}/eda_02_items.png")


# ==============================================================
# PART 6: SESSION ANALYSIS
# ==============================================================
section("6. SESSION ANALYSIS")

sess_df = events.groupby(['visitorid', 'session_id']).agg(
    n_events=('event', 'count'),
    n_items=('itemid', 'nunique'),
    has_purchase=('session_has_purchase', 'first'),
    has_cart=('session_has_cart', 'first'),
    duration_sec=('session_duration_sec', 'first'),
).reset_index()

print(f"  Total sessions: {len(sess_df):,}")
print(f"  Purchase sessions: {sess_df['has_purchase'].sum():,} ({sess_df['has_purchase'].mean()*100:.2f}%)")
print(f"  Cart sessions: {sess_df['has_cart'].sum():,} ({sess_df['has_cart'].mean()*100:.2f}%)")

print(f"\n  Session length (events):")
for p, q in [('P25', 0.25), ('P50', 0.5), ('P75', 0.75), ('P90', 0.9)]:
    print(f"    {p}: {sess_df['n_events'].quantile(q):.0f}")

print(f"\n  Session items:")
for p, q in [('P50', 0.5), ('P75', 0.75), ('P90', 0.9)]:
    print(f"    {p}: {sess_df['n_items'].quantile(q):.0f}")

active = sess_df[sess_df['duration_sec'] > 0]
if len(active) > 0:
    print(f"\n  Session duration (active sessions, n={len(active):,}):")
    for p, q in [('P50', 0.5), ('P75', 0.75), ('P90', 0.9)]:
        val = active['duration_sec'].quantile(q)
        print(f"    {p}: {val:.0f}s ({val/60:.1f}min)")

# Purchase sessions vs non-purchase sessions
purchase_sess = sess_df[sess_df['has_purchase'] == 1]
no_purchase_sess = sess_df[sess_df['has_purchase'] == 0]

print(f"\n  Purchase sessions avg events: {purchase_sess['n_events'].mean():.1f}")
print(f"  Non-purchase sessions avg events: {no_purchase_sess['n_events'].mean():.1f}")
print(f"  → Purchase sessions are longer → session_length is a feature")

stats.update({
    'median_session_events': float(sess_df['n_events'].median()),
    'pct_purchase_sessions': round(sess_df['has_purchase'].mean()*100, 2),
    'avg_events_purchase_session': round(purchase_sess['n_events'].mean(), 1),
    'avg_events_nonpurchase_session': round(no_purchase_sess['n_events'].mean(), 1),
})

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(sess_df['n_events'].clip(upper=20), bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Events per Session')
axes[0].set_title('Session Length', fontweight='bold')

if len(active) > 0:
    axes[1].hist((active['duration_sec']/60).clip(upper=60), bins=50, color='coral', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Duration (minutes)')
    axes[1].set_title('Session Duration', fontweight='bold')

# Purchase rate by session length
sess_df['length_bin'] = pd.cut(sess_df['n_events'], bins=[0,1,2,3,5,10,20,1000],
                                labels=['1','2','3','4-5','6-10','11-20','20+'])
pr_by_len = sess_df.groupby('length_bin', observed=True)['has_purchase'].mean() * 100
axes[2].bar(range(len(pr_by_len)), pr_by_len.values, color='#2ecc71', edgecolor='white')
axes[2].set_xticks(range(len(pr_by_len)))
axes[2].set_xticklabels(pr_by_len.index, fontsize=9)
axes[2].set_xlabel('Session Length (events)')
axes[2].set_ylabel('% Purchase')
axes[2].set_title('Purchase Rate by Session Length', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_03_sessions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"💾 {FIG_DIR}/eda_03_sessions.png")


# ==============================================================
# PART 7: TEMPORAL ANALYSIS
# ==============================================================
section("7. TEMPORAL ANALYSIS")

daily = events.groupby('date').agg(
    total=('event', 'count'),
    trans=('event', lambda x: (x == 'transaction').sum()),
).reset_index()
daily['conv_rate'] = daily['trans'] / daily['total'] * 100

hourly = events.groupby('hour').size()
dow = events.groupby('day_of_week').size()
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

print(f"  Daily events: mean={daily['total'].mean():.0f}, median={daily['total'].median():.0f}")
print(f"  Daily conversion rate: mean={daily['conv_rate'].mean():.3f}%")

print(f"\n  Peak hours: {hourly.nlargest(3).index.tolist()}")
print(f"  Peak days: {[dow_names[d] for d in dow.nlargest(3).index.tolist()]}")

# Conversion by hour
hourly_trans = events[events['event'] == 'transaction'].groupby('hour').size()
hourly_views = events[events['event'] == 'view'].groupby('hour').size()
hourly_conv = (hourly_trans / hourly_views * 100).fillna(0)

print(f"\n  Conversion rate by hour (top 5):")
for h in hourly_conv.nlargest(5).index:
    print(f"    {h:02d}:00 → {hourly_conv[h]:.2f}%")

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0,0].plot(range(len(daily)), daily['total'].values, linewidth=0.8, color='steelblue')
axes[0,0].set_title('Daily Events', fontweight='bold')
tick_step = max(1, len(daily)//10)
axes[0,0].set_xticks(range(0, len(daily), tick_step))
axes[0,0].set_xticklabels([str(daily['date'].iloc[i]) for i in range(0, len(daily), tick_step)],
                           rotation=45, fontsize=8)

axes[0,1].plot(range(len(daily)), daily['conv_rate'].values, linewidth=0.8, color='coral')
axes[0,1].set_title('Daily Conversion Rate %', fontweight='bold')
axes[0,1].set_xticks(range(0, len(daily), tick_step))
axes[0,1].set_xticklabels([str(daily['date'].iloc[i]) for i in range(0, len(daily), tick_step)],
                           rotation=45, fontsize=8)

axes[1,0].bar(range(24), [hourly.get(h,0) for h in range(24)], color='steelblue', edgecolor='white')
axes[1,0].set_xlabel('Hour')
axes[1,0].set_title('Hourly Pattern', fontweight='bold')

axes[1,1].bar(range(7), [dow.get(d,0) for d in range(7)], color='coral', edgecolor='white')
axes[1,1].set_xticks(range(7))
axes[1,1].set_xticklabels(dow_names)
axes[1,1].set_title('Day of Week', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/eda_04_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"💾 {FIG_DIR}/eda_04_temporal.png")


# ==============================================================
# PART 8: TEMPORAL SPLIT SIMULATION
# ==============================================================
section("8. TEMPORAL SPLIT (70/15/15)")

train_cut = events['timestamp_dt'].quantile(0.70)
val_cut = events['timestamp_dt'].quantile(0.85)

print(f"  Train cutoff: {train_cut.date()}")
print(f"  Val cutoff:   {val_cut.date()}")

mask_train = events['timestamp_dt'] < train_cut
mask_val = (events['timestamp_dt'] >= train_cut) & (events['timestamp_dt'] < val_cut)
mask_test = events['timestamp_dt'] >= val_cut

for name, mask in [('Train', mask_train), ('Val', mask_val), ('Test', mask_test)]:
    n = mask.sum()
    n_t = ((events['event'] == 'transaction') & mask).sum()
    print(f"  {name:5s}: {n:>10,} events ({n/N*100:.1f}%), {n_t:>6,} transactions")

# Warm/cold
train_users = set(events.loc[mask_train, 'visitorid'].unique())
train_items = set(events.loc[mask_train, 'itemid'].unique())

for name, mask in [('Val', mask_val), ('Test', mask_test)]:
    s_users = set(events.loc[mask, 'visitorid'].unique())
    s_items = set(events.loc[mask, 'itemid'].unique())
    warm_u = len(s_users & train_users)
    cold_u = len(s_users - train_users)
    warm_i = len(s_items & train_items)
    cold_i = len(s_items - train_items)
    
    n_split = mask.sum()
    ww = events[mask & events['visitorid'].isin(train_users) & events['itemid'].isin(train_items)].shape[0]
    
    print(f"\n  {name}: {len(s_users):,} users (warm:{warm_u:,} cold:{cold_u:,}), "
          f"{len(s_items):,} items (warm:{warm_i:,} cold:{cold_i:,})")
    print(f"    Warm-warm events: {ww:,} ({ww/max(n_split,1)*100:.1f}%)")

stats['train_cutoff'] = str(train_cut.date())
stats['val_cutoff'] = str(val_cut.date())


# ==============================================================
# PART 9: ITEM CATEGORY COVERAGE
# ==============================================================
section("9. ITEM CATEGORY COVERAGE")

try:
    props = pd.concat([
        pd.read_csv("data/raw/item_properties_part1.csv", dtype=str),
        pd.read_csv("data/raw/item_properties_part2.csv", dtype=str),
    ])
    
    cat_props = props[props['property'] == 'categoryid']
    cat_items = set(cat_props['itemid'].unique())
    our_items = set(events['itemid'].unique())
    overlap = len(cat_items & our_items)
    
    print(f"  Items with categoryid: {overlap:,} / {N_ITEMS:,} ({overlap/N_ITEMS*100:.1f}%)")
    
    # Available property
    avail_props = props[props['property'] == 'available']
    avail_items = set(avail_props['itemid'].unique())
    avail_overlap = len(avail_items & our_items)
    print(f"  Items with 'available' property: {avail_overlap:,} / {N_ITEMS:,} ({avail_overlap/N_ITEMS*100:.1f}%)")
    
    stats['pct_items_with_category'] = round(overlap/N_ITEMS*100, 1)
    
    del props  # free memory
except Exception as e:
    print(f"  Could not load item properties: {e}")


# ==============================================================
# PART 10: EMBEDDING SIGNAL SUMMARY
# ==============================================================
section("10. SIGNALS AVAILABLE FOR EMBEDDINGS")

# View sequences per user (for Item2Vec)
view_events = events[events['event'] == 'view']
user_view_seqs = view_events.groupby('visitorid')['itemid'].apply(list)
seq_lengths = user_view_seqs.apply(len)
users_with_2plus = (seq_lengths >= 2).sum()

print(f"  Item2Vec (view sequences):")
print(f"    Users with ≥2 views: {users_with_2plus:,}")
print(f"    Sequence length: median={seq_lengths.median():.0f}, P75={seq_lengths.quantile(0.75):.0f}, P90={seq_lengths.quantile(0.9):.0f}")

# Session-level sequences (for session-based Item2Vec)
session_seqs = view_events.groupby(['visitorid', 'session_id'])['itemid'].apply(list)
session_seq_lens = session_seqs.apply(len)
sessions_2plus = (session_seq_lens >= 2).sum()

print(f"\n  Session-level Item2Vec:")
print(f"    Sessions with ≥2 views: {sessions_2plus:,}")
print(f"    Sequence length: median={session_seq_lens.median():.0f}, P75={session_seq_lens.quantile(0.75):.0f}")

# Graph edges (for LightGCN)
print(f"\n  LightGCN (user-item graph):")
print(f"    Edges: {N:,}")
print(f"    Edge types: view({n_views:,}), cart({n_carts:,}), trans({n_trans:,})")
print(f"    → Multi-weight: view=0.1, cart=0.5, trans=1.0")

# Full sequences (for SASRec)
full_seqs = events.groupby('visitorid')['itemid'].apply(list)
full_lens = full_seqs.apply(len)

print(f"\n  SASRec (full event sequences):")
print(f"    Users with ≥3 events: {(full_lens >= 3).sum():,}")
print(f"    Sequence length: median={full_lens.median():.0f}, P75={full_lens.quantile(0.75):.0f}, P90={full_lens.quantile(0.9):.0f}")
print(f"    → Max sequence length for model: recommend {int(full_lens.quantile(0.9))}")

stats.update({
    'item2vec_users': int(users_with_2plus),
    'item2vec_median_seq': float(seq_lengths.median()),
    'sasrec_users': int((full_lens >= 3).sum()),
    'sasrec_recommended_maxlen': int(full_lens.quantile(0.9)),
    'lightgcn_edges': int(N),
})


# ==============================================================
# SAVE
# ==============================================================
section("SAVE")

# Save final clean events
save_cols = ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid',
             'session_id', 'session_position', 'session_length', 'session_unique_items',
             'session_has_purchase', 'session_has_cart', 'session_duration_sec',
             'time_in_session_sec', 'hour', 'day_of_week']
save_cols = [c for c in save_cols if c in events.columns]

events[save_cols].to_parquet(os.path.join(OUT_DIR, "events_final.parquet"), index=False)
print(f"✅ events_final.parquet: {len(events):,} rows × {len(save_cols)} cols")

# Save stats
with open(os.path.join(OUT_DIR, "eda_final_stats.json"), 'w') as f:
    json.dump(stats, f, indent=2, default=str)
print(f"✅ eda_final_stats.json")

# Save user-item pair analysis (will be useful for label design)
user_item.to_parquet(os.path.join(OUT_DIR, "user_item_pairs_raw.parquet"), index=False)
print(f"✅ user_item_pairs_raw.parquet: {len(user_item):,} pairs")

print(f"\n✅ FULL EDA COMPLETE — events df is FINAL")
print(f"   Next: Step 2 (label design + temporal split + save train/val/test)")# initial build
# initial build
# initial build
# initial build
