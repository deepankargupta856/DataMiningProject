"""
==============================================================================
MODULE 3: Pattern Mining (Apriori & FP-Growth)
BCI606 - Data Mining Mini Project
Domain: Digital Payment Fraud Detection
==============================================================================

UNDER THE HOOD -- Why Pattern Mining on Numerical Fraud Data?
    Association rule mining (Apriori, FP-Growth) was designed for
    "market basket" data: categorical items in transactions.
    Our fraud dataset is NUMERICAL. We must first DISCRETIZE continuous
    features into categorical bins to create "items" that can form
    itemsets. This is a deliberate design decision:

    Instead of: transaction_amount = 8543.21
    We create:  transaction_amount = "amt_HIGH"

    This lets us discover rules like:
      {amt_HIGH, ip_risk_HIGH, failed_txns_HIGH} => {is_fraud=YES}
      Support: 0.8%, Confidence: 72%, Lift: 42x

    Such rules reveal the behavioral RECIPE of a fraudster.

Dependencies: pandas, numpy, mlxtend, time, tracemalloc
==============================================================================
"""

import pandas as pd
import numpy as np
import time
import tracemalloc
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# STEP 1: LOAD THE RAW DATA (PRE-SCALED)
# ==============================================================================
# We reload the ORIGINAL combined data (not the z-score scaled version)
# because discretization needs the original value ranges.
# We will re-apply the same cleaning steps (drop IDs, leaky feature)
# but NOT scale -- instead we discretize.
# ==============================================================================
print("=" * 70)
print("STEP 1: Loading Raw Data for Discretization")
print("=" * 70)

df_train = pd.read_csv("datasets/transactions_train.csv")
df_test = pd.read_csv("datasets/transactions_test.csv")
df = pd.concat([df_train, df_test], ignore_index=True)
del df_train, df_test

# Drop identifiers and leaky feature
df.drop(columns=['transaction_id', 'customer_id', 'merchant_id',
                  'post_auth_risk_score', 'transaction_time'], inplace=True)

print(f"  Loaded {len(df):,} transactions, {df.shape[1]} columns")
print(f"  Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# ==============================================================================
# STEP 2: DISCRETIZE NUMERICAL FEATURES INTO CATEGORICAL BINS
# ==============================================================================
# UNDER THE HOOD -- Discretization Strategy:
#   We use DOMAIN-AWARE binning, not arbitrary equal-width bins.
#   Each feature is binned into semantically meaningful categories:
#
#   - transaction_amount: LOW / MEDIUM / HIGH / VERY_HIGH
#     (based on quartiles, with a special top-5% "VERY_HIGH" bin
#      because extreme amounts are a known fraud signal)
#
#   - ip_risk_score: LOW_RISK / MODERATE / HIGH_RISK
#     (risk scores have domain meaning at specific thresholds)
#
#   This creates "items" for our market-basket transformation.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Discretizing Features into Categorical Bins")
print("=" * 70)

df_disc = pd.DataFrame()

# --- transaction_amount ---
df_disc['amt'] = pd.cut(df['transaction_amount'],
    bins=[0, 1000, 3000, 6000, float('inf')],
    labels=['amt_LOW', 'amt_MED', 'amt_HIGH', 'amt_VHIGH'])

# --- ip_risk_score ---
df_disc['ip_risk'] = pd.cut(df['ip_risk_score'],
    bins=[0, 0.2, 0.4, 0.6, 1.0],
    labels=['ip_LOW', 'ip_MOD', 'ip_HIGH', 'ip_VHIGH'],
    include_lowest=True)

# --- merchant_risk_score ---
df_disc['merch_risk'] = pd.cut(df['merchant_risk_score'],
    bins=[0, 0.15, 0.3, 0.5, 1.0],
    labels=['mrisk_LOW', 'mrisk_MOD', 'mrisk_HIGH', 'mrisk_VHIGH'],
    include_lowest=True)

# --- geo_distance_from_last_txn ---
df_disc['geo_dist'] = pd.cut(df['geo_distance_from_last_txn'],
    bins=[0, 15, 40, 80, float('inf')],
    labels=['geo_NEAR', 'geo_MOD', 'geo_FAR', 'geo_VFAR'],
    include_lowest=True)

# --- failed_txn_count_24h ---
df_disc['failed_txns'] = pd.cut(df['failed_txn_count_24h'],
    bins=[-1, 0, 1, 2, float('inf')],
    labels=['fail_NONE', 'fail_LOW', 'fail_MOD', 'fail_HIGH'])

# --- txn_count_1h (velocity) ---
df_disc['velocity_1h'] = pd.cut(df['txn_count_1h'],
    bins=[-1, 0, 1, 3, float('inf')],
    labels=['vel1h_ZERO', 'vel1h_LOW', 'vel1h_MOD', 'vel1h_HIGH'])

# --- txn_count_24h ---
df_disc['velocity_24h'] = pd.cut(df['txn_count_24h'],
    bins=[-1, 2, 4, 6, float('inf')],
    labels=['vel24h_LOW', 'vel24h_MOD', 'vel24h_HIGH', 'vel24h_VHIGH'])

# --- account_age_days ---
df_disc['acct_age'] = pd.cut(df['account_age_days'],
    bins=[0, 180, 365, 730, float('inf')],
    labels=['acct_NEW', 'acct_6MO', 'acct_1YR', 'acct_OLD'],
    include_lowest=True)

# --- amount_deviation_from_user_mean ---
q75_dev = df['amount_deviation_from_user_mean'].quantile(0.75)
q95_dev = df['amount_deviation_from_user_mean'].quantile(0.95)
df_disc['amt_deviation'] = pd.cut(df['amount_deviation_from_user_mean'],
    bins=[-float('inf'), 500, q75_dev, q95_dev, float('inf')],
    labels=['dev_NORMAL', 'dev_MOD', 'dev_HIGH', 'dev_EXTREME'])

# --- Categorical features (already discrete) ---
df_disc['channel'] = 'ch_' + df['payment_channel'].astype(str)
df_disc['device'] = 'dev_' + df['device_type'].astype(str)
df_disc['intl'] = df['is_international'].map({0: 'domestic', 1: 'international'})
df_disc['kyc'] = 'kyc_' + df['kyc_level'].astype(str)
df_disc['credit_band'] = 'credit_' + df['credit_score_band'].astype(str)

# --- Target variable ---
df_disc['fraud'] = df['is_fraud'].map({0: 'legit', 1: 'FRAUD'})

print("  Discretized features:")
for col in df_disc.columns:
    unique_vals = df_disc[col].unique()
    print(f"    {col:18s}: {list(unique_vals[:6])}")

# ==============================================================================
# STEP 3: ONE-HOT ENCODE INTO BOOLEAN BASKET FORMAT
# ==============================================================================
# UNDER THE HOOD:
#   Apriori and FP-Growth require a BOOLEAN matrix where:
#     - Each row = one transaction
#     - Each column = one possible "item" (e.g., "amt_HIGH")
#     - True/False = whether that item is present
#
#   This is the "market basket" representation.
#   With 400K rows, we SAMPLE to make Apriori tractable.
#   FP-Growth is more efficient and can handle larger data.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating One-Hot Boolean Basket Matrix")
print("=" * 70)

# Create one-hot encoded basket
basket = pd.get_dummies(df_disc)

# Convert to boolean
basket = basket.astype(bool)

print(f"  Full basket shape: {basket.shape}")
print(f"  Number of unique items: {basket.shape[1]}")

# Stratified sample: keep ALL fraud rows + random sample of legit rows
# This ensures fraud patterns are well-represented
fraud_mask = df['is_fraud'] == 1
fraud_basket = basket[fraud_mask]
legit_basket = basket[~fraud_mask].sample(n=15000, random_state=42)
basket_sample = pd.concat([fraud_basket, legit_basket]).sample(frac=1, random_state=42)

print(f"  Sampled basket: {basket_sample.shape}")
print(f"    - All fraud rows: {fraud_mask.sum():,}")
print(f"    - Sampled legit:  15,000")
print(f"    - Total:          {len(basket_sample):,}")

# ==============================================================================
# STEP 4: RUN APRIORI ALGORITHM
# ==============================================================================
# UNDER THE HOOD -- Apriori Algorithm:
#   Core principle: "If an itemset is infrequent, all its supersets
#   are also infrequent" (Downward Closure / Anti-Monotone property).
#
#   Algorithm:
#     1. Scan DB to find all 1-itemsets with support >= min_support
#     2. Generate candidate 2-itemsets from frequent 1-itemsets
#     3. Scan DB again to count support of candidates
#     4. Prune infrequent, repeat for k+1 itemsets
#
#   Complexity: O(2^n) worst case, but pruning makes it practical.
#   WEAKNESS: Requires MULTIPLE full database scans (one per level k).
#   For 400K rows, this is expensive -- hence our sampling strategy.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Running Apriori Algorithm")
print("=" * 70)

# --- Apriori with timing and memory tracking ---
tracemalloc.start()
t_start = time.time()

freq_apriori = apriori(basket_sample, min_support=0.03, use_colnames=True,
                        max_len=3, low_memory=True, verbose=0)

t_apriori = time.time() - t_start
mem_apriori_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
tracemalloc.stop()

print(f"  Apriori completed in {t_apriori:.2f} seconds")
print(f"  Peak memory: {mem_apriori_peak:.2f} MB")
print(f"  Frequent itemsets found: {len(freq_apriori)}")
print(f"  Max itemset length: {freq_apriori['itemsets'].apply(len).max()}")

# ==============================================================================
# STEP 5: RUN FP-GROWTH ALGORITHM
# ==============================================================================
# UNDER THE HOOD -- FP-Growth Algorithm:
#   Addresses Apriori's main weakness (multiple DB scans).
#
#   Algorithm:
#     1. Scan DB ONCE to find frequent 1-itemsets
#     2. Scan DB a SECOND time to build a compressed FP-Tree
#        (a prefix tree where shared prefixes are merged)
#     3. Mine frequent patterns directly from the tree using
#        conditional pattern bases -- NO candidate generation
#
#   Only 2 DB scans (vs. k scans for Apriori at level k).
#   Far more memory-efficient for dense datasets.
#   Typically 5-10x faster than Apriori on real data.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Running FP-Growth Algorithm")
print("=" * 70)

tracemalloc.start()
t_start = time.time()

freq_fpgrowth = fpgrowth(basket_sample, min_support=0.03, use_colnames=True,
                          max_len=3, verbose=0)

t_fpgrowth = time.time() - t_start
mem_fpgrowth_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
tracemalloc.stop()

print(f"  FP-Growth completed in {t_fpgrowth:.2f} seconds")
print(f"  Peak memory: {mem_fpgrowth_peak:.2f} MB")
print(f"  Frequent itemsets found: {len(freq_fpgrowth)}")

speedup = t_apriori / t_fpgrowth if t_fpgrowth > 0 else float('inf')
print(f"\n  --- Speedup: FP-Growth is {speedup:.1f}x faster than Apriori ---")

# ==============================================================================
# STEP 6: GENERATE ASSOCIATION RULES
# ==============================================================================
# UNDER THE HOOD -- Metrics Explained:
#
#   Support(A->B) = P(A and B) = count(A,B) / N
#     "How common is this pattern in the dataset?"
#     Low support = rare rule (but rare can be important in fraud!)
#
#   Confidence(A->B) = P(B|A) = support(A,B) / support(A)
#     "Given A occurred, how likely is B?"
#     High confidence = strong predictive signal
#
#   Lift(A->B) = confidence(A->B) / support(B)
#     = P(A and B) / [P(A) * P(B)]
#     "How much MORE likely is B when A is present vs. by chance?"
#     Lift = 1: independent (no association)
#     Lift > 1: positive association (A promotes B)
#     Lift < 1: negative association (A suppresses B)
#
#   For fraud detection, we want rules where:
#     consequent = {FRAUD} AND lift >> 1
#     This means the antecedent conditions make fraud FAR more likely.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Generating Association Rules")
print("=" * 70)

# Use FP-Growth results (same itemsets, faster computation)
rules = association_rules(freq_fpgrowth, metric="lift", min_threshold=1.5,
                          num_itemsets=len(basket_sample))

print(f"  Total rules generated: {len(rules)}")

# Filter rules where FRAUD is in the consequent
fraud_rules = rules[rules['consequents'].apply(lambda x: 'fraud_FRAUD' in x)]
fraud_rules = fraud_rules.sort_values('lift', ascending=False)

print(f"  Rules with FRAUD as consequent: {len(fraud_rules)}")

# Display top fraud-predicting rules
print("\n  TOP 15 FRAUD-PREDICTING ASSOCIATION RULES:")
print("  " + "-" * 90)
print(f"  {'#':>3}  {'Antecedent':<50} {'Supp':>6} {'Conf':>6} {'Lift':>7}")
print("  " + "-" * 90)

for i, (_, row) in enumerate(fraud_rules.head(15).iterrows()):
    ant = ', '.join(sorted(row['antecedents']))
    supp = row['support']
    conf = row['confidence']
    lift = row['lift']
    print(f"  {i+1:>3}  {ant:<50} {supp:>6.4f} {conf:>6.2%} {lift:>7.2f}")

# ==============================================================================
# STEP 7: ALSO SHOW TOP GENERAL FREQUENT ITEMSETS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Top Frequent Itemsets (by Support)")
print("=" * 70)

freq_sorted = freq_fpgrowth.sort_values('support', ascending=False)

# Show itemsets of length >= 2
multi_item = freq_sorted[freq_sorted['itemsets'].apply(len) >= 2].head(15)
print(f"\n  TOP 15 MULTI-ITEM FREQUENT ITEMSETS:")
print("  " + "-" * 70)
print(f"  {'#':>3}  {'Itemset':<55} {'Support':>8}")
print("  " + "-" * 70)
for i, (_, row) in enumerate(multi_item.iterrows()):
    items = ', '.join(sorted(row['itemsets']))
    print(f"  {i+1:>3}  {items:<55} {row['support']:>8.4f}")

# ==============================================================================
# STEP 8: ALGORITHM COMPARISON TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Apriori vs FP-Growth Comparison")
print("=" * 70)

comparison = {
    'Metric': ['Execution Time (sec)', 'Peak Memory (MB)',
               'Frequent Itemsets Found', 'DB Scans Required',
               'Candidate Generation?', 'Data Structure'],
    'Apriori': [f'{t_apriori:.2f}', f'{mem_apriori_peak:.2f}',
                str(len(freq_apriori)), 'k (one per level)',
                'Yes (expensive)', 'Hash tree'],
    'FP-Growth': [f'{t_fpgrowth:.2f}', f'{mem_fpgrowth_peak:.2f}',
                  str(len(freq_fpgrowth)), '2 (constant)',
                  'No (tree mining)', 'FP-Tree (compressed)']
}
comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

# ==============================================================================
# STEP 9: SAVE RESULTS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving Results")
print("=" * 70)

# Save full rules
rules_out = fraud_rules.copy()
rules_out['antecedents'] = rules_out['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules_out['consequents'] = rules_out['consequents'].apply(lambda x: ', '.join(sorted(x)))
rules_out.to_csv("datasets/fraud_association_rules.csv", index=False)

# Save frequent itemsets
freq_out = freq_fpgrowth.copy()
freq_out['itemsets'] = freq_out['itemsets'].apply(lambda x: ', '.join(sorted(x)))
freq_out.to_csv("datasets/frequent_itemsets.csv", index=False)

print("  Saved: datasets/fraud_association_rules.csv")
print("  Saved: datasets/frequent_itemsets.csv")

print("\n" + "=" * 70)
print("MODULE 3 COMPLETE")
print("=" * 70)
