"""
==============================================================================
MODULE 2: Data Understanding & Preprocessing Pipeline
BCI606 - Data Mining Mini Project
Domain: Digital Payment Fraud Detection
==============================================================================

WHY THIS MODULE MATTERS (Statistical Context):
    In fraud detection, preprocessing is NOT a formality -- it is the most
    critical step. The raw data contains heterogeneous scales (amounts in
    thousands vs. binary flags), severe class imbalance (~1.5% fraud), and
    a leaky feature that would invalidate any model. Every decision below
    is justified with the underlying math.

Dependencies: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
Install: pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Configure Plots ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ==============================================================================
# STEP 1: LOAD DATA & INITIAL INSPECTION
# ==============================================================================
print("=" * 70)
print("STEP 1: Loading Data & Initial Inspection")
print("=" * 70)

# We combine train and test for a unified preprocessing view.
# The dataset provider split by time (before/after Oct 2023).
# For our BCI606 modules (pattern mining, classification, clustering),
# we will work on the combined data and do our own stratified split later.
df_train = pd.read_csv("datasets/transactions_train.csv")
df_test = pd.read_csv("datasets/transactions_test.csv")

print(f"Training set shape : {df_train.shape}")
print(f"Test set shape     : {df_test.shape}")

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Combined shape     : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ==============================================================================
# STEP 2: ATTRIBUTE TYPE IDENTIFICATION
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Data Mining distinguishes attribute types because different algorithms
# require different mathematical operations:
#   - Nominal: Categories with NO ordering (e.g., payment_channel).
#              Only equality (==) is defined. Mode is the only valid central tendency.
#   - Ordinal: Categories WITH a natural order (e.g., credit_score_band 1-5).
#              Median is meaningful, but arithmetic mean is not.
#   - Numeric (Ratio/Interval): Continuous values where +, -, *, / are valid.
#              Mean, variance, std dev are all meaningful.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Attribute Type Identification")
print("=" * 70)

# Programmatic type detection
attribute_types = {
    # --- Identifiers (to be DROPPED -- not features) ---
    'transaction_id':               'Identifier (Drop)',
    'customer_id':                  'Identifier (Drop)',
    'merchant_id':                  'Identifier (Drop)',
    # --- Temporal ---
    'transaction_time':             'Temporal (Parse -> extract features)',
    # --- Nominal (Categorical, unordered) ---
    'payment_channel':              'Nominal',
    'device_type':                  'Nominal',
    # --- Ordinal (Categorical, ordered) ---
    'credit_score_band':            'Ordinal (1=lowest, 5=highest)',
    'kyc_level':                    'Ordinal (1=weakest, 3=strongest)',
    # --- Binary ---
    'is_international':             'Binary (Nominal subset)',
    'is_fraud':                     'Binary -- TARGET VARIABLE',
    # --- Numeric (Continuous / Ratio scale) ---
    'account_age_days':             'Numeric (Ratio)',
    'avg_monthly_spend':            'Numeric (Ratio)',
    'merchant_risk_score':          'Numeric (Ratio, 0-1)',
    'transaction_amount':           'Numeric (Ratio)',
    'ip_risk_score':                'Numeric (Ratio, 0-1)',
    'txn_count_1h':                 'Numeric (Discrete count)',
    'txn_count_24h':                'Numeric (Discrete count)',
    'failed_txn_count_24h':         'Numeric (Discrete count)',
    'geo_distance_from_last_txn':   'Numeric (Ratio)',
    'amount_deviation_from_user_mean': 'Numeric (Ratio)',
    # --- LEAKY FEATURE ---
    'post_auth_risk_score':         'Numeric -- LEAKY (Drop!)',
}

print("\nAttribute Type Classification:")
print("-" * 55)
for col, atype in attribute_types.items():
    print(f"  {col:40s} -> {atype}")

print(f"\nDataFrame dtypes:\n{df.dtypes}")

# ==============================================================================
# STEP 3: CENTRAL TENDENCIES & DISPERSION
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# For numeric features, we compute:
#   Mean (μ)      = (1/N) Σ xᵢ             — sensitive to outliers
#   Median        = middle value            — robust to outliers
#   Variance (σ²) = (1/N) Σ (xᵢ - μ)²      — average squared deviation
#   Std Dev (σ)   = √(σ²)                  — in original units
#   Skewness      = E[(X-μ)³] / σ³         — asymmetry of distribution
#
# WHY THIS MATTERS FOR FRAUD:
#   If transaction_amount has high positive skew, most transactions are small
#   but a few are extremely large → those outliers could be fraud OR legitimate
#   high-value purchases. We must NOT blindly remove them.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Central Tendencies & Dispersion")
print("=" * 70)

numeric_cols = [
    'account_age_days', 'avg_monthly_spend', 'merchant_risk_score',
    'transaction_amount', 'ip_risk_score', 'txn_count_1h', 'txn_count_24h',
    'failed_txn_count_24h', 'geo_distance_from_last_txn',
    'amount_deviation_from_user_mean'
]

stats_df = df[numeric_cols].describe().T
stats_df['variance'] = df[numeric_cols].var()
stats_df['skewness'] = df[numeric_cols].skew()
stats_df['kurtosis'] = df[numeric_cols].kurtosis()

print("\nDescriptive Statistics (Numeric Features):")
print(stats_df[['mean', '50%', 'std', 'variance', 'skewness', 'kurtosis']].round(4).to_string())

# Categorical feature modes
print("\nCategorical Feature Modes:")
for col in ['payment_channel', 'device_type', 'credit_score_band', 'kyc_level']:
    mode_val = df[col].mode()[0]
    print(f"  {col:25s} -> Mode: {mode_val}  (Freq: {df[col].value_counts().iloc[0]})")

# ==============================================================================
# STEP 4: MISSING VALUE ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Missing Value Analysis")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'Missing Count': missing, 'Percent': missing_pct})
print(missing_report[missing_report['Missing Count'] > 0].to_string()
      if missing.sum() > 0
      else "  [OK] No missing values detected in any column.")

# ==============================================================================
# STEP 5: DROP IDENTIFIERS & LEAKY FEATURE
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# post_auth_risk_score is generated AFTER the fraud decision is made.
# Including it would cause DATA LEAKAGE: the model sees future information
# that would not be available at prediction time. This inflates metrics
# but produces a model that is useless in production.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Dropping Identifiers & Leaky Feature")
print("=" * 70)

cols_to_drop = ['transaction_id', 'customer_id', 'merchant_id', 'post_auth_risk_score']
df.drop(columns=cols_to_drop, inplace=True)
print(f"  Dropped: {cols_to_drop}")
print(f"  Remaining columns: {df.shape[1]}")

# ==============================================================================
# STEP 6: TEMPORAL FEATURE ENGINEERING
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Raw datetime strings cannot be fed into ML models. We extract:
#   - hour_of_day: Fraudsters tend to operate in off-peak hours (2-5 AM)
#   - day_of_week: Weekend vs weekday patterns differ
#   - month: Captures temporal drift (fraud patterns shift after month 6)
# These become new NUMERIC features. The original string is then dropped.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Temporal Feature Engineering")
print("=" * 70)

df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['hour_of_day'] = df['transaction_time'].dt.hour
df['day_of_week'] = df['transaction_time'].dt.dayofweek   # 0=Mon, 6=Sun
df['month'] = df['transaction_time'].dt.month
df.drop(columns=['transaction_time'], inplace=True)

print("  Extracted: hour_of_day, day_of_week, month")
print(f"  Shape after engineering: {df.shape}")

# ==============================================================================
# STEP 7: ENCODE CATEGORICAL VARIABLES
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Nominal features (payment_channel, device_type) have NO ordering.
# One-Hot Encoding creates binary indicator columns:
#   payment_channel=card → [1, 0, 0, 0]
#   payment_channel=upi  → [0, 1, 0, 0]
# This avoids imposing a false ordinal relationship (card < upi < wallet).
#
# Ordinal features (credit_score_band, kyc_level) already have integer codes
# that respect their natural ordering, so we keep them as-is.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Encoding Categorical Variables")
print("=" * 70)

# One-hot encode nominal features (drop_first=True to avoid multicollinearity)
df = pd.get_dummies(df, columns=['payment_channel', 'device_type'], drop_first=True)
print(f"  One-hot encoded: payment_channel, device_type")
print(f"  Shape after encoding: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ==============================================================================
# STEP 8: OUTLIER ANALYSIS (Domain-Aware)
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# The IQR method flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
# 
# CRITICAL DECISION FOR FRAUD DETECTION:
#   We DO NOT remove outliers. In fraud detection, outliers ARE the signal.
#   A transaction_amount of $50,000 when the user mean is $200 is exactly
#   the kind of anomaly a fraud model must learn. Removing it destroys
#   the very patterns we are trying to capture.
#
#   Instead, we REPORT outlier counts for awareness and rely on ROBUST
#   SCALING (Z-score / StandardScaler) to reduce their magnitude without
#   eliminating them.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Outlier Analysis (Report Only -- NOT Removing)")
print("=" * 70)

outlier_report = {}
for col in numeric_cols:
    if col not in df.columns:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {'Lower Bound': round(lower, 2),
                           'Upper Bound': round(upper, 2),
                           'Outlier Count': n_outliers,
                           'Outlier %': round(n_outliers / len(df) * 100, 2)}

outlier_df = pd.DataFrame(outlier_report).T
print(outlier_df.to_string())
print("\n  [!] Decision: Outliers RETAINED (they are potential fraud signals).")

# ==============================================================================
# STEP 9: NORMALIZATION / SCALING
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Two primary methods:
#
# (A) Min-Max Scaling:  X' = (X - Xmin) / (Xmax - Xmin)
#     Maps to [0, 1]. Preserves original distribution shape.
#     PROBLEM: Extremely sensitive to outliers. A single $1M transaction
#     compresses all normal transactions into a tiny range near 0.
#
# (B) Z-Score (StandardScaler):  X' = (X - μ) / σ
#     Centers at 0, unit variance. Outliers get large |z| values but
#     do NOT compress the rest of the data.
#
# CHOICE: Z-Score (StandardScaler)
#   Because our fraud features have heavy-tailed distributions with
#   meaningful outliers, Z-score is the correct choice. It preserves the
#   relative extremity of outliers as information, rather than squashing them.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Normalization (Z-Score / StandardScaler)")
print("=" * 70)

# Only scale continuous numeric features (not binary/one-hot/ordinal)
cols_to_scale = [
    'account_age_days', 'avg_monthly_spend', 'merchant_risk_score',
    'transaction_amount', 'ip_risk_score', 'geo_distance_from_last_txn',
    'amount_deviation_from_user_mean', 'txn_count_1h', 'txn_count_24h',
    'failed_txn_count_24h'
]

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("  Applied StandardScaler (Z-score) to continuous features.")
print(f"  Scaled columns: {cols_to_scale}")
print(f"\n  Post-scaling verification (should be ~0 mean, ~1 std):")
print(df[cols_to_scale].describe().loc[['mean', 'std']].round(4).to_string())

# ==============================================================================
# STEP 10: CLASS IMBALANCE ANALYSIS & SMOTE
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Class imbalance is THE defining challenge of fraud detection.
# If fraud is 1.5% of data, a naive model predicting "Not Fraud" always
# achieves 98.5% accuracy — yet catches ZERO fraud. Accuracy is meaningless.
#
# SMOTE (Synthetic Minority Over-sampling Technique):
#   For each minority sample xᵢ, find its k nearest neighbors in feature
#   space. Create a synthetic sample along the line segment between xᵢ
#   and a randomly chosen neighbor:
#       x_new = xᵢ + λ * (x_neighbor - xᵢ),   λ ∈ [0, 1]
#
#   This generates plausible new fraud examples, not just duplicates.
#   We apply SMOTE ONLY to training data to prevent data leakage.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 10: Class Imbalance Analysis & SMOTE Resampling")
print("=" * 70)

print("\n  Class Distribution (BEFORE resampling):")
class_counts = df['is_fraud'].value_counts()
print(f"    Not Fraud (0): {class_counts[0]:>8,d}  ({class_counts[0]/len(df)*100:.2f}%)")
print(f"    Fraud     (1): {class_counts[1]:>8,d}  ({class_counts[1]/len(df)*100:.2f}%)")
print(f"    Imbalance Ratio: 1:{class_counts[0]//class_counts[1]}")

# Separate features and target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Train-test split BEFORE SMOTE (critical to avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train/Test Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")
print(f"  Train fraud rate: {y_train.mean()*100:.2f}%")

# Apply SMOTE only to training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\n  Class Distribution (AFTER SMOTE on training set):")
resampled_counts = pd.Series(y_train_resampled).value_counts()
print(f"    Not Fraud (0): {resampled_counts[0]:>8,d}")
print(f"    Fraud     (1): {resampled_counts[1]:>8,d}")
print(f"    Ratio: 1:1 (balanced) [OK]")

# ==============================================================================
# STEP 11: SAVE PREPROCESSED DATA
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 11: Saving Preprocessed Data")
print("=" * 70)

# Save for subsequent modules
train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
train_resampled['is_fraud'] = y_train_resampled.values

test_set = pd.DataFrame(X_test, columns=X_test.columns)
test_set['is_fraud'] = y_test.values

train_resampled.to_csv("datasets/preprocessed_train.csv", index=False)
test_set.to_csv("datasets/preprocessed_test.csv", index=False)

# Also save the full preprocessed (unbalanced) dataset for clustering/pattern mining
df.to_csv("datasets/preprocessed_full.csv", index=False)

print("  Saved: datasets/preprocessed_train.csv  (SMOTE-balanced training set)")
print("  Saved: datasets/preprocessed_test.csv    (untouched test set)")
print("  Saved: datasets/preprocessed_full.csv    (full preprocessed, unbalanced)")

# ==============================================================================
# STEP 12: VISUALIZATION — Distribution & Correlation
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 12: Generating Visualizations")
print("=" * 70)

# --- 12a: Class Distribution Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Class Distribution (Original)', fontsize=13, fontweight='bold')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[0].set_ylabel('Count')

resampled_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[1].set_title('Class Distribution (After SMOTE)', fontsize=13, fontweight='bold')
axes[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig("datasets/class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: datasets/class_distribution.png")

# --- 12b: Correlation Heatmap (top features) ---
fig, ax = plt.subplots(figsize=(14, 10))
corr_cols = cols_to_scale + ['is_fraud', 'hour_of_day', 'month']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("datasets/correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: datasets/correlation_heatmap.png")

print("\n" + "=" * 70)
print("MODULE 2 COMPLETE [OK]")
print("=" * 70)
print(f"  Final training set (balanced): {X_train_resampled.shape}")
print(f"  Final test set (original):     {X_test.shape}")
print(f"  Total features:                {X_train_resampled.shape[1]}")
