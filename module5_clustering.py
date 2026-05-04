"""
==============================================================================
MODULE 5: Clustering & Anomaly Detection
BCI606 - Data Mining Mini Project
Domain: Digital Payment Fraud Detection
==============================================================================

WHY THIS MODULE MATTERS:
    Unlike classification (supervised), clustering is UNSUPERVISED -- we do
    NOT give the algorithm the is_fraud labels. The goal is to discover
    natural groupings in the data and then check if those groupings correlate
    with fraud. This simulates real-world scenarios where labeled data is
    scarce or unavailable.

CLUSTERING ALGORITHMS (as per BCI606 rubric):
1. k-Means          -- Partitions data into k spherical clusters by minimizing
                       within-cluster sum of squares (inertia).
                       Objective: argmin_C sum_i min_j ||x_i - mu_j||^2
2. Hierarchical     -- Agglomerative (bottom-up): starts with n clusters,
   (Agglomerative)     merges closest pairs using Ward linkage (minimizes
                       variance increase). Produces a dendrogram.
3. DBSCAN           -- Density-based: clusters = dense regions separated by
                       sparse areas. Points in sparse regions are NOISE/OUTLIERS.
                       Key params: eps (neighborhood radius), min_samples.
                       UNIQUELY suited for fraud: fraudulent transactions may
                       form sparse outlier points that DBSCAN labels as noise.

DIMENSIONALITY REDUCTION (for visualization):
    With 20+ features, we cannot plot clusters directly. We use:
    - PCA (Principal Component Analysis): Linear projection that maximizes
      variance. Fast, preserves global structure. First 2 PCs capture the
      directions of maximum data spread.
    - t-SNE: Non-linear embedding that preserves LOCAL neighborhood structure.
      Better at revealing cluster separability but is stochastic and slow.

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
import time
warnings.filterwarnings('ignore')

# -- Configure Plots ----------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("=" * 70)
print("MODULE 5: Clustering & Anomaly Detection")
print("Algorithms: k-Means | Hierarchical | DBSCAN")
print("=" * 70)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("\n[STEP 1] Loading Preprocessed Dataset...")
try:
    df = pd.read_csv("datasets/preprocessed_full.csv")
    print(f"  Full Dataset Shape: {df.shape}")
except FileNotFoundError:
    print("[ERROR] preprocessed_full.csv not found. Run module2_preprocessing.py first.")
    exit()

y_true = df['is_fraud'].values
X = df.drop(columns=['is_fraud'])

print(f"  Features: {X.shape[1]}")
print(f"  Fraud rate: {y_true.mean()*100:.2f}%")

# ==============================================================================
# STEP 2: SUBSAMPLE FOR COMPUTATIONAL FEASIBILITY
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# 400K rows is too large for t-SNE (O(n^2)) and Hierarchical Clustering (O(n^2)
# memory for the linkage matrix). We take a stratified subsample.
# We OVERSAMPLE fraud in the subsample to make cluster visualization meaningful
# (otherwise 1.5% fraud would be invisible in plots).
# ==============================================================================
print("\n[STEP 2] Stratified Subsampling for Clustering...")

SAMPLE_SIZE = 10000
FRAUD_OVERSAMPLE = 2000  # Ensure enough fraud points for visual clusters

np.random.seed(42)
fraud_idx = np.where(y_true == 1)[0]
legit_idx = np.where(y_true == 0)[0]

# Take all fraud up to FRAUD_OVERSAMPLE, rest from legitimate
n_fraud = min(FRAUD_OVERSAMPLE, len(fraud_idx))
n_legit = SAMPLE_SIZE - n_fraud

fraud_sample = np.random.choice(fraud_idx, n_fraud, replace=False)
legit_sample = np.random.choice(legit_idx, n_legit, replace=False)
sample_idx = np.concatenate([fraud_sample, legit_sample])
np.random.shuffle(sample_idx)

X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y_true[sample_idx]

print(f"  Sample size: {len(X_sample):,}")
print(f"  Fraud in sample: {y_sample.sum():,} ({y_sample.mean()*100:.1f}%)")
print(f"  Legit in sample: {(1-y_sample).sum():.0f}")

# ==============================================================================
# STEP 3: DIMENSIONALITY REDUCTION
# ==============================================================================
print("\n[STEP 3] Dimensionality Reduction (PCA + t-SNE)...")

# --- PCA ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sample)
print(f"  PCA: Explained variance = {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"    PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"    PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

# --- t-SNE ---
print("  Running t-SNE (this may take a minute)...")
start = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_sample)
print(f"  t-SNE completed in {time.time()-start:.1f}s")

# --- Ground truth visualization ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, X_embed, title in [(axes[0], X_pca, 'PCA'), (axes[1], X_tsne, 't-SNE')]:
    scatter = ax.scatter(X_embed[:, 0], X_embed[:, 1],
                         c=y_sample, cmap='RdYlGn_r', alpha=0.5, s=8,
                         edgecolors='none')
    ax.set_title(f'{title} -- Ground Truth Labels', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'{title} Component 1')
    ax.set_ylabel(f'{title} Component 2')
    plt.colorbar(scatter, ax=ax, label='Is Fraud')

fig.tight_layout()
fig.savefig('datasets/clustering_ground_truth.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: datasets/clustering_ground_truth.png")

# ==============================================================================
# STEP 4: K-MEANS CLUSTERING
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# k-Means minimizes inertia: J = sum_i ||x_i - mu_{c(i)}||^2
# where c(i) is the cluster assignment for point i.
# We first run the Elbow Method to find optimal k, then fit the final model.
#
# Elbow Method: Plot inertia vs k. The "elbow" (point of diminishing returns)
# suggests the optimal k. We also use Silhouette Score:
#   S(i) = (b(i) - a(i)) / max(a(i), b(i))
# where a(i) = avg intra-cluster distance, b(i) = avg nearest-cluster distance.
# S in [-1, 1]: higher = better separation.
# ==============================================================================
print("\n[STEP 4] k-Means Clustering...")

# Elbow Method
k_range = range(2, 11)
inertias = []
silhouettes = []

print("  Running Elbow Method (k=2..10)...")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels, sample_size=5000)
    silhouettes.append(sil)
    print(f"    k={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

# Plot Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal k', fontsize=13, fontweight='bold')

axes[1].plot(list(k_range), silhouettes, 'rs-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')

fig.tight_layout()
fig.savefig('datasets/clustering_elbow_silhouette.png', dpi=150)
plt.close(fig)
print("  Saved: datasets/clustering_elbow_silhouette.png")

# Fit final k-Means with k=3 (fraud, legitimate-low-risk, legitimate-high-activity)
OPTIMAL_K = 3
print(f"\n  Fitting k-Means with k={OPTIMAL_K}...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_sample)

km_sil = silhouette_score(X_sample, km_labels, sample_size=5000)
km_ari = adjusted_rand_score(y_sample, km_labels)
print(f"  Silhouette Score: {km_sil:.4f}")
print(f"  Adjusted Rand Index vs. ground truth: {km_ari:.4f}")

# Cluster-fraud cross-tabulation
print("\n  k-Means Cluster vs Fraud Cross-Tabulation:")
ct_km = pd.crosstab(km_labels, y_sample, rownames=['Cluster'], colnames=['Is_Fraud'])
ct_km.columns = ['Legitimate', 'Fraud']
ct_km['Fraud_Rate_%'] = (ct_km['Fraud'] / (ct_km['Fraud'] + ct_km['Legitimate']) * 100).round(2)
print(ct_km.to_string())

# ==============================================================================
# STEP 5: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# Agglomerative clustering: start with n singleton clusters, iteratively merge
# the two closest clusters. "Closest" is defined by the linkage criterion:
#   - Ward: Minimize the increase in total within-cluster variance.
#     This is equivalent to minimizing the sum of squared differences within
#     each cluster. Ward tends to produce compact, spherical clusters.
# The dendrogram shows the merge history and cluster distances.
# ==============================================================================
print("\n[STEP 5] Hierarchical (Agglomerative) Clustering...")

# For dendrogram, use scipy linkage on a smaller subsample (dendrogram is hard to read > 500 pts)
dendro_size = 500
dendro_idx = np.random.choice(len(X_sample), dendro_size, replace=False)
X_dendro = X_sample.iloc[dendro_idx]

print(f"  Computing Ward linkage on {dendro_size} points for dendrogram...")
Z = linkage(X_dendro, method='ward')

fig_dendro, ax_dendro = plt.subplots(figsize=(16, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8,
           ax=ax_dendro, color_threshold=0.7*max(Z[:,2]))
ax_dendro.set_title('Agglomerative Clustering Dendrogram (Ward Linkage)',
                    fontsize=13, fontweight='bold')
ax_dendro.set_xlabel('Sample Index (or cluster size)', fontsize=12)
ax_dendro.set_ylabel('Ward Distance', fontsize=12)
fig_dendro.tight_layout()
fig_dendro.savefig('datasets/clustering_dendrogram.png', dpi=150)
plt.close(fig_dendro)
print("  Saved: datasets/clustering_dendrogram.png")

# Fit Agglomerative on full sample
print(f"  Fitting Agglomerative Clustering (n_clusters={OPTIMAL_K})...")
agg = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
agg_labels = agg.fit_predict(X_sample)

agg_sil = silhouette_score(X_sample, agg_labels, sample_size=5000)
agg_ari = adjusted_rand_score(y_sample, agg_labels)
print(f"  Silhouette Score: {agg_sil:.4f}")
print(f"  Adjusted Rand Index vs. ground truth: {agg_ari:.4f}")

print("\n  Hierarchical Cluster vs Fraud Cross-Tabulation:")
ct_agg = pd.crosstab(agg_labels, y_sample, rownames=['Cluster'], colnames=['Is_Fraud'])
ct_agg.columns = ['Legitimate', 'Fraud']
ct_agg['Fraud_Rate_%'] = (ct_agg['Fraud'] / (ct_agg['Fraud'] + ct_agg['Legitimate']) * 100).round(2)
print(ct_agg.to_string())

# ==============================================================================
# STEP 6: DBSCAN CLUSTERING
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# DBSCAN defines clusters as dense regions:
#   - Core point: has >= min_samples neighbors within eps radius
#   - Border point: within eps of a core point but not core itself
#   - Noise point: neither core nor border --> OUTLIER
#
# WHY DBSCAN IS UNIQUELY SUITED FOR FRAUD:
#   Fraud transactions are ANOMALIES -- they don't belong to any dense cluster.
#   DBSCAN explicitly labels them as noise (label = -1). This is fundamentally
#   different from k-Means/Hierarchical, which force every point into a cluster.
#
# PARAMETER SELECTION:
#   eps: We use the k-distance graph method. Sort the distance to each point's
#        kth nearest neighbor. The "knee" of this curve suggests eps.
#   min_samples: Rule of thumb = 2 * n_features. With ~20 features, we use ~20.
#                (Using a smaller value for subsampled data.)
# ==============================================================================
print("\n[STEP 6] DBSCAN Clustering...")

# k-distance plot for eps estimation
from sklearn.neighbors import NearestNeighbors

k_dist = 10  # Use k=min_samples for the k-distance graph
nn = NearestNeighbors(n_neighbors=k_dist)
nn.fit(X_sample)
distances, _ = nn.kneighbors(X_sample)
k_distances = np.sort(distances[:, -1])[::-1]

fig_kdist, ax_kdist = plt.subplots(figsize=(10, 5))
ax_kdist.plot(range(len(k_distances)), k_distances, linewidth=1.5)
ax_kdist.set_xlabel('Points (sorted by distance)', fontsize=12)
ax_kdist.set_ylabel(f'{k_dist}-NN Distance', fontsize=12)
ax_kdist.set_title(f'k-Distance Graph (k={k_dist}) for eps Estimation',
                   fontsize=13, fontweight='bold')
ax_kdist.axhline(y=np.percentile(k_distances, 5), color='r', linestyle='--',
                 label=f'95th percentile = {np.percentile(k_distances, 5):.2f}')
ax_kdist.legend(fontsize=10)
fig_kdist.tight_layout()
fig_kdist.savefig('datasets/clustering_kdistance.png', dpi=150)
plt.close(fig_kdist)
print("  Saved: datasets/clustering_kdistance.png")

# Fit DBSCAN
EPS_VALUE = np.percentile(k_distances, 5)  # Use knee approximation
MIN_SAMPLES = 10
print(f"  DBSCAN params: eps={EPS_VALUE:.2f}, min_samples={MIN_SAMPLES}")

start = time.time()
dbscan = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES, n_jobs=-1)
db_labels = dbscan.fit_predict(X_sample)
print(f"  DBSCAN completed in {time.time()-start:.1f}s")

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"  Clusters found: {n_clusters_db}")
print(f"  Noise points: {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")

# DBSCAN: noise vs fraud correlation
print("\n  DBSCAN Noise vs Fraud Analysis:")
noise_mask = db_labels == -1
fraud_in_noise = y_sample[noise_mask].sum()
fraud_total = y_sample.sum()
legit_in_noise = (~y_sample.astype(bool))[noise_mask].sum()

print(f"    Fraud points labeled as noise: {fraud_in_noise:.0f} / {fraud_total:.0f}"
      f" ({fraud_in_noise/fraud_total*100:.1f}%)")
print(f"    Legit points labeled as noise: {legit_in_noise:.0f} / {(1-y_sample).sum():.0f}"
      f" ({legit_in_noise/(1-y_sample).sum()*100:.1f}%)")

if n_noise > 0 and fraud_in_noise > 0:
    noise_fraud_rate = fraud_in_noise / n_noise * 100
    overall_fraud_rate = y_sample.mean() * 100
    print(f"    Fraud rate in NOISE cluster: {noise_fraud_rate:.1f}%")
    print(f"    Fraud rate in CORE clusters: "
          f"{(fraud_total - fraud_in_noise) / (len(y_sample) - n_noise) * 100:.1f}%")
    print(f"    Overall fraud rate: {overall_fraud_rate:.1f}%")
    print(f"    --> DBSCAN noise fraud enrichment: {noise_fraud_rate/overall_fraud_rate:.1f}x")

# Cross-tabulation for DBSCAN
print("\n  DBSCAN Cluster vs Fraud Cross-Tabulation:")
ct_db = pd.crosstab(db_labels, y_sample, rownames=['Cluster'], colnames=['Is_Fraud'])
ct_db.columns = ['Legitimate', 'Fraud']
ct_db['Fraud_Rate_%'] = (ct_db['Fraud'] / (ct_db['Fraud'] + ct_db['Legitimate']) * 100).round(2)
print(ct_db.to_string())

# ==============================================================================
# STEP 7: CLUSTER VISUALIZATION (PCA + t-SNE)
# ==============================================================================
print("\n[STEP 7] Generating Cluster Visualizations...")

# --- 7a: k-Means, Hierarchical, DBSCAN on PCA ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
titles = ['k-Means', 'Hierarchical (Ward)', 'DBSCAN']
all_labels = [km_labels, agg_labels, db_labels]

for ax, labels, title in zip(axes, all_labels, titles):
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for lbl, col in zip(sorted(unique_labels), colors):
        mask = labels == lbl
        label_name = f'Noise ({mask.sum()})' if lbl == -1 else f'Cluster {lbl} ({mask.sum()})'
        marker = 'x' if lbl == -1 else 'o'
        alpha = 0.8 if lbl == -1 else 0.4
        s = 15 if lbl == -1 else 8
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[col], label=label_name,
                   alpha=alpha, s=s, marker=marker, edgecolors='none')

    ax.set_title(f'{title} (PCA)', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7, loc='best')

fig.suptitle('Cluster Assignments -- PCA Projection', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('datasets/clustering_pca_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: datasets/clustering_pca_comparison.png")

# --- 7b: Same on t-SNE ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, labels, title in zip(axes, all_labels, titles):
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for lbl, col in zip(sorted(unique_labels), colors):
        mask = labels == lbl
        label_name = f'Noise ({mask.sum()})' if lbl == -1 else f'Cluster {lbl} ({mask.sum()})'
        marker = 'x' if lbl == -1 else 'o'
        alpha = 0.8 if lbl == -1 else 0.4
        s = 15 if lbl == -1 else 8
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[col], label=label_name,
                   alpha=alpha, s=s, marker=marker, edgecolors='none')

    ax.set_title(f'{title} (t-SNE)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, loc='best')

fig.suptitle('Cluster Assignments -- t-SNE Projection', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('datasets/clustering_tsne_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: datasets/clustering_tsne_comparison.png")

# ==============================================================================
# STEP 8: COMPARATIVE SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("MODULE 5: CLUSTERING RESULTS SUMMARY")
print("=" * 70)

summary = {
    'k-Means (k=3)': {
        'Clusters': OPTIMAL_K,
        'Noise Points': 0,
        'Silhouette': round(km_sil, 4),
        'Adj. Rand Index': round(km_ari, 4),
    },
    'Hierarchical (Ward)': {
        'Clusters': OPTIMAL_K,
        'Noise Points': 0,
        'Silhouette': round(agg_sil, 4),
        'Adj. Rand Index': round(agg_ari, 4),
    },
    'DBSCAN': {
        'Clusters': n_clusters_db,
        'Noise Points': n_noise,
        'Silhouette': round(silhouette_score(X_sample[db_labels != -1],
                            db_labels[db_labels != -1],
                            sample_size=min(5000, (db_labels != -1).sum())), 4)
                      if n_clusters_db > 1 else 'N/A',
        'Adj. Rand Index': round(adjusted_rand_score(y_sample, db_labels), 4),
    }
}

summary_df = pd.DataFrame(summary).T
print("\nClustering Performance Comparison:")
print("-" * 70)
print(summary_df.to_string())

print(f"""
INTERPRETATION:
----------------------------------------------------------------------
  - Silhouette Score: Measures cluster cohesion vs separation. Range [-1, 1].
    Higher = better-defined clusters.

  - Adjusted Rand Index (ARI): Measures agreement between cluster assignments
    and ground truth (is_fraud). Range [-0.5, 1]. ARI=1 means perfect match.
    ARI~0 means random. NOTE: We do NOT expect high ARI because clustering
    is unsupervised -- it discovers structure, not fraud labels directly.

  ALGORITHM COMPARISON:
  - k-Means: Forces every point into a cluster. Good for finding broad
    behavioral groups but cannot isolate outliers/fraud as anomalies.
  - Hierarchical: Similar to k-Means but reveals merge hierarchy via
    dendrogram. Useful for understanding cluster relationships.
  - DBSCAN: The only algorithm that explicitly labels outliers as NOISE.
    If fraud transactions are sparse/anomalous, they appear in the noise
    cluster. This makes DBSCAN uniquely suited for anomaly-based fraud
    detection where we lack labeled training data.
""")

# Save summary
summary_df.to_csv("datasets/clustering_metrics.csv")
print("  Saved metrics to datasets/clustering_metrics.csv")
print("\nMODULE 5 COMPLETE [OK]")
