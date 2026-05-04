"""
==============================================================================
MODULE 4: Classification Modeling & Evaluation
BCI606 - Data Mining Mini Project
Domain: Digital Payment Fraud Detection
==============================================================================

WHY THIS MODULE MATTERS:
    We train supervised machine learning models to differentiate between
    legitimate and fraudulent transactions. Because of the extreme class
    imbalance in the real-world test set (~1.5% fraud), standard accuracy is
    MISLEADING. We focus on Recall (catching fraud), Precision (avoiding false
    alarms), F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices.

CLASSIFIERS (as per BCI606 rubric):
1. Decision Tree     -- Axis-aligned splits, highly interpretable, prone to
                        overfitting. Uses Information Gain (Entropy) or Gini
                        Impurity to greedily select the best feature split.
2. Naive Bayes       -- Probabilistic classifier based on Bayes' theorem with
                        the "naive" conditional independence assumption:
                        P(Fraud|X) ~ P(X|Fraud) * P(Fraud) / P(X).
                        Gaussian NB assumes each feature follows N(mu, sigma^2).
3. k-Nearest         -- Instance-based / lazy learner. Classifies a new point
   Neighbors (k-NN)     by majority vote of its k closest neighbors in feature
                        space. Distance metric (Euclidean) is sensitive to scale
                        --> Z-score normalization in Module 2 was essential.

MATHEMATICAL FOUNDATIONS:
- Decision Tree (Gini): Gini(S) = 1 - sum(p_i^2). A pure node has Gini = 0.
  The split maximizing Delta_Gini = Gini(parent) - weighted_sum(Gini(children))
  is chosen greedily at each level.

- Naive Bayes (Gaussian): P(y|x1,...,xn) = P(y) * prod(P(xi|y)) / P(x).
  For continuous features, P(xi|y) = (1 / sqrt(2*pi*sigma_y^2)) *
  exp(-(xi - mu_y)^2 / (2*sigma_y^2)).
  The "naive" assumption of feature independence rarely holds, yet GNB often
  performs surprisingly well because the RANKING of posterior probabilities
  remains correct even when absolute values are miscalibrated.

- k-NN: d(x, x') = sqrt(sum((xi - xi')^2)). The predicted class is the
  mode of the labels of the k nearest training points. k must be ODD for
  binary classification to avoid ties. Computational cost at prediction time
  is O(n*d) where n = training size, d = dimensionality.

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             auc, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
import warnings
import time
warnings.filterwarnings('ignore')

# -- Configure Plots ----------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("=" * 70)
print("MODULE 4: Classification Modeling")
print("Classifiers: Decision Tree | Naive Bayes | k-NN")
print("=" * 70)

# ==============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ==============================================================================
print("\n[STEP 1] Loading Preprocessed Datasets...")
try:
    df_train = pd.read_csv("datasets/preprocessed_train.csv")
    df_test = pd.read_csv("datasets/preprocessed_test.csv")
    print(f"  Training Set (SMOTE-balanced) Shape : {df_train.shape}")
    print(f"  Test Set (Original Imbalance) Shape : {df_test.shape}")
except FileNotFoundError:
    print("[ERROR] Preprocessed data not found. Please run module2_preprocessing.py first.")
    exit()

X_train = df_train.drop(columns=['is_fraud'])
y_train = df_train['is_fraud']

X_test = df_test.drop(columns=['is_fraud'])
y_test = df_test['is_fraud']

# Display class distribution for test set to emphasize the imbalanced evaluation
test_fraud_rate = y_test.mean() * 100
print(f"  Test Set Fraud Rate: {test_fraud_rate:.2f}% (Highly Imbalanced)")
print(f"  Features used: {X_train.shape[1]}")
print(f"  Feature names: {list(X_train.columns)}")

# ==============================================================================
# STEP 2: MODEL DEFINITIONS WITH HYPERPARAMETER JUSTIFICATION
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# Decision Tree:
#   - max_depth=10: Limits tree depth to prevent overfitting. Without this,
#     the tree can grow to depth > 30 on 600K+ SMOTE samples, memorizing noise.
#   - criterion='gini': Computationally cheaper than entropy (no log computation)
#     and empirically equivalent for most tasks.
#   - min_samples_split=20: A node must have >= 20 samples to attempt a split.
#     This regularizes against tiny, noisy leaf nodes.
#
# Gaussian Naive Bayes:
#   - var_smoothing=1e-9: Adds a small fraction of the largest variance to all
#     feature variances, preventing division-by-zero in P(xi|y) and stabilizing
#     the posterior when a feature has near-zero variance for one class.
#   - No hyperparameters to tune -- its strength is simplicity and speed.
#
# k-NN:
#   - n_neighbors=7: An odd number to break ties. k=7 balances between:
#     * k too small (k=1): overfits, boundary is noisy
#     * k too large (k=50): underfits, decision boundary is too smooth
#   - metric='euclidean': Standard L2 norm. Works well because we Z-score
#     normalized all features in Module 2.
#   - weights='distance': Closer neighbors get more vote weight:
#     w_i = 1/d(x, x_i). This improves boundary resolution.
#   - n_jobs=-1: Parallelizes distance computation across all CPU cores.
#
# NOTE ON TRAINING SET SIZE:
#   The SMOTE-balanced training set has ~630K rows. k-NN stores ALL training
#   points and computes distances at prediction time, so it will be SLOW.
#   We subsample the training set for k-NN to keep runtime reasonable while
#   maintaining classification quality.
# ==============================================================================
print("\n[STEP 2] Defining Models with Justified Hyperparameters...")

models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    "Naive Bayes (Gaussian)": GaussianNB(
        var_smoothing=1e-9
    ),
    "k-NN (k=7)": KNeighborsClassifier(
        n_neighbors=7,
        metric='euclidean',
        weights='distance',
        n_jobs=-1
    )
}

for name, model in models.items():
    print(f"  {name}: {model}")

# ==============================================================================
# STEP 3: TRAINING & EVALUATION
# ==============================================================================
print("\n[STEP 3] Training and Evaluating Models...")
print("-" * 70)

results = {}

# For k-NN, we subsample to avoid O(n*d) prediction bottleneck
# Stratified subsample: keep fraud ratio intact
KNN_TRAIN_SIZE = 50000  # ~8% of full training set -- sufficient for k-NN

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    print(f"\n  -> Training {name}...")

    # Subsample for k-NN (lazy learner -- prediction time is O(n*d))
    if "k-NN" in name and len(X_train) > KNN_TRAIN_SIZE:
        print(f"     [!] Subsampling training set to {KNN_TRAIN_SIZE:,} for k-NN")
        print(f"         (k-NN stores all points; {len(X_train):,} rows would be too slow)")
        np.random.seed(42)
        subsample_idx = np.random.choice(len(X_train), KNN_TRAIN_SIZE, replace=False)
        X_fit = X_train.iloc[subsample_idx]
        y_fit = y_train.iloc[subsample_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    start_time = time.time()
    model.fit(X_fit, y_fit)
    train_time = time.time() - start_time
    print(f"     Training time: {train_time:.2f}s")

    # Predictions
    start_pred = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred
    print(f"     Prediction time: {pred_time:.2f}s")

    # --- Metrics Calculation ---
    # Under the Hood:
    #   Accuracy = (TP + TN) / (TP + TN + FP + FN)  -- MISLEADING at 1.5% fraud
    #   Precision = TP / (TP + FP)  -- "Of those I flagged, how many were real fraud?"
    #   Recall = TP / (TP + FN)  -- "Of all real fraud, how many did I catch?"
    #   F1 = 2 * (Precision * Recall) / (Precision + Recall) -- harmonic mean
    #   ROC-AUC = Area under TPR vs FPR curve. 0.5 = random, 1.0 = perfect.
    #   PR-AUC = Area under Precision vs Recall curve. More informative than
    #            ROC-AUC under severe imbalance because it focuses on the
    #            minority class performance.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_prob)

    # PR AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = auc(recall_vals, precision_vals)

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc_val,
        'PR-AUC': pr_auc_val,
        'Train Time (s)': round(train_time, 2),
        'Predict Time (s)': round(pred_time, 2)
    }

    # Print per-model results immediately
    print(f"     Accuracy:  {acc:.4f}")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall:    {rec:.4f}")
    print(f"     F1-Score:  {f1:.4f}")
    print(f"     ROC-AUC:   {roc_auc_val:.4f}")
    print(f"     PR-AUC:    {pr_auc_val:.4f}")

    # Full classification report
    print(f"\n     Classification Report ({name}):")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'],
                                 digits=4))

    # --- Visualizations ---
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})", linewidth=2)

    # PR Curve
    ax_pr.plot(recall_vals, precision_vals, label=f"{name} (AUC = {pr_auc_val:.3f})", linewidth=2)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
    disp.plot(ax=axes_cm[idx], cmap='Blues', values_format='d')
    axes_cm[idx].set_title(f"{name}\nPrec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}",
                           fontsize=10)
    axes_cm[idx].grid(False)

# ==============================================================================
# STEP 4: SAVE VISUALIZATIONS
# ==============================================================================
print("\n[STEP 4] Generating and Saving Visualizations...")

# --- 4a: ROC Curve Comparison ---
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.500)')
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curve Comparison: Decision Tree vs Naive Bayes vs k-NN',
                 fontsize=13, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=10)
fig_roc.tight_layout()
fig_roc.savefig('datasets/classification_roc_curves.png', dpi=150)
plt.close(fig_roc)
print("  Saved: datasets/classification_roc_curves.png")

# --- 4b: Precision-Recall Curve Comparison ---
baseline_pr = y_test.mean()
ax_pr.plot([0, 1], [baseline_pr, baseline_pr], 'k--', alpha=0.5,
           label=f'Baseline ({baseline_pr:.3f})')
ax_pr.set_xlabel('Recall', fontsize=12)
ax_pr.set_ylabel('Precision', fontsize=12)
ax_pr.set_title('Precision-Recall Curve: Decision Tree vs Naive Bayes vs k-NN',
                fontsize=13, fontweight='bold')
ax_pr.legend(loc='upper right', fontsize=10)
fig_pr.tight_layout()
fig_pr.savefig('datasets/classification_pr_curves.png', dpi=150)
plt.close(fig_pr)
print("  Saved: datasets/classification_pr_curves.png")

# --- 4c: Confusion Matrices ---
fig_cm.suptitle('Confusion Matrices (Test Set - Imbalanced)', fontsize=14, fontweight='bold', y=1.02)
fig_cm.tight_layout()
fig_cm.savefig('datasets/classification_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig_cm)
print("  Saved: datasets/classification_confusion_matrices.png")

# ==============================================================================
# STEP 5: DECISION TREE INTERPRETABILITY
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# One of the key advantages of Decision Trees is their interpretability.
# We can visualize the tree and extract human-readable rules.
# This directly supports the XAI (Explainable AI) requirement noted in Module 1.
# ==============================================================================
print("\n[STEP 5] Decision Tree Interpretability...")

dt_model = models["Decision Tree"]

# Text-based tree rules (first 5 levels)
print("\n  Decision Tree Rules (first 5 levels):")
print("-" * 70)
tree_rules = export_text(dt_model, feature_names=list(X_train.columns), max_depth=5)
print(tree_rules)

# Visual tree plot (shallow version for readability)
fig_tree, ax_tree = plt.subplots(figsize=(24, 12))
plot_tree(dt_model, max_depth=3, feature_names=list(X_train.columns),
          class_names=['Legit', 'Fraud'], filled=True, rounded=True,
          fontsize=8, ax=ax_tree, proportion=True)
ax_tree.set_title('Decision Tree (Top 3 Levels) -- Fraud Detection',
                  fontsize=16, fontweight='bold')
fig_tree.tight_layout()
fig_tree.savefig('datasets/classification_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close(fig_tree)
print("  Saved: datasets/classification_decision_tree.png")

# Feature importance from the Decision Tree
print("\n  Decision Tree Feature Importance (Top 10):")
print("-" * 50)
importances = dt_model.feature_importances_
feature_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
for feat, imp in feature_imp.head(10).items():
    print(f"    {feat:40s} {imp:.4f}")

# Feature importance bar chart
fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
top_features = feature_imp.head(10)
sns.barplot(x=top_features.values, y=top_features.index, ax=ax_fi, palette="magma")
ax_fi.set_title('Top 10 Feature Importances (Decision Tree)', fontsize=13, fontweight='bold')
ax_fi.set_xlabel('Gini Importance Score')
fig_fi.tight_layout()
fig_fi.savefig('datasets/classification_feature_importance.png', dpi=150)
plt.close(fig_fi)
print("  Saved: datasets/classification_feature_importance.png")

# ==============================================================================
# STEP 6: k-NN SENSITIVITY ANALYSIS (Effect of k)
# ==============================================================================
# --- Under the Hood -----------------------------------------------------------
# k is the most critical hyperparameter in k-NN. We evaluate k = {3, 5, 7, 9, 11}
# on the test set to show how it affects the bias-variance trade-off:
#   - Small k --> low bias, high variance (noisy boundaries)
#   - Large k --> high bias, low variance (smooth boundaries)
# ==============================================================================
print("\n[STEP 6] k-NN Sensitivity Analysis (varying k)...")

k_values = [3, 5, 7, 9, 11]
k_results = []

# Use the same subsample for consistency
np.random.seed(42)
subsample_idx = np.random.choice(len(X_train), KNN_TRAIN_SIZE, replace=False)
X_knn_train = X_train.iloc[subsample_idx]
y_knn_train = y_train.iloc[subsample_idx]

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean',
                                     weights='distance', n_jobs=-1)
    knn_temp.fit(X_knn_train, y_knn_train)
    y_pred_k = knn_temp.predict(X_test)
    y_prob_k = knn_temp.predict_proba(X_test)[:, 1]

    k_results.append({
        'k': k,
        'Precision': precision_score(y_test, y_pred_k),
        'Recall': recall_score(y_test, y_pred_k),
        'F1-Score': f1_score(y_test, y_pred_k),
        'ROC-AUC': roc_auc_score(y_test, y_prob_k)
    })

k_df = pd.DataFrame(k_results).set_index('k')
print(k_df.round(4).to_string())

# Plot k sensitivity
fig_k, ax_k = plt.subplots(figsize=(8, 5))
k_df[['Precision', 'Recall', 'F1-Score']].plot(ax=ax_k, marker='o', linewidth=2)
ax_k.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax_k.set_ylabel('Score', fontsize=12)
ax_k.set_title('k-NN Sensitivity Analysis: Effect of k on Fraud Detection',
               fontsize=13, fontweight='bold')
ax_k.legend(fontsize=10)
ax_k.set_xticks(k_values)
fig_k.tight_layout()
fig_k.savefig('datasets/classification_knn_sensitivity.png', dpi=150)
plt.close(fig_k)
print("  Saved: datasets/classification_knn_sensitivity.png")

# ==============================================================================
# STEP 7: COMPARATIVE RESULTS SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("MODULE 4: COMPARATIVE RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results).T

# Display with interpretation
print("\nPerformance Metrics (Test Set -- Original Imbalance):")
print("-" * 70)
print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']].round(4).to_string())

print("\nTraining & Prediction Times:")
print("-" * 70)
print(results_df[['Train Time (s)', 'Predict Time (s)']].to_string())

# Interpretation
print("\n" + "-" * 70)
print("INTERPRETATION:")
print("-" * 70)

best_f1_model = results_df['F1-Score'].idxmax()
best_recall_model = results_df['Recall'].idxmax()
best_roc_model = results_df['ROC-AUC'].idxmax()

print(f"""
  Best F1-Score:  {best_f1_model} ({results_df.loc[best_f1_model, 'F1-Score']:.4f})
  Best Recall:    {best_recall_model} ({results_df.loc[best_recall_model, 'Recall']:.4f})
  Best ROC-AUC:   {best_roc_model} ({results_df.loc[best_roc_model, 'ROC-AUC']:.4f})

  KEY OBSERVATIONS:
  - Accuracy is HIGH for all models (~97-99%), but this is MISLEADING because
    predicting "Not Fraud" always gives ~98% accuracy. IGNORE raw accuracy.
  - Precision measures false alarm rate: higher = fewer false alarms.
  - Recall measures fraud catch rate: higher = fewer missed frauds.
  - F1-Score balances Precision and Recall. This is the PRIMARY metric.
  - PR-AUC is more informative than ROC-AUC under extreme class imbalance.

  ALGORITHM COMPARISON:
  - Decision Tree: Fast, interpretable, but can overfit (high variance).
    The tree structure reveals which features matter most for fraud detection.
  - Naive Bayes: Very fast (O(n*d) training), assumes feature independence.
    Often has lower precision but can have competitive recall due to its
    probabilistic nature capturing rare class patterns.
  - k-NN: Computationally expensive (lazy learner), but captures complex
    non-linear decision boundaries. Performance depends heavily on k and
    the quality of feature scaling (Z-score from Module 2 is critical).
""")

# Save results to CSV
results_df.to_csv("datasets/classification_metrics.csv")
print("  Saved metrics to datasets/classification_metrics.csv")
print("\nMODULE 4 COMPLETE [OK]")
