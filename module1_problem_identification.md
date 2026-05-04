# MODULE 1: Problem Identification & Domain Understanding
## BCI606 — Data Mining Mini Project (6th Semester)
### Domain: Finance / Cybersecurity
### Dataset: Digital Payment Fraud Detection Benchmark (Kaggle)

---

## 1. Problem Statement

Digital payment systems — spanning credit/debit cards, UPI, mobile wallets, and bank transfers — have become the dominant mode of financial transactions globally. The Reserve Bank of India reported over **14.7 billion digital payment transactions per month** in FY 2024–25, with UPI alone processing over ₹20 lakh crore monthly. This explosive growth in volume has created a proportionally expanding attack surface for financial fraud.

**The core problem is:** Given a stream of real-time transaction records — each described by monetary, behavioral, temporal, and risk-profile attributes — can we automatically distinguish the ~1–2% of transactions that are **fraudulent** from the ~98–99% that are **legitimate**, while minimizing both:

- **False Negatives (missed fraud):** Every undetected fraud is a direct financial loss to the institution or customer, eroding trust and incurring regulatory penalties.
- **False Positives (false alarms):** Every legitimate transaction wrongly flagged as fraud causes customer friction, declined purchases, and operational overhead from manual review.

This is fundamentally a **binary classification problem under extreme class imbalance**, where the cost of errors is **asymmetric** — the cost of a False Negative (missed ₹50,000 fraud) vastly exceeds the cost of a False Positive (temporarily blocking a ₹50,000 legitimate purchase).

---

## 2. Types of Data in Fraud Detection Datasets

The dataset used in this project (*Digital Payment Fraud Detection Benchmark*, 400,000 transactions) contains four primary categories of attributes:

### 2.1 Transactional / Contextual Data
| Attribute | Type | Example Values | Role in Fraud Detection |
|-----------|------|----------------|------------------------|
| `payment_channel` | Nominal (Categorical) | card, upi, wallet, bank_transfer | Different channels exhibit different fraud rates; card-not-present fraud dominates online channels |
| `device_type` | Nominal (Categorical) | mobile, desktop, tablet | Device fingerprinting helps detect account takeover |
| `is_international` | Binary (Boolean) | 0, 1 | Cross-border transactions carry elevated risk |
| `merchant_risk_score` | Numeric (Continuous, 0–1) | 0.12, 0.87 | Aggregated historical fraud propensity of the merchant |

### 2.2 Numerical / Monetary Data
| Attribute | Type | Example Values | Role in Fraud Detection |
|-----------|------|----------------|------------------------|
| `transaction_amount` | Numeric (Ratio scale) | ₹150, ₹48,000 | Extreme amounts (too high or suspiciously low micro-transactions) signal fraud |
| `avg_monthly_spend` | Numeric (Ratio scale) | ₹12,500 | Baseline spending profile; deviations indicate anomaly |
| `amount_deviation_from_user_mean` | Numeric (Ratio scale) | –₹200, +₹35,000 | Quantifies how far the current transaction deviates from the user's historical norm |

### 2.3 Behavioral / Velocity Data
| Attribute | Type | Example Values | Role in Fraud Detection |
|-----------|------|----------------|------------------------|
| `txn_count_1h` | Numeric (Discrete count) | 0, 1, 5 | Burst activity in a short window suggests automated card testing |
| `txn_count_24h` | Numeric (Discrete count) | 2, 12 | Sustained high-frequency activity is abnormal |
| `failed_txn_count_24h` | Numeric (Discrete count) | 0, 4 | Multiple failed attempts indicate brute-force or credential stuffing |
| `ip_risk_score` | Numeric (Continuous, 0–1) | 0.05, 0.92 | Pre-computed risk from IP reputation databases (VPN/Tor/proxy detection) |
| `geo_distance_from_last_txn` | Numeric (Continuous) | 2.5 km, 850 km | Impossible travel: large geo-jumps in short time = likely compromised account |

### 2.4 Temporal Data
| Attribute | Type | Example Values | Role in Fraud Detection |
|-----------|------|----------------|------------------------|
| `transaction_time` | Timestamp | 2023-03-15 02:43:18 | Fraudsters disproportionately operate during off-peak hours (2–5 AM) |
| Derived: `hour_of_day` | Numeric (Ordinal/Cyclic) | 0–23 | Captures intra-day fraud patterns |
| Derived: `day_of_week` | Numeric (Ordinal) | 0 (Mon) – 6 (Sun) | Weekend vs. weekday behavioral differences |
| Derived: `month` | Numeric (Ordinal) | 1–12 | Captures **temporal concept drift** (fraud patterns shift after month 6 in this dataset) |

### 2.5 Customer Profile (Ordinal/Demographic)
| Attribute | Type | Example Values | Role in Fraud Detection |
|-----------|------|----------------|------------------------|
| `account_age_days` | Numeric (Ratio) | 30, 900 | Newer accounts are disproportionately associated with fraud |
| `credit_score_band` | Ordinal (1–5) | 1 (lowest) to 5 (highest) | Lower bands may correlate with elevated risk |
| `kyc_level` | Ordinal (1–3) | 1 (minimal) to 3 (full) | Weaker identity verification = weaker trust signal |

### 2.6 Special / Leaky Feature
| Attribute | Type | Caution |
|-----------|------|---------|
| `post_auth_risk_score` | Numeric (0–1) | Generated **after** the fraud decision — contains target leakage. **Must be excluded** from any predictive model. |

---

## 3. Relevance of Data Mining in Finance & Cybersecurity

### 3.1 Why Traditional Rule-Based Systems Fail
Legacy fraud detection systems rely on manually crafted rules (e.g., *"flag any transaction above ₹50,000 from a new account"*). These systems suffer from:
- **Rigidity:** They cannot adapt to evolving fraud tactics without manual rule updates.
- **High false-positive rates:** Overly broad rules flag too many legitimate transactions.
- **Inability to capture multivariate interactions:** A ₹500 transaction from a known device at 2 PM is normal; the same ₹500 from an unknown IP at 3 AM after 5 failed attempts is suspicious. Rules struggle to capture such combinatorial patterns.

### 3.2 How Data Mining Addresses These Challenges
Data mining techniques provide a **data-driven, adaptive** approach:

| Technique | Application in Fraud Detection |
|-----------|-------------------------------|
| **Classification** (Decision Tree, Naïve Bayes, k-NN) | Supervised learning to predict fraud probability from labeled historical data |
| **Association Rule Mining** (Apriori, FP-Growth) | Discovering co-occurring behavioral patterns (e.g., {high_velocity, international, high_ip_risk} → fraud) |
| **Clustering** (k-Means, DBSCAN) | Unsupervised anomaly detection — fraudulent transactions form sparse outlier clusters distinct from dense legitimate clusters |
| **Temporal Pattern Mining** | Detecting concept drift — fraud tactics evolve over time, requiring models that adapt |

### 3.3 The Interdisciplinary Connection
This project sits at the intersection of:
- **Machine Learning:** Feature engineering, model selection, hyperparameter tuning
- **Statistics:** Understanding distributions, handling class imbalance (SMOTE), evaluating significance
- **Database Systems (DBMS):** Efficient storage and retrieval of transactional data; the Apriori algorithm's performance is fundamentally a database scan optimization problem
- **Cybersecurity:** Understanding attacker behavior, IP risk scoring, geo-velocity checks, KYC verification levels

---

## 4. Expected Outcomes

By the end of this 5-module project, we expect to deliver:

1. **A robust preprocessing pipeline** that handles missing values, outliers (retained, not removed — they are fraud signals), class imbalance (via SMOTE), and feature scaling (Z-score normalization).

2. **Interpretable association rules** that reveal the "behavioral recipes" of fraud — e.g., *"When a transaction has high IP risk, high velocity, and comes from an international source, fraud likelihood increases 40×"* — with rigorously computed Support, Confidence, and Lift metrics.

3. **A comparative classification study** using Decision Tree, Naïve Bayes, and k-NN classifiers, evaluated on **Precision, Recall, F1-Score, and Confusion Matrix** (NOT raw accuracy, which is meaningless under 1.5% fraud rate).

4. **Cluster-based anomaly detection** using k-Means, Hierarchical Clustering, and DBSCAN, with DBSCAN uniquely suited for identifying fraud as density-based outliers — visualized via PCA/t-SNE dimensionality reduction.

5. **Academic deliverables** including performance comparison tables, visualization plots, and interpretation reports suitable for the BCI606 grading rubric.

---

## 5. Recent Trends in Fraud Detection Data Mining (2024–2025)

| Trend | Description | Relevance to This Project |
|-------|-------------|--------------------------|
| **Graph Neural Networks (GNNs)** | Model transaction networks as graphs; detect fraud rings by analyzing relationships between accounts and merchants | Our `merchant_risk_score` and `customer_id` implicitly encode network-level information |
| **Temporal Concept Drift** | Fraud tactics evolve — models trained on January data degrade by June. Adaptive/online learning is critical | This dataset explicitly models drift (fraud dynamics shift after month 6). Our `month` feature captures this |
| **Federated Learning** | Banks train models collaboratively without sharing raw customer data, addressing GDPR/privacy concerns | Not directly implemented, but the privacy-preserving paradigm motivates feature-level risk scores (like `ip_risk_score`) over raw PII |
| **Explainable AI (XAI)** | Regulators (RBI, EU's AI Act) increasingly demand interpretable fraud decisions, not black-box predictions | Our use of Decision Trees and Association Rules inherently provides explainability |
| **Real-Time Streaming Inference** | Sub-100ms fraud scoring at authorization time using edge-deployed models | Our `velocity` features (`txn_count_1h`) simulate real-time behavioral signals |
| **Synthetic Data & SMOTE Variants** | SMOTE, ADASYN, and GAN-based oversampling to combat extreme class imbalance | We apply SMOTE in Module 2 to generate synthetic fraud samples for balanced training |
| **Multi-Modal Fusion** | Combining transactional, device, behavioral, and biometric signals in a single model | This dataset fuses monetary, behavioral, velocity, and identity signals — a multi-modal design |

### Key Statistics:
- Global digital fraud losses exceeded **$48.7 billion in 2023** (Juniper Research), projected to reach **$91 billion by 2028**.
- The average fraud detection system processes **>10,000 transactions per second** with a decision latency requirement of **<50 ms**.
- Machine learning-based fraud detection systems achieve up to **95% detection rates** while reducing false positives by **50–70%** compared to rule-based systems (McKinsey, 2024).

---

## 6. Dataset Summary

| Property | Value |
|----------|-------|
| **Source** | Kaggle — Digital Payment Fraud Detection Benchmark |
| **Total Transactions** | ~400,000 |
| **Fraud Rate** | ~1.5% (post-noise) |
| **Customers** | 40,000 |
| **Merchants** | 8,000 |
| **Time Span** | January–December 2023 |
| **Train/Test Split** | Before Oct 2023 / After Oct 2023 (chronological) |
| **Target Variable** | `is_fraud` (binary: 0 = legitimate, 1 = fraudulent) |
| **Total Features** | 20 raw columns (17 usable after dropping IDs + leaky feature) |
| **License** | CC0: Public Domain |

---

*Prepared for BCI606 Data Mining Mini Project — Module 1*
*Date: May 2025*
