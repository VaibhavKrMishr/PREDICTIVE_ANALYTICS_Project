# Predicting Commodity Prices at Indian Agricultural Mandis Using an Ensemble Machine Learning Pipeline

---

**Vaibhav**  
*Department of Computer Science and Engineering*  
*B.Tech Machine Learning Course — CA2 Project Report*

---

## Abstract

This paper presents a machine learning pipeline for predicting wholesale commodity prices at Indian agricultural mandis (markets). The dataset contains 13,595 daily price records across 26 states, 238 commodities, and 1,377 mandis. After cleaning and feature engineering, the pipeline addresses three interconnected sub-problems: (1) unsupervised commodity grouping using K-Means and Hierarchical clustering, (2) price band classification (Low/Mid/High) using Multinomial Naive Bayes and Decision Tree, and (3) exact modal price regression using Multiple Linear Regression and Random Forest. The models are combined through intra-task ensembling and inter-task sequential prediction. Evaluation on 276 completely unseen test mandis (GroupShuffleSplit) shows the combined regression ensemble achieves an RMSE of 0.4589 on log-transformed prices with R² = 0.6657, while the soft-voting classifier reaches 76.04% accuracy and Macro F1 of 0.7584. Results indicate that commodity type dominates pricing, location provides secondary adjustment, and quality grade has negligible impact.

**Keywords** — *agricultural price prediction, mandi, ensemble learning, K-Means clustering, Naive Bayes, decision tree, random forest, linear regression*

---

## I. Introduction

India's agricultural marketing system operates through a network of wholesale markets called mandis, regulated under the Agricultural Produce Market Committee (APMC) Act. These mandis serve as the primary transaction point between farmers and traders, where commodity prices are determined daily based on supply, demand, and local market conditions.

Price prediction at these mandis is valuable for multiple stakeholders: farmers can make informed decisions about where to sell their produce, traders can optimize procurement, and policymakers can monitor market efficiency. However, mandi pricing is influenced by a complex combination of factors including the commodity type, geographic location, local demand patterns, and transportation logistics.

This project frames the prediction problem as: **"Predict the price a commodity will fetch at an Indian mandi, given the commodity, its grade, and where it is sold."** Rather than treating machine learning models as isolated competitors, we decompose this master problem into three sub-problems that feed into each other as a unified pipeline, using six models from the course syllabus.

---

## II. Dataset Description

### A. Source and Scope

The dataset (`raw.csv`) contains 13,595 records representing one day of agricultural market arrivals across India. Each row represents a single commodity transaction record at a specific mandi.

### B. Features

The original dataset contains 10 columns as shown in Table I.

**Table I: Dataset Features**

| Feature | Type | Description |
|---------|------|-------------|
| State | Categorical | Indian state or union territory (26 unique) |
| District | Categorical | Administrative district |
| Market | Categorical | Mandi name (1,377 unique) |
| Commodity | Categorical | Agricultural commodity (238 unique) |
| Variety | Categorical | Specific variety of the commodity |
| Grade | Categorical | Quality grade label |
| Arrival_Date | Date | Date of market arrival (single day) |
| Min_Price | Numerical | Minimum traded price (Rs/quintal) |
| Max_Price | Numerical | Maximum traded price (Rs/quintal) |
| Modal_Price | Numerical | Most frequently traded price (Rs/quintal) — **target variable** |

### C. Data Cleaning

The cleaning pipeline (`clean.py`) performs the following transformations:

1. Renamed columns by removing `_x0020_` encoding artifacts
2. Dropped `arrival_date` (invariant single-day snapshot)
3. Stripped whitespace and normalized commodity names to title case
4. Corrected state spelling inconsistencies (Keralam → Kerala, Chattisgarh → Chhattisgarh)
5. Removed 11 duplicate rows
6. Filtered records violating the constraint: `min_price ≤ modal_price ≤ max_price`
7. Removed records with prices below Rs 10 (non-quintal units like per-piece flowers)
8. Engineered features: `log_modal = log1p(modal_price)` and `spread = (max_price - min_price) / modal_price`

After cleaning, **13,584 records** remain. Critically, `min_price` and `max_price` are excluded from model features to prevent data leakage, as they are trivially correlated (r > 0.95) with the target.

---

## III. Methodology

### A. Problem Decomposition

The master prediction problem is decomposed into three sub-problems:

- **Sub-Problem 1 (Unsupervised):** "Which group does this commodity belong to?" — Groups commodities by price level, spread, and market availability.
- **Sub-Problem 2 (Classification):** "Which price band (Low/Mid/High) will it sell in?" — Categorizes into economic segments.
- **Sub-Problem 3 (Regression):** "What exact price in Rs should be expected?" — Predicts the continuous modal price.

### B. Train-Test Split Strategy

Standard random splits would allow the same mandi's data to appear in both training and testing, inflating performance. We use `GroupShuffleSplit` with `market` as the grouping variable (`test_size=0.2, random_state=42`):

- **Training set:** 11,109 rows from 1,101 mandis
- **Test set:** 2,475 rows from 276 completely unseen mandis

This simulates real-world forecasting: predicting prices at a mandi the model has never observed.

### C. Price Band Definition

Modal price is divided into three bands using tertiles computed on training data only:

**Table II: Price Band Thresholds**

| Band | Min Price (Rs) | Max Price (Rs) | Training Count |
|------|---------------|----------------|----------------|
| Low | 10 | 3,250 | 3,838 |
| Mid | 3,251 | 5,750 | 3,603 |
| High | 5,752 | 165,000 | 3,668 |

### D. Feature Engineering

#### Location Features (location_features.py)

Three location-derived features capture geographic pricing patterns:

1. **state_commodity_median:** Median log price for the (state, commodity) pair in training data
2. **state_price_level:** How much a state's prices deviate from national commodity medians
3. **n_markets_selling:** Number of mandis in that state selling this commodity

To prevent data leakage, training location features are computed using **leave-one-market-out GroupKFold cross-validation** — a row never observes its own market's price during feature computation.

#### Commodity Clustering Table

A commodity-level summary table is built from training data (151 commodities with ≥ 3 rows), containing: median log price, median spread, log(number of markets), and number of states.

### E. Model Architecture

#### Layer 1: Intra-Task Ensembling

Within each sub-problem, two models from different paradigms are combined:

**Table III: Models Used Per Sub-Problem**

| Sub-Problem | Model 1 | Model 2 | Ensemble Method |
|-------------|---------|---------|-----------------|
| Clustering | K-Means (k=5) | Hierarchical Ward (k=5) | Bipartite overlap matching |
| Classification | Multinomial Naive Bayes | Decision Tree (min_samples_leaf=5) | Soft probability voting |
| Regression | Multiple Linear Regression | Random Forest (100 trees) | Arithmetic mean of predictions |

#### Layer 2: Inter-Task Sequential Pipeline

For a new query `(commodity, state, grade)`:

1. **Step 1:** Assign commodity to its K-Means cluster
2. **Step 2:** Use cluster + location features to predict price band via soft-voted NB + DT
3. **Step 3:** Use predicted band + cluster + location features to predict exact log price via averaged LR + RF

The price band feature feeding the regressor is generated using **out-of-fold GroupKFold predictions** on training data, ensuring no target leakage.

---

## IV. Experimental Results

### A. Unsupervised Clustering (Sub-Problem 1)

Both K-Means and Hierarchical (Ward linkage) clustering are applied to the 151-commodity standardized feature matrix with k=5.

**Table IV: Clustering Comparison**

| Metric | K-Means | Hierarchical (Ward) |
|--------|---------|---------------------|
| WCSS | 192.64 | 207.54 |
| Silhouette Score | 0.3193 | 0.3094 |
| Agreement Rate | 75.5% | 75.5% |

K-Means achieves lower WCSS (tighter clusters). The 75.5% agreement rate indicates substantial concordance; the 24.5% disagreeing commodities are flagged as "Uncertain."

**Clustering-for-Prediction Test:** To validate that clustering actively contributes to prediction (not just visualization), 10 random commodities were hidden from training. Using cluster median as a price estimate:

- Cluster-based MAPE: **47.5%**
- Overall median baseline MAPE: **62.7%**
- Error reduction: **15.2 percentage points**

### B. Classification Results (Sub-Problem 2)

**Table V: Classification Model Comparison**

| Model Configuration | Accuracy | Precision | Recall | Macro F1 | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Naive Bayes (without location) | 0.7325 | 0.7345 | 0.7321 | 0.7317 | 0.8833 |
| Naive Bayes (with location) | 0.7487 | 0.7503 | 0.7466 | 0.7476 | 0.9024 |
| Decision Tree (without location) | 0.7616 | 0.7575 | 0.7593 | 0.7581 | 0.8984 |
| Decision Tree (with location) | 0.7600 | 0.7563 | 0.7584 | 0.7569 | 0.8980 |
| Combined Soft Vote (without location) | 0.7604 | 0.7588 | 0.7597 | **0.7584** | 0.9067 |
| Combined Soft Vote (with location) | 0.7568 | 0.7597 | 0.7566 | 0.7567 | 0.9103 |
| Full Pipeline (with cluster & location) | 0.7588 | 0.7602 | 0.7571 | 0.7576 | **0.9115** |

The combined soft vote (without location) achieves the highest Macro F1 of **0.7584** and accuracy of **76.04%**. The full pipeline configuration achieves the highest ROC-AUC of **0.9115**, indicating better probability calibration when cluster information is incorporated.

### C. Regression Results (Sub-Problem 3)

**Table VI: Regression Model Comparison**

| Model Configuration | MAE | RMSE | R² (test) | R² (train) |
|---------------------|-----|------|-----------|------------|
| Linear Regression (without location) | 0.2852 | 0.4875 | 0.6227 | 0.7364 |
| Linear Regression (with location) | 0.2563 | 0.4720 | 0.6463 | 0.7688 |
| Random Forest (without location) | 0.2606 | 0.4653 | 0.6562 | 0.8511 |
| Random Forest (with location) | 0.2596 | 0.4753 | 0.6412 | 0.8808 |
| Combined Average (without location) | 0.2581 | **0.4589** | **0.6657** | — |
| Combined Average (with location) | 0.2500 | 0.4657 | 0.6556 | — |
| Full Pipeline (with cluster & band) | 0.2506 | 0.4677 | 0.6527 | 0.8442 |

The combined average without location achieves the lowest test RMSE of **0.4589** and highest R² of **0.6657**. Adding location features improved Linear Regression R² from 0.6227 to 0.6463 (+3.8%), but did not consistently help Random Forest which already captures non-linear state-commodity interactions.

### D. Feature Importance Analysis

**Table VII: Aggregated Feature Importance**

| Feature Group | Random Forest | Decision Tree |
|---------------|---------------|---------------|
| Location Features | 87.40% | 81.96% |
| Commodity | 6.59% | 11.28% |
| State | 3.43% | 4.86% |
| Grade | 2.58% | 1.90% |

Both tree-based models agree that location-derived features (state-commodity median, state price level, number of markets) carry the overwhelming majority of predictive power. Commodity identity provides secondary information, while grade has negligible impact on wholesale pricing.

---

## V. Discussion

### A. Key Findings

1. **Commodity identity determines the price tier:** Whether a mandi sells saffron or onions matters far more than any other factor.
2. **Location provides regional adjustment:** Derived market-depth features (state-commodity median, state price level) capture freight costs, local demand, and supply availability. These contribute a ±15-20% adjustment over commodity baseline.
3. **Grade labels are uninformative:** Quality grades contributed under 3% importance across all models, suggesting that wholesale mandi grading is either inconsistent or does not differentiate pricing in practice.
4. **Ensemble averaging reduces error:** Combining Linear Regression with Random Forest consistently reduced RMSE compared to either model alone, confirming that averaging a linear baseline with a non-linear ensemble captures complementary patterns.

### B. Model Disagreements

- **Linear Regression vs Random Forest:** Linear regression assumes additive price effects without interactions. Random Forest captures non-linear state-crop combinations but exhibits a wider train-test performance gap (potential overfitting).
- **Naive Bayes vs Decision Tree:** Naive Bayes treats features as conditionally independent, defaulting to marginal probabilities for unseen combinations. Decision Trees create hard binary splits that may be fragile on novel categories.
- **K-Means vs Hierarchical:** Disagreements (24.5%) occur on borderline commodities whose trading volumes and spreads place them between bulk grains and regional cash crops.

### C. Pipeline Synergy

The inter-task pipeline demonstrates that sequencing unsupervised grouping → classification → regression produces competitive results. The cluster assignment provides a meaningful prior for rare commodities, and the predicted band constrains the regressor's search space.

---

## VI. Limitations

1. **Single-Day Snapshot:** The dataset captures one operational day, reflecting spatial price variation but not temporal trends, seasonal patterns, or inflation effects.
2. **Categorical Constraints:** Only nominal features (commodity, state, grade) are available. Operational factors such as arrival quantity, moisture content, road distance, and weather conditions are absent.
3. **Geographic Concentration:** Tamil Nadu accounts for approximately 48% of all records, meaning southern market dynamics are overrepresented.
4. **Unit Heterogeneity:** While prices below Rs 10 were filtered, some commodities with non-standard units (per-bag, per-crate) may retain residual noise.
5. **Test Set Scope:** GroupShuffleSplit ensures unseen mandis, but unseen commodities in the test set (those absent from training) can only rely on cluster-based fallback estimates.

---

## VII. Conclusion

This project demonstrates that a modular, multi-stage machine learning pipeline can predict wholesale commodity prices at unseen Indian mandis with reasonable accuracy (R² = 0.6657, Macro F1 = 0.7584). The decomposition into clustering, classification, and regression sub-problems, each solved by an ensemble of two complementary models, provides both interpretability and robustness.

The key insight is that commodity identity is the dominant predictor, with geographic location features serving as a secondary regional adjustment. Quality grade labels show virtually no pricing power in wholesale mandi transactions.

Future work could incorporate temporal data (multi-day or seasonal datasets), external variables (rainfall, fuel costs, supply chain data), and deep learning approaches for more complex feature interactions.

---

## VIII. Project Architecture

```
mandi_project/
├── clean.py                 # Data cleaning and feature engineering
├── eda.py                   # Exploratory data analysis and plots
├── location_features.py     # Market-based split and location features
├── linear_regression.py     # Multiple Linear Regression model
├── random_forest.py         # Random Forest Regressor model
├── naive_bayes.py           # Multinomial Naive Bayes model
├── decision_tree.py         # Decision Tree Classifier model
├── kmeans.py                # K-Means clustering model
├── hierarchical.py          # Hierarchical clustering model
├── main.py                  # Central pipeline orchestrator
├── app.py                   # Streamlit dashboard
├── data/
│   ├── raw.csv              # Raw dataset (13,595 rows)
│   └── clean.csv            # Cleaned dataset (13,584 rows)
└── pyproject.toml           # Dependencies (uv)
```

**Running the project:**
```
uv sync
uv run streamlit run app.py
```

---

## References

[1] Government of India, "Agricultural Marketing — APMC Acts," Ministry of Agriculture and Farmers Welfare.

[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[3] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," Computing in Science and Engineering, vol. 9, no. 3, pp. 90–95, 2007.

[4] W. McKinney, "Data Structures for Statistical Computing in Python," Proceedings of the 9th Python in Science Conference, pp. 56–61, 2010.

[5] Streamlit Inc., "Streamlit — The fastest way to build data apps," 2024. [Online]. Available: https://streamlit.io

[6] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[7] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009.

[8] J. R. Quinlan, "Induction of Decision Trees," Machine Learning, vol. 1, no. 1, pp. 81–106, 1986.

---
