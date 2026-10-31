# Indian Mandi Commodity Price Prediction

Machine learning course project analyzing wholesale agricultural commodity prices across Indian markets (mandis).

---

## Master Problem Statement
> **"Predict the price a commodity will fetch at an Indian mandi, given the commodity, its grade and where it is sold."**

---

## Three Sub-Problems Working as One Pipeline

Rather than treating machine learning algorithms as isolated competitors, this project orchestrates three sub-problems into a unified predictive pipeline:

1. **Unsupervised Clustering:** *"Which group does this commodity belong to?"*
   - Groups commodities by typical price level, spread, and market availability.
   - Provides an immediate baseline anchor even for rare or newly introduced crops.
2. **Classification:** *"Which price band (Low / Mid / High, tertiles of modal price) will it sell in?"*
   - Categorizes crops into economic market segments based on their cluster, destination state, grade, and local market depth.
3. **Regression:** *"What exact price in Rs should be expected?"*
   - Predicts the exact modal price in Rs/quintal by incorporating the commodity's natural group and its predicted price band.

---

## How the Models Work Together (Two Layers)

### Layer 1: Intra-Task Ensembling
Inside each task, two complementary models from different paradigms combine into one robust prediction:
- **Regression:** Averages **Multiple Linear Regression** (additive baseline) and **Random Forest Regressor** (non-linear interactions).
- **Classification:** Soft-voting average of class probabilities from **Multinomial Naive Bayes** (independent features) and **Decision Tree** (rule-based splits).
- **Unsupervised:** Bipartite maximum overlap matching aligns **K-Means** (centroid partitions) and **Hierarchical Clustering** (Ward agglomerative tree) at $k=5$. Commodities where both models agree are confirmed; disagreeing commodities are flagged as *"Uncertain"*.

### Layer 2: Inter-Task Prediction Pipeline
Across the three tasks, a new market entry `(commodity, state, grade)` flows sequentially:
1. **Step 1 (Clustering):** Resolves the commodity's natural cluster group.
2. **Step 2 (Classification):** Combines the cluster ID with state location features to predict the price band and class probabilities.
3. **Step 3 (Regression):** Conditions on the predicted price band, cluster group, and location depth to output the final expected modal price in Rs/quintal.

---

## Modular File Architecture

Every model, analysis component, and data pipeline has its own dedicated `.py` file:

```
├── clean.py               # Data cleaning, validation, log transformation, spread calculation
├── eda.py                 # Exploratory data analysis statistics & visualization plots
├── location_features.py   # Honest market-based split & leave-one-market-out location features
├── linear_regression.py   # Multiple Linear Regression model, coefficients & actual vs predicted plot
├── random_forest.py       # Random Forest Regressor, feature importances & actual vs predicted plot
├── naive_bayes.py         # Multinomial Naive Bayes, top class features & confusion matrix plot
├── decision_tree.py       # Decision Tree Classifier, rules export & confusion matrix plot
├── kmeans.py              # K-Means clustering, cluster profiles & elbow method plot
├── hierarchical.py        # Hierarchical Ward clustering, cluster profiles & dendrogram plots
├── main.py                # Central pipeline orchestrator tying all models and layers together
├── app.py                 # Streamlit dashboard consuming main.py and rendering the UI
├── data/
│   ├── raw.csv            # Raw daily market arrivals dataset (13,595 rows)
│   └── clean.csv          # Cleaned dataset (13,584 rows)
└── pyproject.toml         # Dependencies managed via uv
```

---

## Honest Evaluation & Location Features

- **Market-Based Generalization:** Evaluated using `GroupShuffleSplit` on the `market` column (`test_size=0.2, random_state=42`). 1,101 mandis are used for training and 276 mandis are held out strictly for testing. This realistically tests the system's ability to forecast prices at a *brand new, unseen mandi*.
- **Leave-One-Market-Out Location Features:** Training location features (`state_commodity_median`, `state_price_level`, `n_markets_selling`) are calculated via `GroupKFold` cross-validation so a row never observes its own market's price.
- **Out-of-Fold Pipeline Bands:** The price band feature feeding the regressor is generated using out-of-fold cross-validated predictions on the training data, preventing leakage of ground-truth prices.

---

## Syllabus Models & Viewpoints

| Task | Baseline Model | File | Viewpoint Caption | Flexible Model | File | Viewpoint Caption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regression** | Multiple Linear Regression | `linear_regression.py` | *Linear and additive: adds up weighted category effects without interactions* | Random Forest Regressor | `random_forest.py` | *Non-linear ensemble: combines deep trees to capture state-crop interactions* |
| **Classification** | Multinomial Naive Bayes | `naive_bayes.py` | *Independent features: multiplies feature likelihoods assuming conditional independence* | Decision Tree Classifier | `decision_tree.py` | *Yes/no rules: builds hierarchical binary splits on feature presence* |
| **Unsupervised** | K-Means | `kmeans.py` | *Round equal-size groups: partitions commodities into compact spherical clusters* | Hierarchical (Ward) | `hierarchical.py` | *Merge tree: agglomerative tree iteratively joining clusters to minimize variance* |

---

## How to Run

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Clean data (optional standalone check):**
   ```bash
   uv run python clean.py
   ```

3. **Run exploratory data analysis (standalone check):**
   ```bash
   uv run python eda.py
   ```

4. **Run the full headless pipeline (main.py):**
   ```bash
   uv run python main.py
   ```

5. **Launch the Streamlit dashboard:**
   ```bash
   uv run streamlit run app.py
   ```
