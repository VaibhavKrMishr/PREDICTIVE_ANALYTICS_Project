# Mandi Price ML Project

Course project analyzing agricultural mandi commodity prices across India.

## Problem Statement
"Understand and predict what determines agricultural commodity prices across Indian mandis."

### Three Tasks
1. **Regression:** How much will a commodity cost? (exact price) — Linear Regression vs Random Forest.
2. **Classification:** Which price band does it fall in? (Low/Mid/High) — Naive Bayes vs Decision Tree.
3. **Clustering:** What natural groups of commodities exist without labels? — K-Means vs Hierarchical.

## How to run
```bash
uv sync
uv run python clean.py          # creates data/clean.csv (app also does this automatically)
uv run streamlit run app.py
```

## Project Structure
- `clean.py`: data cleaning and EDA script
- `models.py`: training and evaluation functions for the 6 models
- `app.py`: streamlit dashboard with side-by-side comparisons
- `data/raw.csv`: raw daily market arrivals dataset
