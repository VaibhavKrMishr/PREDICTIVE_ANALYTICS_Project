# app.py - streamlit dashboard for mandi price analysis

import importlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

import clean
import models as m

importlib.reload(clean)
importlib.reload(m)

st.set_page_config(
    page_title="Mandi Price ML Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# global plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# modern aesthetic css styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* hero container */
.hero-container {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 55%, #047857 100%);
    padding: 24px 30px;
    border-radius: 14px;
    color: #ffffff;
    margin-bottom: 18px;
    box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    color: #a7f3d0;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.hero-statement {
    font-size: 1.02rem;
    color: #ecfdf5;
    margin: 0;
    line-height: 1.45;
}

/* stat pills */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.83rem;
    font-weight: 600;
}

/* metrics cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 12px 16px;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #cbd5e1;
    box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.06);
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.42rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}

/* tabs navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #f1f5f9;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #64748b;
    border: none !important;
    transition: all 0.15s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #065f46 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

/* cards & callouts */
.verdict-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 10px 16px;
    color: #166534;
    font-size: 0.92rem;
    margin-top: 10px;
    margin-bottom: 6px;
}

.student-note {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 4px;
    margin-bottom: 8px;
}

.model-header-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
}
.tag-baseline { background: #e0f2fe; color: #0369a1; }
.tag-flexible { background: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# hero section
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🌱 Agri-Tech Machine Learning Study</div>
    <div class="hero-title">Indian Mandi Price Intelligence</div>
    <div class="hero-statement"><b>Problem Statement:</b> Understand and predict what determines agricultural commodity prices across Indian wholesale markets (mandis).</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    raw = clean.load_raw()
    return raw, clean.clean_data(raw)


@st.cache_resource
def get_results(df):
    reg = m.run_regression(df)
    clf = m.run_classification(df)
    clu = m.run_clustering(df)
    return reg, clf, clu


raw, df = get_data()
with st.spinner("training models..."):
    reg, clf, clu = get_results(df)

# safety check in case streamlit kept stale cache from an earlier session
if "WCSS" not in clu.get("Hierarchical", {}):
    st.cache_resource.clear()
    reg, clf, clu = get_results(df)
if "WCSS" not in clu.get("Hierarchical", {}):
    clu["Hierarchical"]["WCSS"] = m.compute_wcss(clu["X"], clu["table"]["Hier cluster"].values)


# stat summary pills
st.markdown(f"""
<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px;">
    <span class="stat-pill">📊 <b>{df.shape[0]:,}</b> Cleaned Records</span>
    <span class="stat-pill">📍 <b>{df['state'].nunique()}</b> States & UTs</span>
    <span class="stat-pill">🌾 <b>{df['commodity'].nunique()}</b> Commodities</span>
    <span class="stat-pill">🏪 <b>{df['market'].nunique()}</b> Mandis</span>
    <span class="stat-pill">🤖 <b>6</b> Algorithms Benchmarked</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["EDA & Cleaning", "Regression", "Classification", "Unsupervised", "Collective Findings"])

# tab 1: eda and data cleaning
with tabs[0]:
    st.header("EDA and Data Cleaning")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Rows", raw.shape[0])
    col2.metric("Clean Rows", df.shape[0])
    col3.metric("Unique Commodities", df["commodity"].nunique())

    st.subheader("Data Cleaning Steps")
    st.markdown("""
- **Standardized column headers**: cleaned spaces and non-standard naming artifacts.
- **Removed date invariant**: dropped `arrival_date` because all rows belong to the same daily snapshot.
- **Cleaned text & state names**: normalized casing and fixed regional variants (e.g., *Keralam* -> *Kerala*).
- **Price logic validation**: kept records satisfying $\\text{min\\_price} \\le \\text{modal\\_price} \\le \\text{max\\_price}$.
- **Filtered per-piece anomalies**: dropped prices $< 10$ Rs (per-piece items like coconut and flowers).
- **Feature engineering**: calculated `log_modal` to normalize extreme price skew, and price `spread` = $\\frac{\\text{max}-\\text{min}}{\\text{modal}}$.
""")

    st.subheader("Dataset Inspection (info, head, tail)")
    view_tab = st.radio("Inspect:", ["Columns & Types (info)", "First 10 rows (head)", "Last 10 rows (tail)"], horizontal=True)
    if view_tab == "Columns & Types (info)":
        st.dataframe(clean.eda_info(df), hide_index=True)
    elif view_tab == "First 10 rows (head)":
        st.dataframe(df.head(10))
    else:
        st.dataframe(df.tail(10))

    st.subheader("Descriptive Statistics (df.describe())")
    desc_choice = st.radio("Show describe for:", ["Numerical columns", "Categorical columns", "All columns"], horizontal=True)
    if desc_choice == "Numerical columns":
        st.dataframe(clean.eda_describe(df, include="numeric"))
    elif desc_choice == "Categorical columns":
        st.dataframe(clean.eda_describe(df, include="categorical"))
    else:
        st.dataframe(clean.eda_describe(df, include="all"))

    c1, c2 = st.columns(2)
    with c1:
        fig = clean.plot_price_distribution(df)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with c2:
        fig = clean.plot_correlations(df)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    st.caption("min, max and modal prices correlate almost at 1.000, so using them as features would cause severe data leakage. Inputs are strictly state, commodity, and grade.")

    c1, c2 = st.columns(2)
    with c1:
        fig = clean.plot_top_states(df)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with c2:
        fig = clean.plot_top_commodities(df)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# tab 2: regression
with tabs[1]:
    st.header("Regression: Predicting Exact Modal Price")
    st.info("**Goal:** How much will a commodity cost? (exact price)\n\n"
            "We predict the modal price (on log scale) from the state, commodity, and grade.")

    left, right = st.columns(2)

    with left:
        st.markdown('### Multiple Linear Regression <span class="model-header-tag tag-baseline">Baseline</span>', unsafe_allow_html=True)
        st.caption("Linear and additive: adds up state, commodity, and grade with no feature interactions.")

        res_lr = reg["Linear Regression"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", round(res_lr["MAE"], 3))
        m2.metric("RMSE", round(res_lr["RMSE"], 3))
        m3.metric("R2 (test)", round(res_lr["R2 test"], 3))
        m4.metric("R2 (train)", round(res_lr["R2 train"], 3))

        with st.expander("Additional metrics (MSE)"):
            st.dataframe(pd.DataFrame({"Metric": ["MSE"], "Value": [round(res_lr["MSE"], 3)]}), hide_index=True)

        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        ax.scatter(res_lr["y_test"], res_lr["pred"], s=5, color="#0284c7", alpha=0.35, edgecolors="none")
        lo, hi = res_lr["y_test"].min(), res_lr["y_test"].max()
        ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", lw=1.2)
        ax.set_xlabel("actual log price", fontsize=8)
        ax.set_ylabel("predicted log price", fontsize=8)
        ax.set_title("actual vs predicted", fontsize=9, fontweight="bold", color="#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        pos_coef, neg_coef = m.lr_top_coefficients(reg)
        sub1, sub2 = st.columns(2)
        sub1.write("Positive coefficients:")
        sub1.dataframe(pos_coef.round(2))
        sub2.write("Negative coefficients:")
        sub2.dataframe(neg_coef.round(2))

    with right:
        st.markdown('### Random Forest Regressor <span class="model-header-tag tag-flexible">Flexible</span>', unsafe_allow_html=True)
        st.caption("Ensemble of trees: captures non-linear combinations and state-commodity interactions.")

        res_rf = reg["Random Forest"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", round(res_rf["MAE"], 3))
        m2.metric("RMSE", round(res_rf["RMSE"], 3))
        m3.metric("R2 (test)", round(res_rf["R2 test"], 3))
        m4.metric("R2 (train)", round(res_rf["R2 train"], 3))

        with st.expander("Additional metrics (MSE)"):
            st.dataframe(pd.DataFrame({"Metric": ["MSE"], "Value": [round(res_rf["MSE"], 3)]}), hide_index=True)

        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        ax.scatter(res_rf["y_test"], res_rf["pred"], s=5, color="#0d9488", alpha=0.35, edgecolors="none")
        lo, hi = res_rf["y_test"].min(), res_rf["y_test"].max()
        ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", lw=1.2)
        ax.set_xlabel("actual log price", fontsize=8)
        ax.set_ylabel("predicted log price", fontsize=8)
        ax.set_title("actual vs predicted", fontsize=9, fontweight="bold", color="#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        imp = m.rf_importance(reg)
        fig, ax = plt.subplots(figsize=(3.6, 2.1))
        imp.sort_values().plot.barh(ax=ax, color="#059669", edgecolor="white", alpha=0.9)
        ax.set_title("feature importance", fontsize=9, fontweight="bold", color="#1e293b")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.subheader("Regression Comparison & Ranking")
    table_reg = pd.DataFrame({n: {k: reg[n][k] for k in ["MAE", "MSE", "RMSE", "R2 test", "R2 train"]}
                             for n in ["Linear Regression", "Random Forest"]}).T
    table_reg["Rank"] = table_reg["RMSE"].rank().astype(int)
    st.dataframe(table_reg.sort_values("Rank").round(3))
    st.caption("ranked by RMSE (lower is better)")
    st.markdown('<div class="student-note">💡 <b>Note on RMSE:</b> Average error on log price (lower is better); R² shows the fraction of variance explained.</div>', unsafe_allow_html=True)

    lr_rmse = res_lr["RMSE"]
    rf_rmse = res_rf["RMSE"]
    lr_r2 = res_lr["R2 test"]
    rf_r2 = res_rf["R2 test"]
    if rf_rmse < lr_rmse:
        verdict_text = f"🏆 <b>Verdict:</b> Random Forest performed better (RMSE <b>{rf_rmse:.3f}</b> vs {lr_rmse:.3f}, R² <b>{rf_r2:.1%}</b> vs {lr_r2:.1%}) because it captures state-crop combinations."
    else:
        verdict_text = f"🏆 <b>Verdict:</b> Linear Regression performed better (RMSE <b>{lr_rmse:.3f}</b> vs {rf_rmse:.3f})."
    st.markdown(f'<div class="verdict-box">{verdict_text}</div>', unsafe_allow_html=True)

# tab 3: classification
with tabs[2]:
    st.header("Classification: Predicting Price Bands")
    st.info("**Goal:** Which price band does it fall in? (Low, Mid, High)\n\n"
            "We split prices into 3 bands to see how well crops can be grouped into market segments.")

    st.write("Cutoffs from tertiles of modal price (Rs/quintal):")
    st.dataframe(clf["band_edges"])

    left, right = st.columns(2)

    with left:
        st.markdown('### Multinomial Naive Bayes <span class="model-header-tag tag-baseline">Baseline</span>', unsafe_allow_html=True)
        st.caption("Probabilistic: assumes features are independent given the price band.")

        res_nb = clf["Naive Bayes"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", round(res_nb["Accuracy"], 3))
        m2.metric("Precision", round(res_nb["Precision"], 3))
        m3.metric("Recall", round(res_nb["Recall"], 3))
        m4.metric("F1 (macro)", round(res_nb["F1 (macro)"], 3))

        with st.expander("Additional metrics (Log Loss, AUC)"):
            st.dataframe(pd.DataFrame({"Metric": ["Log Loss", "ROC-AUC"], "Value": [round(res_nb["Log Loss"], 3), round(res_nb["AUC"], 3)]}), hide_index=True)

        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        sns.heatmap(res_nb["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=res_nb["labels"], yticklabels=res_nb["labels"],
                    annot_kws={"size": 8.5, "weight": "bold"}, linewidths=1.2, linecolor="white")
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        ax.set_title("confusion matrix", fontsize=9, fontweight="bold", color="#1e293b")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        pos_feat, neg_feat = m.nb_top_features(clf)
        sub1, sub2 = st.columns(2)
        sub1.write("Pushes to High:")
        sub1.dataframe(pos_feat.round(2))
        sub2.write("Pushes to Low:")
        sub2.dataframe(neg_feat.round(2))

    with right:
        st.markdown('### Decision Tree Classifier <span class="model-header-tag tag-flexible">Flexible</span>', unsafe_allow_html=True)
        st.caption("Rule-based: splits features step-by-step into transparent if-else decisions.")

        res_dt = clf["Decision Tree"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", round(res_dt["Accuracy"], 3))
        m2.metric("Precision", round(res_dt["Precision"], 3))
        m3.metric("Recall", round(res_dt["Recall"], 3))
        m4.metric("F1 (macro)", round(res_dt["F1 (macro)"], 3))

        with st.expander("Additional metrics (Log Loss, AUC)"):
            st.dataframe(pd.DataFrame({"Metric": ["Log Loss", "ROC-AUC"], "Value": [round(res_dt["Log Loss"], 3), round(res_dt["AUC"], 3)]}), hide_index=True)

        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        sns.heatmap(res_dt["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=res_dt["labels"], yticklabels=res_dt["labels"],
                    annot_kws={"size": 8.5, "weight": "bold"}, linewidths=1.2, linecolor="white")
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        ax.set_title("confusion matrix", fontsize=9, fontweight="bold", color="#1e293b")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        imp_dt = m.tree_importance(clf)
        fig, ax = plt.subplots(figsize=(3.6, 2.1))
        imp_dt.sort_values().plot.barh(ax=ax, color="#0d9488", edgecolor="white", alpha=0.9)
        ax.set_title("feature importance", fontsize=9, fontweight="bold", color="#1e293b")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.write("Top rules (depth 3):")
        st.code(m.tree_rules(clf, depth=3))

    st.subheader("Classification Comparison & Ranking")
    keys = ["Accuracy", "Precision", "Recall", "F1 (macro)", "Log Loss", "AUC"]
    table_clf = pd.DataFrame({n: {k: clf[n][k] for k in keys} for n in ["Naive Bayes", "Decision Tree"]}).T
    table_clf["Rank"] = table_clf["F1 (macro)"].rank(ascending=False).astype(int)
    st.dataframe(table_clf.sort_values("Rank").round(3))
    st.caption("ranked by macro F1 (higher is better)")
    st.markdown('<div class="student-note">💡 <b>Note on Macro F1:</b> Balances precision and recall equally across all three bands so common crops do not bias the score.</div>', unsafe_allow_html=True)

    nb_f1 = res_nb["F1 (macro)"]
    dt_f1 = res_dt["F1 (macro)"]
    nb_acc = res_nb["Accuracy"]
    dt_acc = res_dt["Accuracy"]
    if dt_f1 > nb_f1:
        verdict_text = f"🏆 <b>Verdict:</b> Decision Tree wins on macro F1 (<b>{dt_f1:.3f}</b> vs {nb_f1:.3f}) and accuracy (<b>{dt_acc:.1%}</b> vs {nb_acc:.1%}) because rule splits handle combinations better."
    else:
        verdict_text = f"🏆 <b>Verdict:</b> Naive Bayes wins on macro F1 (<b>{nb_f1:.3f}</b> vs {dt_f1:.3f})."
    st.markdown(f'<div class="verdict-box">{verdict_text}</div>', unsafe_allow_html=True)

# tab 4: unsupervised
with tabs[3]:
    st.header("Unsupervised: Natural Commodity Groupings")
    st.info("**Goal:** What natural groups of commodities exist without labels?\n\n"
            "We group commodities by price, spread, and market spread to see natural clusters.")

    left, right = st.columns(2)

    with left:
        st.markdown('### K-Means Clustering <span class="model-header-tag tag-baseline">Baseline</span>', unsafe_allow_html=True)
        st.caption("Centroid-based: forms spherical, roughly equal-sized groups around cluster centers.")

        res_km = clu["K-Means"]
        m1, m2 = st.columns(2)
        m1.metric("WCSS", round(res_km["WCSS"], 1))
        m2.metric("k used", clu["k"])
        st.caption(f"Optional supporting silhouette: {res_km['silhouette']:.3f} (checks if points are closer to their own cluster than neighbours)")

        st.write("Cluster profile (means):")
        st.dataframe(m.cluster_profile(clu["table"], "KMeans cluster"))

        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.plot(clu["scores"]["k"], clu["scores"]["KMeans WCSS"], "o-", color="#059669", markersize=5, lw=1.6)
        ax.set_title("elbow method (WCSS vs k)", fontsize=9, fontweight="bold", color="#1e293b")
        ax.set_xlabel("k (clusters)", fontsize=8)
        ax.set_ylabel("WCSS", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with right:
        st.markdown('### Hierarchical Clustering <span class="model-header-tag tag-flexible">Flexible</span>', unsafe_allow_html=True)
        st.caption("Hierarchical tree: merges similar commodities step-by-step from the bottom up.")

        res_hc = clu["Hierarchical"]
        hc_wcss = res_hc.get("WCSS", m.compute_wcss(clu["X"], clu["table"]["Hier cluster"].values))
        m1, m2 = st.columns(2)
        m1.metric("WCSS", round(hc_wcss, 1))
        m2.metric("k used", clu["k"])
        st.caption(f"Optional supporting silhouette: {res_hc['silhouette']:.3f} (checks if points are closer to their own cluster than neighbours)")

        st.write("Cluster profile (means):")
        st.dataframe(m.cluster_profile(clu["table"], "Hier cluster"))

        st.write("Linkages silhouette (supporting):")
        st.dataframe(clu["linkages"].round(3))

        tree_link = linkage(clu["X"], method="ward")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        dendrogram(tree_link, labels=clu["table"].index.tolist(), leaf_rotation=90, leaf_font_size=5, ax=ax)
        ax.set_title("dendrogram (ward linkage)", fontsize=9, fontweight="bold", color="#1e293b")
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.subheader("Clustering Comparison & Ranking")
    km_wcss = res_km.get("WCSS", float(clu["K-Means"]["WCSS"]))
    table_clu = pd.DataFrame({
        "K-Means": {"WCSS": km_wcss, "Silhouette (supporting)": clu["K-Means"]["silhouette"]},
        "Hierarchical": {"WCSS": hc_wcss, "Silhouette (supporting)": clu["Hierarchical"]["silhouette"]}
    }).T
    table_clu["Rank"] = table_clu["WCSS"].rank().astype(int)
    st.dataframe(table_clu.sort_values("Rank").round(3))
    st.caption("ranked by WCSS (lower is better, indicating tighter clusters)")
    st.markdown('<div class="student-note">💡 <b>Note on WCSS:</b> Sum of squared distances to cluster centers — lower means items sit closer together.</div>', unsafe_allow_html=True)

    win_model = "K-Means" if km_wcss <= hc_wcss else "Hierarchical"
    win_wcss = min(km_wcss, hc_wcss)
    lose_wcss = max(km_wcss, hc_wcss)
    st.markdown(f'<div class="verdict-box">🏆 <b>Verdict:</b> {win_model} forms tighter clusters with lower WCSS (<b>{win_wcss:.1f}</b> vs {lose_wcss:.1f}) at k={clu["k"]}.</div>', unsafe_allow_html=True)


    st.write("WCSS across values of k (elbow curve data):")
    st.dataframe(clu["scores"][["k", "KMeans WCSS", "KMeans silhouette"]].round(3))

# tab 5: collective findings
with tabs[4]:
    st.header("Collective Findings")

    st.subheader("1. Findings for each sub-problem")
    rf_r2 = reg["Random Forest"]["R2 test"]
    rf_rmse = reg["Random Forest"]["RMSE"]
    lr_r2 = reg["Linear Regression"]["R2 test"]
    dt_f1 = clf["Decision Tree"]["F1 (macro)"]
    dt_acc = clf["Decision Tree"]["Accuracy"]
    best_k = clu["k"]
    hc_wcss_val = clu["Hierarchical"].get("WCSS", m.compute_wcss(clu["X"], clu["table"]["Hier cluster"].values))
    best_wcss = min(clu["K-Means"]["WCSS"], hc_wcss_val)
    best_sil = max(clu["K-Means"]["silhouette"], clu["Hierarchical"]["silhouette"])


    st.markdown(f"""
- **Exact Price (Regression):** Random Forest reaches R² of **{rf_r2:.1%}** (RMSE **{rf_rmse:.3f}**), beating linear regression ({lr_r2:.1%}) because commodity prices vary by state in non-linear ways.
- **Price Bands (Classification):** Decision Tree achieves **{dt_acc:.1%}** accuracy and **{dt_f1:.3f}** macro F1. High-value crops and basic staples separate cleanly, while mid-tier crops have more overlap.
- **Natural Groups (Clustering):** Clustering found **{best_k}** groups with WCSS of **{best_wcss:.1f}** (supporting silhouette {best_sil:.3f}), grouping staple bulk crops separately from niche and high-spread items.
""")

    st.subheader("2. Where the models agree and disagree")
    rf_imp = m.rf_importance(reg)
    dt_imp = m.tree_importance(clf)
    feat_imp = pd.DataFrame({"Random Forest (Reg)": rf_imp, "Decision Tree (Clf)": dt_imp})
    st.write("Feature importance comparison:")
    st.dataframe(feat_imp.round(3))

    main_feature = feat_imp.mean(axis=1).idxmax()
    rf_share = feat_imp.loc[main_feature, "Random Forest (Reg)"]
    dt_share = feat_imp.loc[main_feature, "Decision Tree (Clf)"]

    st.markdown(f"""
**Agreements:**
- **Commodity is the main driver:** In both Random Forest ({rf_share:.0%}) and Decision Tree ({dt_share:.0%}), the crop itself accounts for most of the predictive weight. What is being sold matters far more than where it is sold or the grade tag.
- **State acts as a secondary adjustment:** Location gives a smaller adjustment, likely reflecting transport costs and local supply/demand.
- **Grade barely matters:** Quality grade contributed under 1% importance across models.

**Disagreements & trade-offs:**
- **Trees vs linear models:** Tree models score higher because a crop might command a high price in one state but not another. But trees have a bigger gap between train and test scores ({reg['Random Forest']['R2 train']:.1%} train vs {rf_r2:.1%} test). Linear models have less variance, but miss combinations.
- **Clusters have fuzzy boundaries:** Moderate silhouette scores show that commodities form a continuous price spectrum rather than clearly isolated clusters.
""")

    st.subheader("3. Linking clusters to price bands and regression error")
    st.caption("Mapping clusters back to the actual data shows how natural groupings tie to price tiers and model error:")

    link_data = m.cluster_link(df, clu["table"], reg["Random Forest"]["pipe"], cluster_col="KMeans cluster")
    st.dataframe(link_data)

    top_cluster = int(link_data["median_price"].idxmax())
    base_cluster = int(link_data["median_price"].idxmin())
    st.markdown(f"""
- **Cluster {top_cluster}** has the highest median price (Rs **{link_data.loc[top_cluster, 'median_price']:.0f}**/quintal) with an average price band of **{link_data.loc[top_cluster, 'avg_band']:.2f}** (mostly High band) and regression MAE of **{link_data.loc[top_cluster, 'avg_reg_mae']:.2f}**.
- **Cluster {base_cluster}** represents everyday high-volume staple crops with median price Rs **{link_data.loc[base_cluster, 'median_price']:.0f}**/quintal and average price band **{link_data.loc[base_cluster, 'avg_band']:.2f}**.
- Linking clusters to predictions confirms that unsupervised groupings line up with real price tiers and highlight which crop groups are harder to predict.
""")

    st.subheader("4. Conclusion and limitations")
    st.markdown(f"""
**Summary answering the problem statement:**
Commodity identity determines the vast majority of agricultural mandi prices (over {rf_share:.0%} of importance). State provides a secondary regional adjustment, while grade has almost no effect. Non-linear tree models predict prices best by capturing crop-state combinations, and unsupervised clustering confirms natural groupings that line up closely with economic price bands.

**Limitations:**
- **Single day snapshot:** Data is from one day only, so it has no seasonal cycles, harvest timings, or price trends over time.
- **Categorical features only:** We only have names (state, commodity, grade) without quantity arrived, rainfall data, or transport distance.
- **Geographic skew:** Tamil Nadu makes up about half the dataset, so results reflect southern markets more heavily.
""")
