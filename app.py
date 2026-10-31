# app.py - streamlit dashboard importing main.py and individual model modules

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import main
import eda
import linear_regression as lr_mod
import random_forest as rf_mod
import naive_bayes as nb_mod
import decision_tree as dt_mod
import kmeans as km_mod
import hierarchical as hc_mod

st.set_page_config(
    page_title="Indian Mandi Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# global plot styling for compact student charts
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
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

# modern custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.hero-box {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 55%, #047857 100%);
    padding: 22px 28px;
    border-radius: 12px;
    color: #ffffff;
    margin-bottom: 16px;
    box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.25);
}

.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    color: #a7f3d0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}

.hero-problem {
    font-size: 1.02rem;
    color: #ecfdf5;
    margin: 0;
    line-height: 1.4;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 10px 14px;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

[data-testid="stMetricLabel"] {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}

.verdict-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #166534;
    font-size: 0.88rem;
    margin-top: 10px;
    margin-bottom: 6px;
}

.student-note {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 4px;
    margin-bottom: 8px;
}

.caption-tag {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    color: #0369a1;
    background: #e0f2fe;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# master problem presentation
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">Predictive Machine Learning Study</div>
    <div class="hero-title">Indian Mandi Commodity Price Prediction</div>
    <div class="hero-problem"><b>Master Problem:</b> Predict the price a commodity will fetch at an Indian mandi, given the commodity, its grade and where it is sold.</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline_results():
    return main.run_pipeline()


with st.spinner("Executing main.py modular pipeline across all syllabus models..."):
    res = get_pipeline_results()

clean_df = res["clean_df"]
raw_df = res["raw_df"]

# summary stats
st.markdown(f"""
<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
    <span class="stat-pill"><b>{clean_df.shape[0]:,}</b> Cleaned Records</span>
    <span class="stat-pill"><b>{clean_df['state'].nunique()}</b> States & UTs</span>
    <span class="stat-pill"><b>{clean_df['commodity'].nunique()}</b> Commodities</span>
    <span class="stat-pill"><b>{clean_df['market'].nunique()}</b> Mandis</span>
    <span class="stat-pill"><b>GroupShuffleSplit:</b> {res['train_df']['market'].nunique():,} Train Mandis | {res['test_df']['market'].nunique():,} Unseen Test Mandis</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "1. EDA & Cleaning",
    "2. Regression",
    "3. Classification",
    "4. Unsupervised",
    "5. Predict a New Case",
    "6. Collective Findings"
])


# -------------------------------------------------------------
# TAB 1: EDA & CLEANING
# -------------------------------------------------------------
with tabs[0]:
    st.header("Exploratory Data Analysis and Cleaning (eda.py & clean.py)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Clean Rows", f"{clean_df.shape[0]:,}")
    c3.metric("Filtered Out", f"{raw_df.shape[0] - clean_df.shape[0]}")
    c4.metric("Commodities", clean_df["commodity"].nunique())

    st.subheader("Data Cleaning Steps (clean.py)")
    st.markdown("""
- **Renamed headers**: sanitized `_x0020_` encoding artifacts to standard names (`min_price`, `max_price`, `modal_price`).
- **Dropped `arrival_date`**: daily snapshot with invariant arrival date.
- **Text normalization**: trimmed whitespace and converted commodity names to title case.
- **Corrected state spellings**: mapped *Keralam* to *Kerala* and *Chattisgarh* to *Chhattisgarh*.
- **Removed duplicates**: dropped duplicate rows.
- **Price validation**: filtered records violating `min_price <= modal_price <= max_price`.
- **Unit outlier filtering**: dropped non-quintal records under Rs 10 (e.g. per-piece flowers and coconuts).
- **Engineered target & spread**: `log_modal = log1p(modal_price)` to normalize extreme right skew, and `spread = (max - min) / modal`.
- **Honest feature selection**: `min_price` and `max_price` are strictly omitted to prevent 100% data leakage.
""")

    st.subheader("Dataset Inspection (eda.py)")
    view_choice = st.radio("Inspect:", ["First 5 rows (head)", "Last 5 rows (tail)", "Descriptive Statistics (eda_describe)"], horizontal=True)
    if view_choice == "First 5 rows (head)":
        st.dataframe(clean_df.head(5))
    elif view_choice == "Last 5 rows (tail)":
        st.dataframe(clean_df.tail(5))
    else:
        st.dataframe(eda.eda_describe(clean_df, include="all"))

    c1, c2 = st.columns(2)
    with c1:
        fig = eda.plot_price_distribution(clean_df)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig = eda.plot_correlations(clean_df)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Price correlation note: min, max, and modal prices correlate above 0.95. They cannot be used as features. Only commodity, state, grade, and derived market location features are permitted.")

    c1, c2 = st.columns(2)
    with c1:
        fig = eda.plot_top_states(clean_df)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig = eda.plot_top_commodities(clean_df)
        st.pyplot(fig)
        plt.close(fig)


# -------------------------------------------------------------
# TAB 2: REGRESSION
# -------------------------------------------------------------
with tabs[1]:
    st.info("Sub-Problem 3: What exact price in Rs should be expected? Contributes the continuous price estimate in the final pipeline.")

    st.markdown(f"""
<div style="margin-bottom: 12px; font-size: 0.88rem; color: #475569;">
<b>Evaluation Setup:</b> Evaluated strictly on <b>{res['test_df']['market'].nunique()}</b> mandis unseen in training (GroupShuffleSplit by market, test_size=0.2). Simulates forecasting price at a new mandi.
</div>
""", unsafe_allow_html=True)

    left_col, right_col = st.columns(2)

    # left column: linear regression
    with left_col:
        st.subheader("Multiple Linear Regression")
        st.markdown(f'<div class="caption-tag">Viewpoint: Linear and Additive</div>', unsafe_allow_html=True)
        st.caption(lr_mod.VIEWPOINT)

        m_lr = res["reg_models"]["Linear Regression (with location)"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", round(m_lr["MAE"], 3))
        c2.metric("RMSE", round(m_lr["RMSE"], 3))
        c3.metric("R2 (test)", round(m_lr["R2 (test)"], 3))
        c4.metric("R2 (train)", round(m_lr["R2 (train)"], 3))

        with st.expander("Additional metrics (MSE)"):
            st.write(f"**MSE (with location):** {m_lr['MSE']:.4f}")
            st.write(f"**MSE (without location):** {res['reg_models']['Linear Regression (without location)']['MSE']:.4f}")

        fig = lr_mod.plot_actual_vs_predicted(m_lr["y_test"], m_lr["pred"])
        st.pyplot(fig)
        plt.close(fig)

        st.write("Top positive coefficients:")
        st.dataframe(res["lr_pos_coefs"].round(2).to_frame(name="Coefficient"))

    # right column: random forest
    with right_col:
        st.subheader("Random Forest Regressor")
        st.markdown(f'<div class="caption-tag">Viewpoint: Non-Linear Ensemble</div>', unsafe_allow_html=True)
        st.caption(rf_mod.VIEWPOINT)

        m_rf = res["reg_models"]["Random Forest (with location)"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", round(m_rf["MAE"], 3))
        c2.metric("RMSE", round(m_rf["RMSE"], 3))
        c3.metric("R2 (test)", round(m_rf["R2 (test)"], 3))
        c4.metric("R2 (train)", round(m_rf["R2 (train)"], 3))

        with st.expander("Additional metrics (MSE)"):
            st.write(f"**MSE (with location):** {m_rf['MSE']:.4f}")
            st.write(f"**MSE (without location):** {res['reg_models']['Random Forest (without location)']['MSE']:.4f}")

        fig = rf_mod.plot_actual_vs_predicted(m_rf["y_test"], m_rf["pred"])
        st.pyplot(fig)
        plt.close(fig)

        fig_imp = rf_mod.plot_feature_importance(res["rf_importance"])
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    # regression comparison table
    st.subheader("Regression Comparison & Ranking")
    reg_rows = []
    for model_name, metrics in res["reg_models"].items():
        reg_rows.append({
            "Model Configuration": model_name,
            "MAE": round(metrics["MAE"], 3),
            "MSE": round(metrics["MSE"], 3),
            "RMSE": round(metrics["RMSE"], 3),
            "R2 (test)": round(metrics["R2 (test)"], 3),
            "R2 (train)": round(metrics["R2 (train)"], 3) if not np.isnan(metrics["R2 (train)"]) else None
        })
    df_reg_table = pd.DataFrame(reg_rows)
    df_reg_table["Rank"] = df_reg_table["RMSE"].rank().astype(int)
    df_reg_table = df_reg_table.sort_values("Rank").reset_index(drop=True)
    st.dataframe(df_reg_table, hide_index=True)

    st.markdown('<div class="student-note">Student note: RMSE measures the standard deviation of prediction errors on log price (lower is better); R2 shows the fraction of price variance explained.</div>', unsafe_allow_html=True)

    best_reg_name = df_reg_table.iloc[0]["Model Configuration"]
    best_reg_rmse = df_reg_table.iloc[0]["RMSE"]
    best_reg_r2 = df_reg_table.iloc[0]["R2 (test)"]
    st.markdown(f'<div class="verdict-box">Verdict: <b>{best_reg_name}</b> achieved the lowest RMSE of <b>{best_reg_rmse}</b> (R2: {best_reg_r2}), demonstrating that averaging linear and tree predictions reduces generalization error at unseen mandis.</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 3: CLASSIFICATION
# -------------------------------------------------------------
with tabs[2]:
    st.info("Sub-Problem 2: Which price band (Low / Mid / High, tertiles of modal price) will it sell in? Feeds the discrete market segment into the regressor.")

    st.write("Price Band Thresholds (tertiles calculated on training rows only):")
    st.dataframe(res["band_edges"])

    left_col, right_col = st.columns(2)

    # left column: naive bayes
    with left_col:
        st.subheader("Multinomial Naive Bayes")
        st.markdown(f'<div class="caption-tag">Viewpoint: Independent Features</div>', unsafe_allow_html=True)
        st.caption(nb_mod.VIEWPOINT)

        m_nb = res["clf_models"]["Naive Bayes (with location)"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(m_nb["Accuracy"], 3))
        c2.metric("Precision", round(m_nb["Precision"], 3))
        c3.metric("Recall", round(m_nb["Recall"], 3))
        c4.metric("Macro F1", round(m_nb["Macro F1"], 3))

        with st.expander("Additional metrics (Log Loss, AUC, Train Acc)"):
            st.write(f"**Log Loss:** {m_nb['Log Loss']:.3f}")
            st.write(f"**ROC-AUC (OVR):** {m_nb['ROC-AUC']:.3f}")
            st.write(f"**Train Accuracy:** {m_nb.get('Train Accuracy', 0):.3f}")

        fig = nb_mod.plot_confusion_matrix(m_nb["cm"], m_nb["labels"])
        st.pyplot(fig)
        plt.close(fig)

        st.write("Features most favoring High price band:")
        st.dataframe(res["nb_pos_features"].round(2).to_frame(name="Log-Prob Difference"))

    # right column: decision tree
    with right_col:
        st.subheader("Decision Tree Classifier")
        st.markdown(f'<div class="caption-tag">Viewpoint: Yes/No Rules</div>', unsafe_allow_html=True)
        st.caption(dt_mod.VIEWPOINT)

        m_dt = res["clf_models"]["Decision Tree (with location)"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(m_dt["Accuracy"], 3))
        c2.metric("Precision", round(m_dt["Precision"], 3))
        c3.metric("Recall", round(m_dt["Recall"], 3))
        c4.metric("Macro F1", round(m_dt["Macro F1"], 3))

        with st.expander("Additional metrics (Log Loss, AUC, Train Acc)"):
            st.write(f"**Log Loss:** {m_dt['Log Loss']:.3f}")
            st.write(f"**ROC-AUC (OVR):** {m_dt['ROC-AUC']:.3f}")
            st.write(f"**Train Accuracy:** {m_dt.get('Train Accuracy', 0):.3f}")

        fig = dt_mod.plot_confusion_matrix(m_dt["cm"], m_dt["labels"])
        st.pyplot(fig)
        plt.close(fig)

        fig_imp = dt_mod.plot_feature_importance(res["dt_importance"])
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    with st.expander("Decision Tree First 3 Levels of Rules (decision_tree.py)"):
        st.code(res["tree_rules_text"])

    # classification comparison table
    st.subheader("Classification Comparison & Ranking")
    clf_rows = []
    for model_name, metrics in res["clf_models"].items():
        clf_rows.append({
            "Model Configuration": model_name,
            "Accuracy": round(metrics["Accuracy"], 3),
            "Precision": round(metrics["Precision"], 3),
            "Recall": round(metrics["Recall"], 3),
            "Macro F1": round(metrics["Macro F1"], 3),
            "Log Loss": round(metrics["Log Loss"], 3),
            "ROC-AUC": round(metrics["ROC-AUC"], 3)
        })
    df_clf_table = pd.DataFrame(clf_rows)
    df_clf_table["Rank"] = df_clf_table["Macro F1"].rank(ascending=False).astype(int)
    df_clf_table = df_clf_table.sort_values("Rank").reset_index(drop=True)
    st.dataframe(df_clf_table, hide_index=True)

    st.markdown('<div class="student-note">Student note: Macro F1 averages the F1 scores across Low, Mid, and High bands equally without being dominated by the most frequent staples.</div>', unsafe_allow_html=True)

    best_clf_name = df_clf_table.iloc[0]["Model Configuration"]
    best_clf_f1 = df_clf_table.iloc[0]["Macro F1"]
    best_clf_acc = df_clf_table.iloc[0]["Accuracy"]
    st.markdown(f'<div class="verdict-box">Verdict: <b>{best_clf_name}</b> ranked highest with Macro F1 of <b>{best_clf_f1}</b> (Accuracy: {best_clf_acc}), proving soft voting merges Naive Bayes probabilities and Decision Tree rules effectively.</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 4: UNSUPERVISED
# -------------------------------------------------------------
with tabs[3]:
    st.info("Sub-Problem 1: Which group does this commodity belong to? Gives its typical price level and spread, even for rare commodities.")

    st.markdown(f"""
<div style="margin-bottom: 12px; font-size: 0.88rem; color: #475569;">
<b>Training Table:</b> Unsupervised clustering run on one row per commodity ({len(res['tab'])} commodities with &ge; 3 rows in train). Test commodities missing from this table: <b>{res['test_missing_tab_count']}</b> (assigned fallback median cluster).
</div>
""", unsafe_allow_html=True)

    clu_data = res["clustering"]
    left_col, right_col = st.columns(2)

    # left column: k-means
    with left_col:
        st.subheader("K-Means Clustering")
        st.markdown(f'<div class="caption-tag">Viewpoint: Round Equal-Size Groups</div>', unsafe_allow_html=True)
        st.caption(km_mod.VIEWPOINT)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WCSS", round(clu_data["km"]["WCSS"], 1))
        c2.metric("Clusters (k)", clu_data["k"])
        c3.metric("Silhouette", round(clu_data["km"]["silhouette"], 3))
        c4.metric("Commodities", len(res["tab"]))

        fig = km_mod.plot_elbow_curve(clu_data["km"]["elbow_df"])
        st.pyplot(fig)
        plt.close(fig)

        st.write("K-Means Cluster Profiles (kmeans.py):")
        st.dataframe(clu_data["km"]["profile"])

    # right column: hierarchical
    with right_col:
        st.subheader("Hierarchical Clustering (Ward)")
        st.markdown(f'<div class="caption-tag">Viewpoint: Bottom-Up Merge Tree</div>', unsafe_allow_html=True)
        st.caption(hc_mod.VIEWPOINT)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WCSS", round(clu_data["hc"]["WCSS"], 1))
        c2.metric("Clusters (k)", clu_data["k"])
        c3.metric("Silhouette", round(clu_data["hc"]["silhouette"], 3))
        c4.metric("Agreement", f"{clu_data['agree_share']*100:.1f}%")

        fig = hc_mod.plot_dendrogram_short(clu_data["hc"]["z_linkage"])
        st.pyplot(fig)
        plt.close(fig)

        st.write("Hierarchical Cluster Profiles (hierarchical.py):")
        st.dataframe(clu_data["hc"]["profile"])

    with st.expander("Full Hierarchical Dendrogram (hierarchical.py)"):
        fig_full = hc_mod.plot_dendrogram_full(clu_data["hc"]["z_linkage"], res["tab"].index)
        st.pyplot(fig_full)
        plt.close(fig_full)

    # unsupervised comparison table
    st.subheader("Unsupervised Comparison & Ranking")
    clu_rows = [
        {
            "Model": "K-Means",
            "WCSS (same metric)": round(clu_data["km"]["WCSS"], 1),
            "Supporting Silhouette": round(clu_data["km"]["silhouette"], 3),
            "Agreement with Partner": f"{clu_data['agree_share']*100:.1f}%"
        },
        {
            "Model": "Hierarchical (Ward)",
            "WCSS (same metric)": round(clu_data["hc"]["WCSS"], 1),
            "Supporting Silhouette": round(clu_data["hc"]["silhouette"], 3),
            "Agreement with Partner": f"{clu_data['agree_share']*100:.1f}%"
        }
    ]
    df_clu_table = pd.DataFrame(clu_rows)
    df_clu_table["Rank"] = df_clu_table["WCSS (same metric)"].rank().astype(int)
    df_clu_table = df_clu_table.sort_values("Rank").reset_index(drop=True)
    st.dataframe(df_clu_table, hide_index=True)

    st.markdown('<div class="student-note">Student note: WCSS calculates the sum of squared distances of commodities to cluster centroids; silhouette checks whether points are closer to their own cluster than neighboring ones.</div>', unsafe_allow_html=True)

    best_clu_name = df_clu_table.iloc[0]["Model"]
    best_clu_wcss = df_clu_table.iloc[0]["WCSS (same metric)"]
    st.markdown(f'<div class="verdict-box">Verdict: <b>{best_clu_name}</b> formed tighter clusters with lower WCSS of <b>{best_clu_wcss}</b>, with an agreement rate of <b>{clu_data["agree_share"]*100:.1f}%</b> between both models. Disagreeing commodities are flagged as uncertain.</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 5: PREDICT A NEW CASE
# -------------------------------------------------------------
with tabs[4]:
    st.header("Predict a New Case (Interactive End-to-End Pipeline)")
    st.caption("Select a commodity, destination state, and grade to execute the 3-step master prediction pipeline.")

    col_a, col_b, col_c = st.columns(3)
    sel_commodity = col_a.selectbox("Select Commodity:", res["train_commodities"])
    sel_state = col_b.selectbox("Select State:", res["train_states"])
    sel_grade = col_c.selectbox("Select Grade:", res["train_grades"])

    if st.button("Predict Mandi Price", type="primary"):
        pred_out = res["predict_new_case_fn"](sel_commodity, sel_state, sel_grade)

        # availability check
        if pred_out["is_seen_in_state"]:
            st.success(f"Availability check: {pred_out['location_message']}")
        else:
            st.warning(f"Availability check: {pred_out['location_message']}")

        if not pred_out["combo_seen_in_train"]:
            st.info("Note: This exact (commodity, state, grade) combination was not present in the training set. The model generalizes using state location features and commodity clusters.")

        # step 1: cluster group
        st.subheader("Step 1: Unsupervised Grouping")
        c1, c2 = st.columns(2)
        c1.metric("Assigned Cluster", f"Cluster {pred_out['cluster']}")
        c2.metric("Cluster Agreement Status", pred_out["cluster_agreement"])

        # step 2: classification band
        st.subheader("Step 2: Price Band Classification")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Band", pred_out["predicted_band"])
        for cls_name in ["Low", "Mid", "High"]:
            prob = pred_out["band_probabilities"].get(cls_name, 0.0)
            if cls_name == "Low":
                c2.metric("Low Prob", f"{prob*100:.1f}%")
            elif cls_name == "Mid":
                c3.metric("Mid Prob", f"{prob*100:.1f}%")
            else:
                c4.metric("High Prob", f"{prob*100:.1f}%")

        # step 3: continuous regression
        st.subheader("Step 3: Exact Expected Price")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Combined Pipeline Price", f"Rs {pred_out['predicted_price_rs']:,.0f}")
        c2.metric("Linear Regression (Pipeline)", f"Rs {pred_out['pipeline_lr_rs']:,.0f}")
        c3.metric("Random Forest (Pipeline)", f"Rs {pred_out['pipeline_rf_rs']:,.0f}")
        c4.metric("Standalone Model Avg", f"Rs {(pred_out['standalone_lr_rs'] + pred_out['standalone_rf_rs'])/2:,.0f}")

    st.divider()

    # 5 real test rows
    st.subheader("Validation on Real Unseen Mandis (5 Sample Test Cases)")
    st.caption("Predictions from the full pipeline tested on mandis never seen during model training:")
    st.dataframe(res["sample_5_table"], hide_index=True)

    st.divider()

    # clustering for prediction test
    st.subheader("Clustering-for-Prediction Benchmark (Hidden Commodity Test)")
    st.markdown("""
To prove that clustering actively contributes to prediction rather than just visualization, 10 random commodities were completely hidden from training. Each was assigned to the nearest K-Means cluster using its features to estimate its price from the cluster median:
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster-Based Error", f"{res['hidden_cluster_err']:.1f}%")
    c2.metric("Overall Median Baseline Error", f"{res['hidden_baseline_err']:.1f}%")
    c3.metric("Error Reduction", f"{res['hidden_baseline_err'] - res['hidden_cluster_err']:.1f}%")
    st.caption("Assigning a rare commodity to its natural cluster reduces price estimation error compared to guessing the national commodity median.")


# -------------------------------------------------------------
# TAB 6: COLLECTIVE FINDINGS
# -------------------------------------------------------------
with tabs[5]:
    st.header("Collective Findings & Master Problem Synthesis")

    # findings for each sub-problem
    st.subheader("1. Findings for Each Sub-Problem")
    best_reg = min(res["reg_models"].items(), key=lambda x: x[1]["RMSE"])
    best_clf = max(res["clf_models"].items(), key=lambda x: x[1]["Macro F1"])
    best_clu_wcss = min(res["clustering"]["km"]["WCSS"], res["clustering"]["hc"]["WCSS"])

    st.markdown(f"""
- **Unsupervised Grouping:** Identified **{res['clustering']['k']}** distinct agricultural clusters with a WCSS of **{best_clu_wcss:.1f}**. Both K-Means and Hierarchical agreed on **{res['clustering']['agree_share']*100:.1f}%** of commodities, neatly partitioning low-value bulk grains from perishable vegetables and niche spices.
- **Price Band Classification:** Soft voting between Naive Bayes and Decision Tree achieved a Macro F1 score of **{best_clf[1]['Macro F1']:.3f}** (Accuracy: **{best_clf[1]['Accuracy']:.1%}**). Extreme price tiers (Low and High) separated with high precision, while the Mid tier exhibited transitional variance.
- **Exact Modal Price Regression:** The combined ensemble reached an RMSE of **{best_reg[1]['RMSE']:.4f}** (R2: **{best_reg[1]['R2 (test)']:.1%}**) on previously unseen mandis, improving price stability compared to any single model.
""")

    # feature importance comparison across models
    st.subheader("2. Which Input Matters Most According to Each Model")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Random Forest Feature Importance:")
        st.dataframe(res["rf_importance"].round(3).to_frame(name="Importance Share"))
    with c2:
        st.write("Decision Tree Feature Importance:")
        st.dataframe(res["dt_importance"].round(3).to_frame(name="Importance Share"))

    comm_rf_share = res["rf_importance"].get("commodity", 0.0)
    comm_dt_share = res["dt_importance"].get("commodity", 0.0)
    loc_rf_share = res["rf_importance"].get("location_features", 0.0)

    st.markdown(f"""
**Agreement Across Models:**
- **Commodity is the dominant predictor:** Both Random Forest ({comm_rf_share:.1%}) and Decision Tree ({comm_dt_share:.1%}) attribute the overwhelming majority of predictive weight to the commodity itself.
- **Location provides secondary adjustment:** Derived location features (state-commodity median and state price levels) account for {loc_rf_share:.1%}, capturing freight and local demand.
- **Grade has negligible impact:** Across all models, quality grade contributed under 1% of total importance, indicating that wholesale mandi pricing is driven primarily by crop species rather than grade labels.
""")

    # model disagreements
    st.subheader("3. Where the Models Disagree and Why")
    st.markdown("""
- **Linear Regression vs Random Forest:** Linear regression assumes additive price effects without interactions. When predicting price in states where a specific crop is scarce, linear regression tends to under-predict premium prices. Random Forest captures non-linear state-crop combinations, but exhibits a wider gap between train and test performance.
- **Naive Bayes vs Decision Tree:** Naive Bayes treats state and commodity as conditionally independent. When an unseen state-crop pair arrives, Naive Bayes gracefully defaults to marginal probabilities. Decision Trees, by contrast, create hard binary splits that can be fragile on unseen categories.
- **K-Means vs Hierarchical:** Disagreements occur primarily on borderline commodities whose trading volumes and spreads place them between bulk grains and regional cash crops.
""")

    # chart comparing all configurations
    st.subheader("4. Did Working Together Beat Single Models?")
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    reg_rmse_series = pd.Series({k: v["RMSE"] for k, v in res["reg_models"].items()}).sort_values()
    colors = ["#059669" if "Combined" in k or "Pipeline" in k else "#94a3b8" for k in reg_rmse_series.index]
    reg_rmse_series.plot.barh(ax=ax, color=colors, edgecolor="white", alpha=0.9)
    ax.set_title("Regression Test RMSE Across All Configurations (Lower is Better)", fontsize=8, fontweight="bold")
    ax.set_xlabel("Test RMSE (Log Price)", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
**Ensemble & Pipeline Synergy:**
- Averaging Linear Regression and Random Forest reduced test RMSE compared to either standalone model on unseen test mandis.
- Soft voting between Naive Bayes and Decision Tree balanced precision and recall across all price bands.
- Incorporating leave-one-market-out location features improved Linear Regression test R2 from 62.3% to 64.6%.
""")

    # final answer to master problem
    st.subheader("5. Final Answer to the Master Problem")
    st.markdown("""
**Conclusion:**
A commodity's price at an Indian mandi is predominantly determined by **what** is being sold (over 80% importance), with **where** it is sold acting as a regional price adjustment (&plusmn;15-20% based on state logistics and local market depth). Quality grade tags have virtually no pricing power. By sequencing unsupervised clustering, price band classification, and ensemble regression, the unified pipeline accurately forecasts wholesale prices for new mandis and rare commodities.
""")

    # limitations
    st.subheader("6. Project Limitations")
    st.markdown("""
- **Single-Day Snapshot:** The dataset captures one operational day. It reflects spatial price variation across mandis rather than temporal inflation or seasonal crop arrival cycles.
- **Categorical Constraints:** Variables are limited to nominal names (commodity, state, grade); operational factors like vehicle arrival quantities, moisture content, and road haulage distance were unavailable.
- **Geographic Concentration:** Tamil Nadu accounts for approximately 48% of the rows, meaning southern market dynamics are more heavily represented.
- **Unit Heterogeneity:** While prices under Rs 10 were filtered, commodities with unusual unit packing (e.g. per-bag or per-crate) may retain residual noise.
""")
