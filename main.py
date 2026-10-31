import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.tree import export_text
from sklearn.base import clone
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, roc_auc_score, confusion_matrix)

import clean
import eda
import location_features as loc
import linear_regression as lr_mod
import random_forest as rf_mod
import naive_bayes as nb_mod
import decision_tree as dt_mod
import kmeans as km_mod
import hierarchical as hc_mod

SEED = 42


def run_pipeline(data_path="data/raw.csv"):
    raw_df = clean.load_raw(data_path)
    clean_df = clean.clean_data(raw_df)

    # split by market (unseen test mandis)
    train, test, band_edges, band_bins = loc.split_by_market(clean_df)

    # clustering on train commodity table
    tab = loc.build_commodity_table(train, min_rows=3)
    scaler = StandardScaler()
    X_clu = scaler.fit_transform(tab[["median_log_price", "median_spread", "log_n_markets", "n_states"]])

    k = 5
    m_km, km_labels, km_wcss, km_sil, elbow_df = km_mod.train_kmeans(X_clu, k=k, seed=SEED)
    m_hc, hc_labels, hc_wcss, hc_sil, z_link = hc_mod.train_hierarchical(X_clu, k=k)

    # match clusters by maximum overlap
    ct = pd.crosstab(km_labels, hc_labels)
    row_ind, col_ind = linear_sum_assignment(-ct.values)
    hc_to_km = {hc_c: km_c for km_c, hc_c in zip(row_ind, col_ind)}
    mapped_hc = np.array([hc_to_km[c] for c in hc_labels])

    agreed = (km_labels == mapped_hc)
    agree_share = float(np.mean(agreed))

    tab = tab.copy()
    tab["cluster"] = km_labels
    tab["KMeans cluster"] = km_labels
    tab["Hier cluster"] = mapped_hc
    tab["cluster_agreement"] = np.where(agreed, "Agreed", "Uncertain")

    km_profile = tab.groupby("KMeans cluster").agg(
        median_price_rs=("median_log_price", lambda s: float(np.expm1(s.median()))),
        median_spread=("median_spread", "median"),
        n_markets=("n_markets", "median"),
        n_states=("n_states", "median"),
        n_commodities=("median_log_price", "size"),
    ).round(2)

    hc_profile = tab.groupby("Hier cluster").agg(
        median_price_rs=("median_log_price", lambda s: float(np.expm1(s.median()))),
        median_spread=("median_spread", "median"),
        n_markets=("n_markets", "median"),
        n_states=("n_states", "median"),
        n_commodities=("median_log_price", "size"),
    ).round(2)

    # location features
    stats = loc.compute_location_stats(train, tab)
    test_missing_tab_count = sum(c not in tab.index for c in test["commodity"].unique())

    train_loc = loc.add_location_features_train(train, stats)
    test_loc = loc.add_location_features_target(test, stats)

    # regression
    cat_cols = ["state", "commodity", "grade"]
    num_cols = ["state_commodity_median", "state_price_level", "n_markets_selling"]
    y_tr = train_loc["log_modal"]
    y_te = test_loc["log_modal"]

    # without location
    prep_no = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    X_tr_no = prep_no.fit_transform(train_loc[cat_base := cat_cols])
    X_te_no = prep_no.transform(test_loc[cat_base])

    m_lr_no, lr_no = lr_mod.train_linear_regression(X_tr_no, y_tr, X_te_no, y_te)
    m_rf_no, rf_no = rf_mod.train_random_forest(X_tr_no, y_tr, X_te_no, y_te, seed=SEED)

    # with location
    prep_loc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    X_tr_loc = prep_loc.fit_transform(train_loc[cat_cols + num_cols])
    X_te_loc = prep_loc.transform(test_loc[cat_cols + num_cols])

    m_lr_loc, lr_loc = lr_mod.train_linear_regression(X_tr_loc, y_tr, X_te_loc, y_te)
    m_rf_loc, rf_loc = rf_mod.train_random_forest(X_tr_reg_loc := X_tr_loc, y_tr, X_te_reg_loc := X_te_loc, y_te, seed=SEED)

    # combined regression
    def eval_reg_comb(pred_test, pred_train=None):
        mse = mean_squared_error(y_te, pred_test)
        mae = mean_absolute_error(y_te, pred_test)
        return {
            "MAE": mae, "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2 (test)": r2_score(y_te, pred_test),
            "R2 (train)": r2_score(y_tr, pred_train) if pred_train is not None else np.nan,
            "pred": pred_test, "y_test": np.array(y_te)
        }

    comb_reg_no = eval_reg_comb((lr_no["pred"] + rf_no["pred"]) / 2)
    comb_reg_loc = eval_reg_comb((lr_loc["pred"] + rf_loc["pred"]) / 2)

    # classification
    labels = ["Low", "Mid", "High"]
    y_tr_band = train_loc["band"]
    y_te_band = test_loc["band"]

    prep_clf_loc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile", subsample=None), num_cols)
    ])
    X_tr_clf_loc = prep_clf_loc.fit_transform(train_loc[cat_cols + num_cols])
    X_te_clf_loc = prep_clf_loc.transform(test_loc[cat_cols + num_cols])

    m_nb_no, nb_no = nb_mod.train_naive_bayes(X_tr_no, y_tr_band, X_te_no, y_te_band, labels=labels)
    m_dt_no, dt_no = dt_mod.train_decision_tree(X_tr_no, y_tr_band, X_te_no, y_te_band, labels=labels, seed=SEED)

    m_nb_loc, nb_loc = nb_mod.train_naive_bayes(X_tr_clf_loc, y_tr_band, X_te_clf_loc, y_te_band, labels=labels)
    m_dt_loc, dt_loc = dt_mod.train_decision_tree(X_tr_clf_loc, y_tr_band, X_te_clf_loc, y_te_band, labels=labels, seed=SEED)

    classes = list(m_nb_loc.classes_)
    def eval_clf_comb(prob_test):
        pred_test = np.array(classes)[np.argmax(prob_test, axis=1)]
        return {
            "Accuracy": accuracy_score(y_te_band, pred_test),
            "Precision": precision_score(y_te_band, pred_test, average="macro", zero_division=0),
            "Recall": recall_score(y_te_band, pred_test, average="macro", zero_division=0),
            "Macro F1": f1_score(y_te_band, pred_test, average="macro", zero_division=0),
            "Log Loss": log_loss(y_te_band, prob_test, labels=classes),
            "ROC-AUC": roc_auc_score(y_te_band, prob_test, multi_class="ovr", labels=classes),
            "cm": confusion_matrix(y_te_band, pred_test, labels=labels),
            "labels": labels, "pred": pred_test, "prob": prob_test
        }

    comb_clf_no = eval_clf_comb((nb_no["prob"] + dt_no["prob"]) / 2)
    comb_clf_loc = eval_clf_comb((nb_loc["prob"] + dt_loc["prob"]) / 2)

    # layer 2 pipeline (cluster -> band -> price)
    prep_clf_pipe = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols + ["cluster"]),
        ("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile", subsample=None), num_cols)
    ])
    X_tr_clf_pipe = prep_clf_pipe.fit_transform(train_loc[cat_cols + ["cluster"] + num_cols])
    X_te_clf_pipe = prep_clf_pipe.transform(test_loc[cat_cols + ["cluster"] + num_cols])

    m_nb_pipe = nb_mod.MultinomialNB().fit(X_tr_clf_pipe, y_tr_band)
    m_dt_pipe = dt_mod.DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED).fit(X_tr_clf_pipe, y_tr_band)

    prob_te_pipe = (m_nb_pipe.predict_proba(X_te_clf_pipe) + m_dt_pipe.predict_proba(X_te_clf_pipe)) / 2
    pred_te_pipe = m_nb_pipe.classes_[np.argmax(prob_te_pipe, axis=1)]
    clf_pipe_eval = eval_clf_comb(prob_te_pipe)

    # out-of-fold bands on train using GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_bands = np.empty(len(train_loc), dtype=object)
    for tr_idx, val_idx in gkf.split(train_loc, groups=train_loc["market"]):
        fold_prep = clone(prep_clf_pipe)
        X_tr_f = fold_prep.fit_transform(train_loc.iloc[tr_idx][cat_cols + ["cluster"] + num_cols])
        X_val_f = fold_prep.transform(train_loc.iloc[val_idx][cat_cols + ["cluster"] + num_cols])
        nb_f = nb_mod.MultinomialNB().fit(X_tr_f, train_loc.iloc[tr_idx]["band"])
        dt_f = dt_mod.DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED).fit(X_tr_f, train_loc.iloc[tr_idx]["band"])
        prob_val = (nb_f.predict_proba(X_val_f) + dt_f.predict_proba(X_val_f)) / 2
        oof_bands[val_idx] = nb_f.classes_[np.argmax(prob_val, axis=1)]

    train_loc["pred_band"] = oof_bands
    test_loc["pred_band"] = pred_te_pipe

    prep_reg_pipe = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols + ["cluster", "pred_band"]),
        ("num", StandardScaler(), num_cols)
    ])
    X_tr_reg_pipe = prep_reg_pipe.fit_transform(train_loc[cat_cols + ["cluster", "pred_band"] + num_cols])
    X_te_reg_pipe = prep_reg_pipe.transform(test_loc[cat_cols + ["cluster", "pred_band"] + num_cols])

    m_lr_pipe = lr_mod.LinearRegression().fit(X_tr_reg_pipe, y_tr)
    m_rf_pipe = rf_mod.RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X_tr_reg_pipe, y_tr)

    p_comb_pipe = (m_lr_pipe.predict(X_te_reg_pipe) + m_rf_pipe.predict(X_te_reg_pipe)) / 2
    reg_pipe_eval = eval_reg_comb(p_comb_pipe, (m_lr_pipe.predict(X_tr_reg_pipe) + m_rf_pipe.predict(X_tr_reg_pipe)) / 2)

    # feature importance & coefficients
    reg_names = [x.replace("cat__", "").replace("num__", "") for x in prep_loc.get_feature_names_out()]
    coef_series = pd.Series(m_lr_loc.coef_, index=reg_names).sort_values()
    lr_pos = coef_series.tail(8)[::-1]
    lr_neg = coef_series.head(8)

    rf_imp = {"commodity": 0.0, "state": 0.0, "grade": 0.0, "location_features": 0.0}
    for name, val in zip(reg_names, m_rf_loc.feature_importances_):
        if name.startswith("commodity_"):
            rf_imp["commodity"] += val
        elif name.startswith("state_") and not name.startswith("state_commodity") and not name.startswith("state_price"):
            rf_imp["state"] += val
        elif name.startswith("grade_"):
            rf_imp["grade"] += val
        else:
            rf_imp["location_features"] += val
    rf_importance = pd.Series(rf_imp).sort_values(ascending=False)

    clf_names = [x.replace("cat__", "").replace("bin__", "") for x in prep_clf_loc.get_feature_names_out()]
    dt_imp = {"commodity": 0.0, "state": 0.0, "grade": 0.0, "location_features": 0.0}
    for name, val in zip(clf_names, m_dt_loc.feature_importances_):
        if name.startswith("commodity_"):
            dt_imp["commodity"] += val
        elif name.startswith("state_") and not name.startswith("state_commodity") and not name.startswith("state_price"):
            dt_imp["state"] += val
        elif name.startswith("grade_"):
            dt_imp["grade"] += val
        else:
            dt_imp["location_features"] += val
    dt_importance = pd.Series(dt_imp).sort_values(ascending=False)

    diff_nb = m_nb_loc.feature_log_prob_[classes.index("High")] - m_nb_loc.feature_log_prob_[classes.index("Low")]
    nb_series = pd.Series(diff_nb, index=clf_names).sort_values()
    nb_pos = nb_series.tail(8)[::-1]
    nb_neg = nb_series.head(8)

    tree_rules = export_text(m_dt_loc, feature_names=clf_names, max_depth=3)

    # clustering for prediction test (hide 10 commodities)
    np.random.seed(SEED)
    all_comms = tab.index.tolist()
    hidden = np.random.choice(all_comms, 10, replace=False)
    sub_tab = tab.drop(index=hidden)
    sub_X = StandardScaler().fit_transform(sub_tab[["median_log_price", "median_spread", "log_n_markets", "n_states"]])
    sub_km, sub_labels, _, _, _ = km_mod.train_kmeans(sub_X, k=k, seed=SEED)

    sub_cluster_med = sub_tab.groupby(sub_labels)["median_log_price"].median()
    sub_overall_med = sub_tab["median_log_price"].median()

    hidden_X = StandardScaler().fit(sub_tab[["median_log_price", "median_spread", "log_n_markets", "n_states"]]).transform(tab.loc[hidden, ["median_log_price", "median_spread", "log_n_markets", "n_states"]])
    hidden_pred = sub_km.predict(hidden_X)
    actual_hidden = np.expm1(tab.loc[hidden, "median_log_price"]).values
    pred_hidden = np.expm1([sub_cluster_med[c] for c in hidden_pred])
    base_hidden = np.expm1([sub_overall_med] * len(hidden))

    cluster_err = float(np.mean(np.abs(pred_hidden - actual_hidden) / actual_hidden) * 100)
    base_err = float(np.mean(np.abs(base_hidden - actual_hidden) / actual_hidden) * 100)

    # 5 sample test rows
    sample_5_idx = [0, 25, 50, 75, 100]
    sample_rows = test_loc.iloc[sample_5_idx].copy()
    sample_rows["Actual Modal Price (Rs)"] = sample_rows["modal_price"].round(0)
    sample_rows["Predicted Price (Rs)"] = np.expm1(p_comb_pipe[sample_5_idx]).round(0)
    sample_rows["Error (Rs)"] = (sample_rows["Predicted Price (Rs)"] - sample_rows["Actual Modal Price (Rs)"]).round(0)
    sample_rows["Error (%)"] = (np.abs(sample_rows["Error (Rs)"]) / sample_rows["Actual Modal Price (Rs)"] * 100).round(1)
    sample_table = sample_rows[["commodity", "state", "grade", "band", "pred_band", "Actual Modal Price (Rs)", "Predicted Price (Rs)", "Error (Rs)", "Error (%)"]]

    # predict new case function
    def predict_new_case(cm, st, gr):
        cl = stats["comm_to_cluster"].get(cm, stats["best_cluster"])
        agreed_status = tab.loc[cm, "cluster_agreement"] if cm in tab.index else "Fallback Cluster (Rare Crop)"

        is_seen = (st, cm) in stats["scm"]
        n_mkts = stats["nms"].get((st, cm), 0)
        if is_seen:
            loc_msg = f"Sold in {n_mkts} market(s) of {st}."
            scm_val = stats["scm"][(st, cm)]
        else:
            loc_msg = f"Not seen in {st} in this dataset (this is one day of data, so it may just be unreported). Using national/cluster fallback price."
            scm_val = stats["nat_med"].get(cm, stats["cluster_meds"].get(cl, stats["global_med"]))

        spl_val = stats["spl"].get(st, 0.0)

        df_query = pd.DataFrame({
            "state": [st], "commodity": [cm], "grade": [gr],
            "cluster": [cl], "state_commodity_median": [scm_val],
            "state_price_level": [spl_val], "n_markets_selling": [n_mkts]
        })

        X_q_clf = prep_clf_pipe.transform(df_query)
        p_nb = m_nb_pipe.predict_proba(X_q_clf)[0]
        p_dt = m_dt_pipe.predict_proba(X_q_clf)[0]
        p_comb = (p_nb + p_dt) / 2
        pred_band = m_nb_pipe.classes_[int(np.argmax(p_comb))]
        band_probs = {c: float(p_comb[i]) for i, c in enumerate(m_nb_pipe.classes_)}

        df_query_reg = df_query.copy()
        df_query_reg["pred_band"] = pred_band
        X_q_reg = prep_reg_pipe.transform(df_query_reg)

        pred_lr = float(m_lr_pipe.predict(X_q_reg)[0])
        pred_rf = float(m_rf_pipe.predict(X_q_reg)[0])
        pred_comb = (pred_lr + pred_rf) / 2

        X_q_std = prep_loc.transform(df_query[["state", "commodity", "grade", "state_commodity_median", "state_price_level", "n_markets_selling"]])
        std_lr = float(m_lr_loc.predict(X_q_std)[0])
        std_rf = float(m_rf_loc.predict(X_q_std)[0])

        combo_seen = (st, cm, gr) in set(zip(train["state"], train["commodity"], train["grade"]))

        return {
            "commodity": cm, "state": st, "grade": gr,
            "cluster": cl, "cluster_agreement": agreed_status,
            "is_seen_in_state": is_seen, "n_markets_selling": n_mkts,
            "location_message": loc_msg, "combo_seen_in_train": combo_seen,
            "predicted_band": pred_band, "band_probabilities": band_probs,
            "predicted_price_rs": float(np.expm1(pred_comb)),
            "pipeline_lr_rs": float(np.expm1(pred_lr)),
            "pipeline_rf_rs": float(np.expm1(pred_rf)),
            "standalone_lr_rs": float(np.expm1(std_lr)),
            "standalone_rf_rs": float(np.expm1(std_rf)),
        }

    return {
        "clean_df": clean_df,
        "raw_df": raw_df,
        "train_df": train,
        "test_df": test,
        "band_edges": band_edges,
        "tab": tab,
        "test_missing_tab_count": test_missing_tab_count,
        "reg_models": {
            "Linear Regression (without location)": lr_no,
            "Linear Regression (with location)": lr_loc,
            "Random Forest (without location)": rf_no,
            "Random Forest (with location)": rf_loc,
            "Combined Average (without location)": comb_reg_no,
            "Combined Average (with location)": comb_reg_loc,
            "Full Pipeline (with cluster & band)": reg_pipe_eval,
        },
        "clf_models": {
            "Naive Bayes (without location)": nb_no,
            "Naive Bayes (with location)": nb_loc,
            "Decision Tree (without location)": dt_no,
            "Decision Tree (with location)": dt_loc,
            "Combined Soft Vote (without location)": comb_clf_no,
            "Combined Soft Vote (with location)": comb_clf_loc,
            "Full Pipeline (with cluster & location)": clf_pipe_eval,
        },
        "clustering": {
            "k": k,
            "km": {"WCSS": km_wcss, "silhouette": km_sil, "profile": km_profile, "elbow_df": elbow_df},
            "hc": {"WCSS": hc_wcss, "silhouette": hc_sil, "profile": hc_profile, "z_linkage": z_link},
            "agree_share": agree_share,
        },
        "lr_pos_coefs": lr_pos,
        "lr_neg_coefs": lr_neg,
        "rf_importance": rf_importance,
        "dt_importance": dt_importance,
        "nb_pos_features": nb_pos,
        "nb_neg_features": nb_neg,
        "tree_rules_text": tree_rules,
        "hidden_cluster_err": cluster_err,
        "hidden_baseline_err": base_err,
        "sample_5_table": sample_table,
        "predict_new_case_fn": predict_new_case,
        "train_states": sorted(train["state"].unique().tolist()),
        "train_commodities": sorted(train["commodity"].unique().tolist()),
        "train_grades": sorted(train["grade"].unique().tolist()),
    }


if __name__ == "__main__":
    res = run_pipeline()
    print("pipeline finished successfully!")
