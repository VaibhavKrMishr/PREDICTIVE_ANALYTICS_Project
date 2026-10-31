# main.py - central pipeline orchestrator bringing together data cleaning, location features, and all models

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
    # 1. load raw and clean data
    raw_df = clean.load_raw(data_path)
    clean_df = clean.clean_data(raw_df)

    # 2. honest market-based train/test split (unseen mandis)
    train_df, test_df, band_edges, band_bins = loc.split_by_market(clean_df)

    # 3. unsupervised clustering on train commodity table
    tab = loc.build_commodity_table(train_df, min_rows=3)
    clu_features = ["median_log_price", "median_spread", "log_n_markets", "n_states"]
    scaler_clu = StandardScaler()
    x_clu_scaled = scaler_clu.fit_transform(tab[clu_features])

    k = 5
    m_km, km_labels, km_wcss, km_sil, elbow_df = km_mod.train_kmeans(x_clu_scaled, k=k, seed=SEED)
    m_hc, hc_labels, hc_wcss, hc_sil, z_linkage = hc_mod.train_hierarchical(x_clu_scaled, k=k)

    # bipartite maximum overlap matching between hierarchical and k-means clusters
    ct = pd.crosstab(km_labels, hc_labels)
    row_ind, col_ind = linear_sum_assignment(-ct.values)
    hc_to_km = {hc_c: km_c for km_c, hc_c in zip(row_ind, col_ind)}
    mapped_hc_labels = np.array([hc_to_km[c] for c in hc_labels])

    agreed = (km_labels == mapped_hc_labels)
    agree_share = float(np.mean(agreed))

    tab = tab.copy()
    tab["cluster"] = km_labels
    tab["KMeans cluster"] = km_labels
    tab["Hier cluster"] = mapped_hc_labels
    tab["cluster_agreement"] = np.where(agreed, "Agreed", "Uncertain")

    # cluster profiles
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

    # 4. location features (leave-one-market-out for train, full stats for test)
    stats = loc.compute_location_stats(train_df, tab)
    test_missing_tab_count = sum(c not in tab.index for c in test_df["commodity"].unique())

    train_loc = loc.add_location_features_train(train_df, stats)
    test_loc = loc.add_location_features_target(test_df, stats)

    # 5. prepare regression data and train models
    cat_base = ["state", "commodity", "grade"]
    num_loc = ["state_commodity_median", "state_price_level", "n_markets_selling"]
    y_train = train_loc["log_modal"]
    y_test = test_loc["log_modal"]

    # without location
    prep_reg_no = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_base)])
    x_tr_reg_no = prep_reg_no.fit_transform(train_loc[cat_base])
    x_te_reg_no = prep_reg_no.transform(test_loc[cat_base])

    m_lr_no, lr_no = lr_mod.train_linear_regression(x_tr_reg_no, y_train, x_te_reg_no, y_test)
    m_rf_no, rf_no = rf_mod.train_random_forest(x_tr_reg_no, y_train, x_te_reg_no, y_test, seed=SEED)

    # with location
    prep_reg_loc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_base),
        ("num", StandardScaler(), num_loc)
    ])
    x_tr_reg_loc = prep_reg_loc.fit_transform(train_loc[cat_base + num_loc])
    x_te_reg_loc = prep_reg_loc.transform(test_loc[cat_base + num_loc])

    m_lr_loc, lr_loc = lr_mod.train_linear_regression(x_tr_reg_loc, y_train, x_te_reg_loc, y_test)
    m_rf_loc, rf_loc = rf_mod.train_random_forest(x_tr_reg_loc, y_train, x_te_reg_loc, y_test, seed=SEED)

    # combined average regression (layer 1)
    def eval_reg_comb(pred_test, pred_train=None):
        mse = mean_squared_error(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = float(np.sqrt(mse))
        r2_te = r2_score(y_test, pred_test)
        r2_tr = r2_score(y_train, pred_train) if pred_train is not None else np.nan
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 (test)": r2_te, "R2 (train)": r2_tr, "pred": pred_test, "y_test": np.array(y_test)}

    comb_reg_no = eval_reg_comb((lr_no["pred"] + rf_no["pred"]) / 2)
    comb_reg_loc = eval_reg_comb((lr_loc["pred"] + rf_loc["pred"]) / 2)

    # 6. prepare classification data and train models
    labels = ["Low", "Mid", "High"]
    y_tr_band = train_loc["band"]
    y_te_band = test_loc["band"]

    # without location
    prep_clf_no = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_base)])
    x_tr_clf_no = prep_clf_no.fit_transform(train_loc[cat_base])
    x_te_clf_no = prep_clf_no.transform(test_loc[cat_base])

    m_nb_no, nb_no = nb_mod.train_naive_bayes(x_tr_clf_no, y_tr_band, x_te_clf_no, y_te_band, labels=labels)
    m_dt_no, dt_no = dt_mod.train_decision_tree(x_tr_clf_no, y_tr_band, x_te_clf_no, y_te_band, labels=labels, seed=SEED)

    # with location
    prep_clf_loc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_base),
        ("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile", subsample=None), num_loc)
    ])
    x_tr_clf_loc = prep_clf_loc.fit_transform(train_loc[cat_base + num_loc])
    x_te_clf_loc = prep_clf_loc.transform(test_loc[cat_base + num_loc])

    m_nb_loc, nb_loc = nb_mod.train_naive_bayes(x_tr_clf_loc, y_tr_band, x_te_clf_loc, y_te_band, labels=labels)
    m_dt_loc, dt_loc = dt_mod.train_decision_tree(x_tr_clf_loc, y_tr_band, x_te_clf_loc, y_te_band, labels=labels, seed=SEED)

    # combined soft voting classification (layer 1)
    classes = list(m_nb_loc.classes_)
    def eval_clf_comb(prob_test):
        pred_test = np.array(classes)[np.argmax(prob_test, axis=1)]
        acc = accuracy_score(y_te_band, pred_test)
        prec = precision_score(y_te_band, pred_test, average="macro", zero_division=0)
        rec = recall_score(y_te_band, pred_test, average="macro", zero_division=0)
        f1 = f1_score(y_te_band, pred_test, average="macro", zero_division=0)
        ll = log_loss(y_te_band, prob_test, labels=classes)
        auc = roc_auc_score(y_te_band, prob_test, multi_class="ovr", labels=classes)
        cm = confusion_matrix(y_te_band, pred_test, labels=labels)
        return {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "Macro F1": f1,
            "Log Loss": ll, "ROC-AUC": auc, "cm": cm, "labels": labels, "pred": pred_test, "prob": prob_test
        }

    comb_clf_no = eval_clf_comb((nb_no["prob"] + dt_no["prob"]) / 2)
    comb_clf_loc = eval_clf_comb((nb_loc["prob"] + dt_loc["prob"]) / 2)

    # 7. layer 2 pipeline (cluster -> classifier band -> regressor price)
    prep_clf_pipe = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_base + ["cluster"]),
        ("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile", subsample=None), num_loc)
    ])
    x_tr_clf_pipe = prep_clf_pipe.fit_transform(train_loc[cat_base + ["cluster"] + num_loc])
    x_te_clf_pipe = prep_clf_pipe.transform(test_loc[cat_base + ["cluster"] + num_loc])

    m_nb_pipe = nb_mod.MultinomialNB().fit(x_tr_clf_pipe, y_tr_band)
    m_dt_pipe = dt_mod.DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED).fit(x_tr_clf_pipe, y_tr_band)

    prob_te_pipe = (m_nb_pipe.predict_proba(x_te_clf_pipe) + m_dt_pipe.predict_proba(x_te_clf_pipe)) / 2
    pred_te_pipe = m_nb_pipe.classes_[np.argmax(prob_te_pipe, axis=1)]
    clf_pipe_eval = eval_clf_comb(prob_te_pipe)

    # out-of-fold bands on train using GroupKFold for regressor input
    gkf = GroupKFold(n_splits=5)
    oof_bands = np.empty(len(train_loc), dtype=object)
    for tr_idx, val_idx in gkf.split(train_loc, groups=train_loc["market"]):
        fold_prep = clone(prep_clf_pipe)
        x_tr_f = fold_prep.fit_transform(train_loc.iloc[tr_idx][cat_base + ["cluster"] + num_loc])
        x_val_f = fold_prep.transform(train_loc.iloc[val_idx][cat_base + ["cluster"] + num_loc])
        nb_f = nb_mod.MultinomialNB().fit(x_tr_f, train_loc.iloc[tr_idx]["band"])
        dt_f = dt_mod.DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED).fit(x_tr_f, train_loc.iloc[tr_idx]["band"])
        prob_val = (nb_f.predict_proba(x_val_f) + dt_f.predict_proba(x_val_f)) / 2
        oof_bands[val_idx] = nb_f.classes_[np.argmax(prob_val, axis=1)]

    train_loc["pred_band"] = oof_bands
    test_loc["pred_band"] = pred_te_pipe

    prep_reg_pipe = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_base + ["cluster", "pred_band"]),
        ("num", StandardScaler(), num_loc)
    ])
    x_tr_reg_pipe = prep_reg_pipe.fit_transform(train_loc[cat_base + ["cluster", "pred_band"] + num_loc])
    x_te_reg_pipe = prep_reg_pipe.transform(test_loc[cat_base + ["cluster", "pred_band"] + num_loc])

    m_lr_pipe = lr_mod.LinearRegression().fit(x_tr_reg_pipe, y_train)
    m_rf_pipe = rf_mod.RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1).fit(x_tr_reg_pipe, y_train)

    p_lr_pipe = m_lr_pipe.predict(x_te_reg_pipe)
    p_rf_pipe = m_rf_pipe.predict(x_te_reg_pipe)
    p_comb_pipe = (p_lr_pipe + p_rf_pipe) / 2
    reg_pipe_eval = eval_reg_comb(p_comb_pipe, (m_lr_pipe.predict(x_tr_reg_pipe) + m_rf_pipe.predict(x_tr_reg_pipe)) / 2)

    # 8. feature coefficients & importances
    reg_feat_names = [x.replace("cat__", "").replace("num__", "") for x in prep_reg_loc.get_feature_names_out()]
    coef_series = pd.Series(m_lr_loc.coef_, index=reg_feat_names).sort_values()
    lr_pos_coefs = coef_series.tail(8)[::-1]
    lr_neg_coefs = coef_series.head(8)

    # rf importances aggregated
    rf_grouped = {"commodity": 0.0, "state": 0.0, "grade": 0.0, "location_features": 0.0}
    for name, val in zip(reg_feat_names, m_rf_loc.feature_importances_):
        if name.startswith("commodity_"):
            rf_grouped["commodity"] += val
        elif name.startswith("state_") and not name.startswith("state_commodity") and not name.startswith("state_price"):
            rf_grouped["state"] += val
        elif name.startswith("grade_"):
            rf_grouped["grade"] += val
        else:
            rf_grouped["location_features"] += val
    rf_importance = pd.Series(rf_grouped).sort_values(ascending=False)

    # dt importances aggregated
    clf_feat_names = [x.replace("cat__", "").replace("bin__", "") for x in prep_clf_loc.get_feature_names_out()]
    dt_grouped = {"commodity": 0.0, "state": 0.0, "grade": 0.0, "location_features": 0.0}
    for name, val in zip(clf_feat_names, m_dt_loc.feature_importances_):
        if name.startswith("commodity_"):
            dt_grouped["commodity"] += val
        elif name.startswith("state_") and not name.startswith("state_commodity") and not name.startswith("state_price"):
            dt_grouped["state"] += val
        elif name.startswith("grade_"):
            dt_grouped["grade"] += val
        else:
            dt_grouped["location_features"] += val
    dt_importance = pd.Series(dt_grouped).sort_values(ascending=False)

    # nb top features
    nb_classes = list(m_nb_loc.classes_)
    diff = m_nb_loc.feature_log_prob_[nb_classes.index("High")] - m_nb_loc.feature_log_prob_[nb_classes.index("Low")]
    nb_series = pd.Series(diff, index=clf_feat_names).sort_values()
    nb_pos_features = nb_series.tail(8)[::-1]
    nb_neg_features = nb_series.head(8)

    tree_rules_text = export_text(m_dt_loc, feature_names=clf_feat_names, max_depth=3)

    # 9. hidden commodity prediction benchmark (clustering for prediction test)
    np.random.seed(SEED)
    all_tab_comms = tab.index.tolist()
    hidden_comms = np.random.choice(all_tab_comms, 10, replace=False)
    sub_tab = tab.drop(index=hidden_comms)
    sub_scaler = StandardScaler()
    sub_x = sub_scaler.fit_transform(sub_tab[clu_features])
    sub_km, sub_labels, _, _, _ = km_mod.train_kmeans(sub_x, k=k, seed=SEED)

    sub_cluster_med = sub_tab.groupby(sub_labels)["median_log_price"].median()
    sub_overall_med = sub_tab["median_log_price"].median()

    hidden_x = sub_scaler.transform(tab.loc[hidden_comms, clu_features])
    hidden_pred_clusters = sub_km.predict(hidden_x)
    actual_hidden_prices = np.expm1(tab.loc[hidden_comms, "median_log_price"]).values
    pred_hidden_prices = np.expm1([sub_cluster_med[c] for c in hidden_pred_clusters])
    baseline_hidden_prices = np.expm1([sub_overall_med] * len(hidden_comms))

    cluster_test_err = float(np.mean(np.abs(pred_hidden_prices - actual_hidden_prices) / actual_hidden_prices) * 100)
    baseline_test_err = float(np.mean(np.abs(baseline_hidden_prices - actual_hidden_prices) / actual_hidden_prices) * 100)

    # 10. 5 real test rows with predictions
    sample_5_idx = [0, 25, 50, 75, 100]
    sample_rows = test_loc.iloc[sample_5_idx].copy()
    sample_rows["Actual Modal Price (Rs)"] = sample_rows["modal_price"].round(0)
    sample_rows["Predicted Price (Rs)"] = np.expm1(p_comb_pipe[sample_5_idx]).round(0)
    sample_rows["Error (Rs)"] = (sample_rows["Predicted Price (Rs)"] - sample_rows["Actual Modal Price (Rs)"]).round(0)
    sample_rows["Error (%)"] = (np.abs(sample_rows["Error (Rs)"]) / sample_rows["Actual Modal Price (Rs)"] * 100).round(1)
    sample_table = sample_rows[["commodity", "state", "grade", "band", "pred_band", "Actual Modal Price (Rs)", "Predicted Price (Rs)", "Error (Rs)", "Error (%)"]]

    # 11. predict new case function
    def predict_new_case(comm, st, gr):
        comm_cluster = stats["comm_cluster_map"].get(comm, stats["overall_median_cluster"])
        agreed_status = tab.loc[comm, "cluster_agreement"] if comm in tab.index else "Fallback Cluster (Rare Crop)"

        is_seen = (st, comm) in stats["scm_map"]
        n_mkts = stats["nms_map"].get((st, comm), 0)
        if is_seen:
            loc_msg = f"Sold in {n_mkts} market(s) of {st}."
            scm_val = stats["scm_map"][(st, comm)]
        else:
            loc_msg = f"Not seen in {st} in this dataset (this is one day of data, so it may just be unreported). Using national/cluster fallback price."
            scm_val = stats["nat_med"].get(comm, stats["cluster_medians"].get(comm_cluster, stats["global_median"]))

        spl_val = stats["spl_map"].get(st, 0.0)

        row_dict = {
            "state": [st], "commodity": [comm], "grade": [gr],
            "cluster": [comm_cluster], "state_commodity_median": [scm_val],
            "state_price_level": [spl_val], "n_markets_selling": [n_mkts]
        }
        df_query = pd.DataFrame(row_dict)
        x_q_clf = prep_clf_pipe.transform(df_query)
        p_nb = m_nb_pipe.predict_proba(x_q_clf)[0]
        p_dt = m_dt_pipe.predict_proba(x_q_clf)[0]
        p_comb = (p_nb + p_dt) / 2
        pred_band_idx = int(np.argmax(p_comb))
        pred_band = m_nb_pipe.classes_[pred_band_idx]
        band_probs = {cls: float(p_comb[i]) for i, cls in enumerate(m_nb_pipe.classes_)}

        df_query_reg = df_query.copy()
        df_query_reg["pred_band"] = pred_band
        x_q_reg = prep_reg_pipe.transform(df_query_reg)

        pred_lr_log = float(m_lr_pipe.predict(x_q_reg)[0])
        pred_rf_log = float(m_rf_pipe.predict(x_q_reg)[0])
        pred_comb_log = (pred_lr_log + pred_rf_log) / 2

        # standalone comparison
        x_q_standalone = prep_reg_loc.transform(df_query[["state", "commodity", "grade", "state_commodity_median", "state_price_level", "n_markets_selling"]])
        pred_standalone_lr = float(m_lr_loc.predict(x_q_standalone)[0])
        pred_standalone_rf = float(m_rf_loc.predict(x_q_standalone)[0])

        train_triplets = set(zip(train_df["state"], train_df["commodity"], train_df["grade"]))
        combo_seen = (st, comm, gr) in train_triplets

        return {
            "commodity": comm, "state": st, "grade": gr,
            "cluster": comm_cluster, "cluster_agreement": agreed_status,
            "is_seen_in_state": is_seen, "n_markets_selling": n_mkts,
            "location_message": loc_msg, "combo_seen_in_train": combo_seen,
            "predicted_band": pred_band, "band_probabilities": band_probs,
            "predicted_price_rs": float(np.expm1(pred_comb_log)),
            "pipeline_lr_rs": float(np.expm1(pred_lr_log)),
            "pipeline_rf_rs": float(np.expm1(pred_rf_log)),
            "standalone_lr_rs": float(np.expm1(pred_standalone_lr)),
            "standalone_rf_rs": float(np.expm1(pred_standalone_rf)),
        }

    # bundle comparison tables
    reg_models_dict = {
        "Linear Regression (without location)": lr_no,
        "Linear Regression (with location)": lr_loc,
        "Random Forest (without location)": rf_no,
        "Random Forest (with location)": rf_loc,
        "Combined Average (without location)": comb_reg_no,
        "Combined Average (with location)": comb_reg_loc,
        "Full Pipeline (with cluster & band)": reg_pipe_eval,
    }

    clf_models_dict = {
        "Naive Bayes (without location)": nb_no,
        "Naive Bayes (with location)": nb_loc,
        "Decision Tree (without location)": dt_no,
        "Decision Tree (with location)": dt_loc,
        "Combined Soft Vote (without location)": comb_clf_no,
        "Combined Soft Vote (with location)": comb_clf_loc,
        "Full Pipeline (with cluster & location)": clf_pipe_eval,
    }

    clustering_dict = {
        "k": k,
        "km": {"WCSS": km_wcss, "silhouette": km_sil, "profile": km_profile, "elbow_df": elbow_df},
        "hc": {"WCSS": hc_wcss, "silhouette": hc_sil, "profile": hc_profile, "z_linkage": z_linkage},
        "agree_share": agree_share,
    }

    return {
        "clean_df": clean_df,
        "raw_df": raw_df,
        "train_df": train_df,
        "test_df": test_df,
        "band_edges": band_edges,
        "tab": tab,
        "test_missing_tab_count": test_missing_tab_count,
        "reg_models": reg_models_dict,
        "clf_models": clf_models_dict,
        "clustering": clustering_dict,
        "lr_pos_coefs": lr_pos_coefs,
        "lr_neg_coefs": lr_neg_coefs,
        "rf_importance": rf_importance,
        "dt_importance": dt_importance,
        "nb_pos_features": nb_pos_features,
        "nb_neg_features": nb_neg_features,
        "tree_rules_text": tree_rules_text,
        "hidden_cluster_err": cluster_test_err,
        "hidden_baseline_err": baseline_test_err,
        "sample_5_table": sample_table,
        "predict_new_case_fn": predict_new_case,
        "train_states": sorted(train_df["state"].unique().tolist()),
        "train_commodities": sorted(train_df["commodity"].unique().tolist()),
        "train_grades": sorted(train_df["grade"].unique().tolist()),
    }


if __name__ == "__main__":
    print("Running main.py pipeline...")
    res = run_pipeline()
    print("Pipeline run completed successfully!\n")
    print("--- Regression Comparison (ranked by RMSE) ---")
    reg_df = pd.DataFrame([{"Model": k, "RMSE": v["RMSE"], "R2 (test)": v["R2 (test)"]} for k, v in res["reg_models"].items()])
    print(reg_df.sort_values("RMSE").to_string(index=False))

    print("\n--- Classification Comparison (ranked by Macro F1) ---")
    clf_df = pd.DataFrame([{"Model": k, "Macro F1": v["Macro F1"], "Accuracy": v["Accuracy"]} for k, v in res["clf_models"].items()])
    print(clf_df.sort_values("Macro F1", ascending=False).to_string(index=False))

    print(f"\n--- Unsupervised Clustering (k={res['clustering']['k']}) ---")
    print(f"K-Means WCSS: {res['clustering']['km']['WCSS']:.1f} | Silhouette: {res['clustering']['km']['silhouette']:.3f}")
    print(f"Hierarchical WCSS: {res['clustering']['hc']['WCSS']:.1f} | Silhouette: {res['clustering']['hc']['silhouette']:.3f}")
    print(f"Cluster Agreement Rate: {res['clustering']['agree_share']*100:.1f}%")
