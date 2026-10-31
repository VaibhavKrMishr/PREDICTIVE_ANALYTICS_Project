# location_features.py - honest market-based splitting and leave-one-market-out location features

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

SEED = 42


# honest split by market to simulate predicting at a brand new mandi
def split_by_market(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["market"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # tertiles of modal price on train set only (low, mid, high)
    train_df["band"], band_bins = pd.qcut(train_df["modal_price"], q=3, labels=["Low", "Mid", "High"], retbins=True)
    bins = band_bins.copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    test_df["band"] = pd.cut(test_df["modal_price"], bins=bins, labels=["Low", "Mid", "High"])

    train_df["band"] = train_df["band"].astype(str)
    test_df["band"] = test_df["band"].astype(str)

    band_edges = train_df.groupby("band", observed=True)["modal_price"].agg(["min", "max", "count"])
    band_edges = band_edges.reindex(["Low", "Mid", "High"])

    return train_df, test_df, band_edges, band_bins


# commodity table computed from train rows only (commodities with >= 3 rows)
def build_commodity_table(train_df, min_rows=3):
    g = train_df.groupby("commodity").agg(
        median_log_price=("log_modal", "median"),
        median_spread=("spread", "median"),
        n_markets=("market", "nunique"),
        n_states=("state", "nunique"),
        n_rows=("modal_price", "size"),
    )
    tab = g[g["n_rows"] >= min_rows].copy()
    tab["log_n_markets"] = np.log1p(tab["n_markets"])
    return tab


# compute location stats from training rows
def compute_location_stats(train_df, tab):
    comm_cluster_map = tab["cluster"].to_dict() if "cluster" in tab.columns else {}
    if "cluster" in tab.columns:
        cluster_medians = tab.groupby("cluster")["median_log_price"].median().to_dict()
    else:
        cluster_medians = {}
    global_median = float(train_df["log_modal"].median())
    if cluster_medians:
        overall_median_cluster = min(cluster_medians.keys(), key=lambda c: abs(cluster_medians[c] - global_median))
    else:
        overall_median_cluster = 0

    nat_med = train_df.groupby("commodity")["log_modal"].median().to_dict()
    scm_map = train_df.groupby(["state", "commodity"])["log_modal"].median().to_dict()
    nms_map = train_df.groupby(["state", "commodity"])["market"].nunique().to_dict()

    diff = train_df["log_modal"] - train_df["commodity"].map(nat_med).fillna(global_median)
    spl_map = diff.groupby(train_df["state"]).median().to_dict()

    return {
        "comm_cluster_map": comm_cluster_map,
        "cluster_medians": cluster_medians,
        "global_median": global_median,
        "overall_median_cluster": overall_median_cluster,
        "nat_med": nat_med,
        "scm_map": scm_map,
        "nms_map": nms_map,
        "spl_map": spl_map,
    }


# leave-one-market-out location features for train set (GroupKFold by market)
def add_location_features_train(train_df, stats):
    gkf = GroupKFold(n_splits=5)
    scm_arr = np.zeros(len(train_df))
    spl_arr = np.zeros(len(train_df))
    nms_arr = np.zeros(len(train_df))
    seen_arr = np.zeros(len(train_df), dtype=int)

    comm_cluster_map = stats["comm_cluster_map"]
    cluster_medians = stats["cluster_medians"]
    overall_median_cluster = stats["overall_median_cluster"]

    for tr_idx, val_idx in gkf.split(train_df, groups=train_df["market"]):
        tr = train_df.iloc[tr_idx]

        fold_nat_med = tr.groupby("commodity")["log_modal"].median().to_dict()
        fold_g_med = float(tr["log_modal"].median())
        fold_scm = tr.groupby(["state", "commodity"])["log_modal"].median().to_dict()
        fold_nms = tr.groupby(["state", "commodity"])["market"].nunique().to_dict()

        fold_diff = tr["log_modal"] - tr["commodity"].map(fold_nat_med).fillna(fold_g_med)
        fold_spl = fold_diff.groupby(tr["state"]).median().to_dict()

        for idx in val_idx:
            row = train_df.iloc[idx]
            st = row["state"]
            cm = row["commodity"]
            cl = comm_cluster_map.get(cm, overall_median_cluster)

            if (st, cm) in fold_scm:
                scm_arr[idx] = fold_scm[(st, cm)]
                nms_arr[idx] = fold_nms[(st, cm)]
                seen_arr[idx] = 1
            else:
                seen_arr[idx] = 0
                nms_arr[idx] = 0
                if cm in fold_nat_med:
                    scm_arr[idx] = fold_nat_med[cm]
                elif cl in cluster_medians:
                    scm_arr[idx] = cluster_medians[cl]
                else:
                    scm_arr[idx] = fold_g_med

            spl_arr[idx] = fold_spl.get(st, 0.0)

    res = train_df.copy()
    res["state_commodity_median"] = scm_arr
    res["state_price_level"] = spl_arr
    res["n_markets_selling"] = nms_arr
    res["seen_in_state"] = seen_arr
    res["cluster"] = res["commodity"].map(comm_cluster_map).fillna(overall_median_cluster).astype(int)
    return res


# apply full-train location features to test set or a single query
def add_location_features_target(target_df, stats):
    comm_cluster_map = stats["comm_cluster_map"]
    cluster_medians = stats["cluster_medians"]
    overall_median_cluster = stats["overall_median_cluster"]
    global_median = stats["global_median"]
    nat_med = stats["nat_med"]
    scm_map = stats["scm_map"]
    nms_map = stats["nms_map"]
    spl_map = stats["spl_map"]

    n = len(target_df)
    scm_arr = np.zeros(n)
    spl_arr = np.zeros(n)
    nms_arr = np.zeros(n)
    seen_arr = np.zeros(n, dtype=int)
    cluster_arr = np.zeros(n, dtype=int)

    for i in range(n):
        row = target_df.iloc[i]
        st = row["state"]
        cm = row["commodity"]
        cl = comm_cluster_map.get(cm, overall_median_cluster)
        cluster_arr[i] = cl

        if (st, cm) in scm_map:
            scm_arr[i] = scm_map[(st, cm)]
            nms_arr[i] = nms_map[(st, cm)]
            seen_arr[i] = 1
        else:
            seen_arr[i] = 0
            nms_arr[i] = 0
            if cm in nat_med:
                scm_arr[i] = nat_med[cm]
            elif cl in cluster_medians:
                scm_arr[i] = cluster_medians[cl]
            else:
                scm_arr[i] = global_median

        spl_arr[i] = spl_map.get(st, 0.0)

    res = target_df.copy()
    res["state_commodity_median"] = scm_arr
    res["state_price_level"] = spl_arr
    res["n_markets_selling"] = nms_arr
    res["seen_in_state"] = seen_arr
    res["cluster"] = cluster_arr
    return res
