import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

SEED = 42


def split_by_market(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["market"]))
    train = df.iloc[tr_idx].copy().reset_index(drop=True)
    test = df.iloc[te_idx].copy().reset_index(drop=True)

    train["band"], bins = pd.qcut(train["modal_price"], q=3, labels=["Low", "Mid", "High"], retbins=True)
    bins[0], bins[-1] = -np.inf, np.inf
    test["band"] = pd.cut(test["modal_price"], bins=bins, labels=["Low", "Mid", "High"]).astype(str)
    train["band"] = train["band"].astype(str)

    band_edges = train.groupby("band", observed=True)["modal_price"].agg(["min", "max", "count"]).reindex(["Low", "Mid", "High"])
    return train, test, band_edges, bins


def build_commodity_table(train, min_rows=3):
    tab = train.groupby("commodity").agg(
        median_log_price=("log_modal", "median"),
        median_spread=("spread", "median"),
        n_markets=("market", "nunique"),
        n_states=("state", "nunique"),
        n_rows=("modal_price", "size"),
    )
    tab = tab[tab["n_rows"] >= min_rows].copy()
    tab["log_n_markets"] = np.log1p(tab["n_markets"])
    return tab


def compute_location_stats(train, tab):
    comm_to_cluster = tab["cluster"].to_dict() if "cluster" in tab.columns else {}
    cluster_meds = tab.groupby("cluster")["median_log_price"].median().to_dict() if "cluster" in tab.columns else {}
    global_med = float(train["log_modal"].median())
    best_cluster = min(cluster_meds.keys(), key=lambda c: abs(cluster_meds[c] - global_med)) if cluster_meds else 0

    nat_med = train.groupby("commodity")["log_modal"].median().to_dict()
    scm = train.groupby(["state", "commodity"])["log_modal"].median().to_dict()
    nms = train.groupby(["state", "commodity"])["market"].nunique().to_dict()

    diff = train["log_modal"] - train["commodity"].map(nat_med).fillna(global_med)
    spl = diff.groupby(train["state"]).median().to_dict()

    return {
        "comm_to_cluster": comm_to_cluster,
        "cluster_meds": cluster_meds,
        "global_med": global_med,
        "best_cluster": best_cluster,
        "nat_med": nat_med,
        "scm": scm,
        "nms": nms,
        "spl": spl,
    }


def add_location_features_train(train, stats):
    gkf = GroupKFold(n_splits=5)
    n = len(train)
    scm_arr, spl_arr, nms_arr = np.zeros(n), np.zeros(n), np.zeros(n)
    seen_arr = np.zeros(n, dtype=int)

    comm_to_cluster = stats["comm_to_cluster"]
    cluster_meds = stats["cluster_meds"]
    best_cluster = stats["best_cluster"]

    for tr_idx, val_idx in gkf.split(train, groups=train["market"]):
        tr = train.iloc[tr_idx]

        f_nat = tr.groupby("commodity")["log_modal"].median().to_dict()
        f_med = float(tr["log_modal"].median())
        f_scm = tr.groupby(["state", "commodity"])["log_modal"].median().to_dict()
        f_nms = tr.groupby(["state", "commodity"])["market"].nunique().to_dict()

        diff = tr["log_modal"] - tr["commodity"].map(f_nat).fillna(f_med)
        f_spl = diff.groupby(tr["state"]).median().to_dict()

        for idx in val_idx:
            row = train.iloc[idx]
            st, cm = row["state"], row["commodity"]
            cl = comm_to_cluster.get(cm, best_cluster)

            if (st, cm) in f_scm:
                scm_arr[idx] = f_scm[(st, cm)]
                nms_arr[idx] = f_nms[(st, cm)]
                seen_arr[idx] = 1
            else:
                seen_arr[idx] = 0
                nms_arr[idx] = 0
                scm_arr[idx] = f_nat.get(cm, cluster_meds.get(cl, f_med))

            spl_arr[idx] = f_spl.get(st, 0.0)

    res = train.copy()
    res["state_commodity_median"] = scm_arr
    res["state_price_level"] = spl_arr
    res["n_markets_selling"] = nms_arr
    res["seen_in_state"] = seen_arr
    res["cluster"] = res["commodity"].map(comm_to_cluster).fillna(best_cluster).astype(int)
    return res


def add_location_features_target(df, stats):
    n = len(df)
    scm_arr, spl_arr, nms_arr = np.zeros(n), np.zeros(n), np.zeros(n)
    seen_arr, cluster_arr = np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    comm_to_cluster = stats["comm_to_cluster"]
    cluster_meds = stats["cluster_meds"]
    best_cluster = stats["best_cluster"]
    global_med = stats["global_med"]
    nat_med = stats["nat_med"]
    scm = stats["scm"]
    nms = stats["nms"]
    spl = stats["spl"]

    for i in range(n):
        row = df.iloc[i]
        st, cm = row["state"], row["commodity"]
        cl = comm_to_cluster.get(cm, best_cluster)
        cluster_arr[i] = cl

        if (st, cm) in scm:
            scm_arr[i] = scm[(st, cm)]
            nms_arr[i] = nms[(st, cm)]
            seen_arr[i] = 1
        else:
            seen_arr[i] = 0
            nms_arr[i] = 0
            scm_arr[i] = nat_med.get(cm, cluster_meds.get(cl, global_med))

        spl_arr[i] = spl.get(st, 0.0)

    res = df.copy()
    res["state_commodity_median"] = scm_arr
    res["state_price_level"] = spl_arr
    res["n_markets_selling"] = nms_arr
    res["seen_in_state"] = seen_arr
    res["cluster"] = cluster_arr
    return res
