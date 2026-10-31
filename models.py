import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, roc_auc_score, confusion_matrix,
                             silhouette_score)

SEED = 42
FEATURES = ["state", "commodity", "grade"]


def make_preprocessor():
    ohe = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("cat", ohe, FEATURES)])


# regression
def run_regression(df):
    X = df[FEATURES]
    y = df["log_modal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                               random_state=SEED, n_jobs=-1),
    }

    out = {}
    for name, m in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", m)])
        pipe.fit(X_train, y_train)
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        mse = mean_squared_error(y_test, pred_test)
        out[name] = {
            "pipe": pipe,
            "MAE": mean_absolute_error(y_test, pred_test),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2 test": r2_score(y_test, pred_test),
            "R2 train": r2_score(y_train, pred_train),
            "y_test": y_test.values,
            "pred": pred_test,
        }
    return out


def lr_top_coefficients(reg_result, n=10):
    pipe = reg_result["Linear Regression"]["pipe"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    s = pd.Series(coefs, index=[x.replace("cat__", "") for x in names]).sort_values()
    return s.tail(n)[::-1], s.head(n)


def group_importance(names, values):
    names = [x.replace("cat__", "") for x in names]
    imp = {f: 0.0 for f in FEATURES}
    for n, v in zip(names, values):
        for f in FEATURES:
            if n.startswith(f + "_"):
                imp[f] += v
    return pd.Series(imp).sort_values(ascending=False)


def rf_importance(reg_result):
    pipe = reg_result["Random Forest"]["pipe"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    return group_importance(names, pipe.named_steps["model"].feature_importances_)


# classification
def run_classification(df):
    d = df.copy()
    d["band"] = pd.qcut(d["modal_price"], 3, labels=["Low", "Mid", "High"])

    X = d[FEATURES]
    y = d["band"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED),
    }
    labels = ["Low", "Mid", "High"]

    out = {}
    for name, m in models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        classes = list(pipe.named_steps["model"].classes_)

        out[name] = {
            "pipe": pipe,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="macro"),
            "Recall": recall_score(y_test, pred, average="macro"),
            "F1 (macro)": f1_score(y_test, pred, average="macro"),
            "Log Loss": log_loss(y_test, proba, labels=classes),
            "AUC": roc_auc_score(y_test, proba, multi_class="ovr", labels=classes),
            "cm": confusion_matrix(y_test, pred, labels=labels),
            "labels": labels,
            "Accuracy train": accuracy_score(y_train, pipe.predict(X_train)),
        }
    out["band_edges"] = d.groupby("band", observed=True)["modal_price"].agg(["min", "max"])
    return out


def nb_top_features(clf_result, n=8):
    pipe = clf_result["Naive Bayes"]["pipe"]
    nb = pipe.named_steps["model"]
    names = [x.replace("cat__", "") for x in pipe.named_steps["prep"].get_feature_names_out()]
    classes = list(nb.classes_)
    diff = nb.feature_log_prob_[classes.index("High")] - nb.feature_log_prob_[classes.index("Low")]
    s = pd.Series(diff, index=names).sort_values()
    return s.tail(n)[::-1], s.head(n)


def tree_importance(clf_result):
    pipe = clf_result["Decision Tree"]["pipe"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    return group_importance(names, pipe.named_steps["model"].feature_importances_)


def tree_rules(clf_result, depth=3):
    pipe = clf_result["Decision Tree"]["pipe"]
    names = [x.replace("cat__", "") for x in pipe.named_steps["prep"].get_feature_names_out()]
    return export_text(pipe.named_steps["model"], feature_names=names, max_depth=depth)


# clustering
def commodity_table(df, min_rows=3):
    g = df.groupby("commodity").agg(
        median_log_price=("log_modal", "median"),
        median_spread=("spread", "median"),
        n_markets=("market", "nunique"),
        n_states=("state", "nunique"),
        n_rows=("modal_price", "size"),
    )
    g = g[g["n_rows"] >= min_rows].copy()
    g["log_n_markets"] = np.log1p(g["n_markets"])
    return g


CLUSTER_COLS = ["median_log_price", "median_spread", "log_n_markets", "n_states"]


def compute_wcss(X_mat, labels):
    total = 0.0
    for c in np.unique(labels):
        pts = X_mat[labels == c]
        centroid = pts.mean(axis=0)
        total += np.sum((pts - centroid) ** 2)
    return float(total)


def run_clustering(df, k_range=range(2, 9)):
    tab = commodity_table(df)
    X = StandardScaler().fit_transform(tab[CLUSTER_COLS])

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        rows.append({
            "k": k,
            "KMeans WCSS": km.inertia_,
            "KMeans silhouette": silhouette_score(X, km.labels_),
            "Hier silhouette": silhouette_score(X, hc.labels_),
        })
    scores = pd.DataFrame(rows)

    best_k = int(scores.loc[scores["KMeans silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)

    link_rows = []
    for link in ["single", "complete", "average", "ward"]:
        lab = AgglomerativeClustering(n_clusters=best_k, linkage=link).fit_predict(X)
        link_rows.append({"linkage": link, "silhouette": silhouette_score(X, lab)})
    linkages = pd.DataFrame(link_rows)

    tab["KMeans cluster"] = km.labels_
    tab["Hier cluster"] = hc.labels_

    return {
        "table": tab,
        "X": X,
        "k": best_k,
        "scores": scores,
        "linkages": linkages,
        "K-Means": {"silhouette": silhouette_score(X, km.labels_),
                    "WCSS": float(km.inertia_),
                    "sizes": pd.Series(km.labels_).value_counts().sort_index()},
        "Hierarchical": {"silhouette": silhouette_score(X, hc.labels_),
                         "WCSS": compute_wcss(X, hc.labels_),
                         "sizes": pd.Series(hc.labels_).value_counts().sort_index()},
    }


def cluster_profile(tab, col):
    p = tab.groupby(col)[["median_log_price", "median_spread", "n_markets", "n_states"]].mean()
    p["median_price_rs"] = np.expm1(p["median_log_price"])
    p["n_commodities"] = tab.groupby(col).size()
    return p.round(2)


def cluster_link(df, tab, reg_pipe, cluster_col="KMeans cluster"):
    d = df.copy()
    d["band_num"] = pd.qcut(d["modal_price"], 3, labels=[1, 2, 3]).astype(int)
    pred_log = reg_pipe.predict(d[FEATURES])
    d["reg_abs_err"] = (d["log_modal"] - pred_log).abs()
    d["cluster"] = d["commodity"].map(tab[cluster_col])
    valid = d.dropna(subset=["cluster"])
    g = valid.groupby("cluster").agg(
        avg_band=("band_num", "mean"),
        avg_reg_mae=("reg_abs_err", "mean"),
        median_price=("modal_price", "median"),
        n_commodities=("commodity", "nunique"),
        n_records=("modal_price", "count"),
    )
    return g.round(2)
