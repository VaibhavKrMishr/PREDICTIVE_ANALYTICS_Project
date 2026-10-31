import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_raw(path="data/raw.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # standardize column names
    df.columns = ["state", "district", "market", "commodity", "variety",
                  "grade", "arrival_date", "min_price", "max_price", "modal_price"]

    # arrival_date is the same for every row in this daily snapshot
    df = df.drop(columns=["arrival_date"])

    # clean text
    text_cols = ["state", "district", "market", "commodity", "variety", "grade"]
    for c in text_cols:
        df[c] = df[c].str.strip()
    df["commodity"] = df["commodity"].str.title()

    # fix state spelling
    df["state"] = df["state"].replace({"Keralam": "Kerala", "Chattisgarh": "Chhattisgarh"})

    # drop duplicates
    df = df.drop_duplicates()

    # price validation: min <= modal <= max
    df = df[(df["min_price"] <= df["modal_price"]) & (df["modal_price"] <= df["max_price"])]

    # drop per-piece/non-quintal prices under 10
    df = df[(df["min_price"] >= 10) & (df["max_price"] >= 10) & (df["modal_price"] >= 10)]

    # feature engineering
    df["log_modal"] = np.log1p(df["modal_price"])
    df["spread"] = (df["max_price"] - df["min_price"]) / df["modal_price"]

    df = df.reset_index(drop=True)
    return df


# eda functions
def eda_info(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })


def eda_describe(df, include="numeric"):
    if include == "all":
        return df.describe(include="all").round(2)
    elif include in ["categorical", "cat", "str"]:
        return df.select_dtypes(include=["object", "string", "str"]).describe()
    return df.describe().round(2)


def eda_summary(df):
    return eda_describe(df)



def eda_correlations(df):
    return df[["min_price", "max_price", "modal_price"]].corr().round(3)


def eda_top_states(df, n=10):
    return df["state"].value_counts().head(n)


def eda_top_commodities(df, n=10):
    return df.groupby("commodity")["modal_price"].median().sort_values().tail(n)


def plot_price_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].hist(df["modal_price"], bins=45, color="#059669", edgecolor="white", alpha=0.85)
    ax[0].set_title("Modal Price (Skewed)", fontsize=9, fontweight="bold", color="#1e293b")
    ax[0].tick_params(labelsize=8)
    ax[0].grid(axis="y", linestyle="--", alpha=0.3)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    ax[1].hist(df["log_modal"], bins=45, color="#0284c7", edgecolor="white", alpha=0.85)
    ax[1].set_title("Log-Transformed Price", fontsize=9, fontweight="bold", color="#1e293b")
    ax[1].tick_params(labelsize=8)
    ax[1].grid(axis="y", linestyle="--", alpha=0.3)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_correlations(df):
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    corr = eda_correlations(df)
    sns.heatmap(corr, annot=True, fmt=".3f", ax=ax, cmap="YlGnBu",
                annot_kws={"size": 8.5, "weight": "bold"}, cbar=False, linewidths=1.2, linecolor="white")
    ax.set_title("Price Correlations", fontsize=9, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_top_states(df, n=10):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    data = eda_top_states(df, n)
    data.plot.barh(ax=ax, color="#0d9488", edgecolor="white", alpha=0.9)
    ax.invert_yaxis()
    ax.set_title(f"Market Records per State (Top {n})", fontsize=9, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_top_commodities(df, n=10):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    data = eda_top_commodities(df, n)
    data.plot.barh(ax=ax, color="#d97706", edgecolor="white", alpha=0.9)
    ax.set_title("Highest Median Modal Price (Rs)", fontsize=9, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig



def run_eda(df):
    print("=== Dataset Dimensions ===")
    print("Shape:", df.shape)
    print("\n=== Dataset Info (df.info()) ===")
    df.info()
    print("\n=== First 5 Rows (df.head()) ===")
    print(df.head(5))
    print("\n=== Last 5 Rows (df.tail()) ===")
    print(df.tail(5))
    print("\n=== Missing Values (df.isnull().sum()) ===")
    print(df.isnull().sum())
    print("\n=== Numerical Statistics (df.describe()) ===")
    print(eda_describe(df))
    print("\n=== Categorical Statistics (df.describe(include='categorical')) ===")
    print(eda_describe(df, include="categorical"))
    print("\n=== Price Correlations ===")
    print(eda_correlations(df))
    print("\n=== Top 5 States by Volume ===")
    print(eda_top_states(df, 5))
    print("\n=== Top 5 Costliest Commodities (median Rs) ===")
    print(eda_top_commodities(df, 5))



if __name__ == "__main__":
    raw = load_raw()
    clean = clean_data(raw)
    print("before:", raw.shape)
    print("after :", clean.shape)
    clean.to_csv("data/clean.csv", index=False)
    print("saved data/clean.csv\n")
    run_eda(clean)

