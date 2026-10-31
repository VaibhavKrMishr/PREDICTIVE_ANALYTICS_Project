# eda.py - exploratory data analysis functions and visualization for mandi dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def eda_correlations(df):
    return df[["min_price", "max_price", "modal_price"]].corr().round(3)


def eda_top_states(df, n=10):
    return df["state"].value_counts().head(n)


def eda_top_commodities(df, n=10):
    return df.groupby("commodity")["modal_price"].median().sort_values().tail(n)


def plot_price_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].hist(df["modal_price"], bins=45, color="#059669", edgecolor="white", alpha=0.85)
    ax[0].set_title("Modal Price (Skewed)", fontsize=8, fontweight="bold", color="#1e293b")
    ax[0].set_xlabel("Price in Rs", fontsize=8)
    ax[0].tick_params(labelsize=8)
    ax[0].grid(axis="y", linestyle="--", alpha=0.3)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    ax[1].hist(df["log_modal"], bins=45, color="#0284c7", edgecolor="white", alpha=0.85)
    ax[1].set_title("Log-Transformed Price", fontsize=8, fontweight="bold", color="#1e293b")
    ax[1].set_xlabel("log1p(Price)", fontsize=8)
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
    ax.set_title("Price Correlations", fontsize=8, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_top_states(df, n=10):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    data = eda_top_states(df, n)
    data.plot.barh(ax=ax, color="#0d9488", edgecolor="white", alpha=0.9)
    ax.invert_yaxis()
    ax.set_title(f"Market Records per State (Top {n})", fontsize=8, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Number of Records", fontsize=8)
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
    ax.set_title("Highest Median Modal Price (Rs)", fontsize=8, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Median Price (Rs)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import clean
    raw = clean.load_raw()
    df = clean.clean_data(raw)
    print("=== eda info ===")
    print(eda_info(df))
    print("\n=== price correlations ===")
    print(eda_correlations(df))
    print("\n=== top 5 states ===")
    print(eda_top_states(df, 5))
