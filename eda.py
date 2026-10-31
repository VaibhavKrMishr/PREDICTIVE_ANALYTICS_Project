import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda_info(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Nulls": df.isnull().sum(),
        "Unique": df.nunique()
    }).reset_index(drop=True)


def eda_describe(df, include="numeric"):
    if include == "all":
        return df.describe(include="all").round(2)
    elif include in ["categorical", "cat", "str"]:
        return df.select_dtypes(include="object").describe()
    return df.describe().round(2)


def eda_correlations(df):
    return df[["min_price", "max_price", "modal_price"]].corr().round(3)


def eda_top_states(df, n=10):
    return df["state"].value_counts().head(n)


def eda_top_commodities(df, n=10):
    return df.groupby("commodity")["modal_price"].median().sort_values().tail(n)


def plot_price_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
    ax[0].hist(df["modal_price"], bins=40, color="teal")
    ax[0].set_title("Modal Price (Skewed)")
    ax[0].set_xlabel("Price in Rs")

    ax[1].hist(df["log_modal"], bins=40, color="steelblue")
    ax[1].set_title("Log Modal Price")
    ax[1].set_xlabel("log1p(Price)")

    fig.tight_layout()
    return fig


def plot_correlations(df):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.heatmap(eda_correlations(df), annot=True, cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Price Correlation")
    fig.tight_layout()
    return fig


def plot_top_states(df, n=10):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    eda_top_states(df, n).plot.barh(ax=ax, color="teal")
    ax.invert_yaxis()
    ax.set_title(f"Top {n} States by Volume")
    ax.set_xlabel("Records")
    fig.tight_layout()
    return fig


def plot_top_commodities(df, n=10):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    eda_top_commodities(df, n).plot.barh(ax=ax, color="orange")
    ax.set_title(f"Top {n} Costliest Commodities")
    ax.set_xlabel("Median Price in Rs")
    fig.tight_layout()
    return fig
