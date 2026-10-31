# clean.py - data loading, cleaning, validation, and feature engineering for indian mandi prices

import numpy as np
import pandas as pd


def load_raw(path="data/raw.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # standardize column names
    df.columns = ["state", "district", "market", "commodity", "variety",
                  "grade", "arrival_date", "min_price", "max_price", "modal_price"]

    # arrival_date is invariant for this daily snapshot
    df = df.drop(columns=["arrival_date"])

    # clean text
    text_cols = ["state", "district", "market", "commodity", "variety", "grade"]
    for c in text_cols:
        df[c] = df[c].str.strip()
    df["commodity"] = df["commodity"].str.replace(r"/हरी मटर", "", regex=False).str.strip()
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
    # log transformation to normalize positive skew of modal price
    df["log_modal"] = np.log1p(df["modal_price"])
    # intra-day price spread
    df["spread"] = (df["max_price"] - df["min_price"]) / df["modal_price"]

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_data(raw)
    print("before:", raw.shape)
    print("after :", clean.shape)
    clean.to_csv("data/clean.csv", index=False)
    print("saved data/clean.csv successfully")
