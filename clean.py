import numpy as np
import pandas as pd


def load_raw(path="data/raw.csv"):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    df.columns = ["state", "district", "market", "commodity", "variety",
                  "grade", "arrival_date", "min_price", "max_price", "modal_price"]
    df = df.drop(columns=["arrival_date"])

    for col in ["state", "district", "market", "commodity", "variety", "grade"]:
        df[col] = df[col].str.strip()

    df["commodity"] = df["commodity"].str.replace(r"/हरी मटर", "", regex=False).str.strip().str.title()
    df["state"] = df["state"].replace({"Keralam": "Kerala", "Chattisgarh": "Chhattisgarh"})
    df = df.drop_duplicates()

    df = df[(df["min_price"] <= df["modal_price"]) & (df["modal_price"] <= df["max_price"])]
    df = df[(df["min_price"] >= 10) & (df["max_price"] >= 10) & (df["modal_price"] >= 10)]

    df["log_modal"] = np.log1p(df["modal_price"])
    df["spread"] = (df["max_price"] - df["min_price"]) / df["modal_price"]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    data = clean_data(raw)
    data.to_csv("data/clean.csv", index=False)
    print("saved clean data:", data.shape)
