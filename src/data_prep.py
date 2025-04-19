import pandas as pd
import numpy as np
import os


train_df = pd.read_csv("./data/raw/train.csv")
test_df = pd.read_csv("./data/raw/test.csv")


def fill_missing_values_with_median(df_original):
    df = df_original.copy()
    for column in df.columns:
        if df[column].isna().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    return df


# Fill missing values with median
train_processed_df = fill_missing_values_with_median(train_df)
test_processed_df = fill_missing_values_with_median(test_df)

data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

train_processed_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
