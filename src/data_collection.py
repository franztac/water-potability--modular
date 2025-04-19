import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple
import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data(filepath: str) -> pd.DataFrame:
    logging.info(f"----- {filepath} loaded")
    return pd.read_csv(filepath)


# df = pd.read_csv(r"C:/Users/UTENTE/Git Repos/water_potability.csv")


def load_params(filepath: str) -> float:
    with open(filepath, "r") as file:
        params = yaml.safe_load(file)
        logging.info(f"----- test_size loaded from {filepath}")
        return params["data_collection"]["test_size"]


# test_size = params["data_collection"]["test_size"]


def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"----- train test split with test size of {test_size * 100} %")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df


# train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    logging.info(f"----- saving data to {filepath}")
    df.to_csv(filepath, index=False)


# train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


def main():
    params_filepath = "params.yaml"
    data_filepath = "C:/Users/UTENTE/Git Repos/water_potability.csv"
    raw_datapath = os.path.join("data", "raw")

    test_size = load_params(params_filepath)
    df = load_data(data_filepath)
    train_df, test_df = split_data(df, test_size=test_size)

    logging.info(f"----- creating folder {raw_datapath}")
    os.makedirs(raw_datapath, exist_ok=True)

    save_data(train_df, os.path.join(raw_datapath, "train.csv"))
    save_data(test_df, os.path.join(raw_datapath, "test.csv"))


if __name__ == "__main__":
    main()
