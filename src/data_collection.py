import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

df = pd.read_csv(r"C:/Users/UTENTE/Git Repos/water_potability.csv")

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["data_collection"]["test_size"]

train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
