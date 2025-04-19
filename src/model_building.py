import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml


train_data = pd.read_csv("./data/processed/train_processed.csv ")

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["model_building"]["n_estimators"]

clf = RandomForestClassifier(n_estimators=n_estimators)

clf.fit(X_train, y_train)

# save
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
