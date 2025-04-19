from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import json


test_data = pd.read_csv("./data/processed/test_processed.csv ")
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


metrics_dict = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}

# save score
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f)
