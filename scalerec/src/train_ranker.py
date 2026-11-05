import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_features():
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))

    df = interactions.merge(users, on="user_id").merge(items, on="item_id")
    df["price_log"] = np.log1p(df["price"])
    df["age_bucket"] = (df["age"] // 10) * 10

    cat_cols = ["gender", "city", "category", "brand"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    feature_cols = [c for c in df.columns if c not in ["user_id", "item_id", "rating", "clicked", "purchased", "age", "price"]]
    X = df[feature_cols]
    y = df["purchased"]
    return X, y


def train():
    X, y = build_features()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation AUC: {val_auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "xgb_ranker.json"))
    print(f"Saved ranker to {MODEL_DIR}/xgb_ranker.json")


if __name__ == "__main__":
    train()
