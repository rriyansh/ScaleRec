import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NUM_USERS = 10000
NUM_ITEMS = 5000
NUM_INTERACTIONS = 200000


def generate_synthetic_data():
    rng = np.random.default_rng(42)

    users = pd.DataFrame({
        "user_id": range(NUM_USERS),
        "age": rng.integers(18, 70, size=NUM_USERS),
        "gender": rng.choice(["M", "F", "O"], size=NUM_USERS),
        "city": rng.choice(["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"], size=NUM_USERS),
    })

    items = pd.DataFrame({
        "item_id": range(NUM_ITEMS),
        "category": rng.choice(["electronics", "fashion", "grocery", "books", "home"], size=NUM_ITEMS),
        "price": rng.uniform(100, 50000, size=NUM_ITEMS).round(2),
        "brand": rng.choice(["BrandA", "BrandB", "BrandC", "BrandD"], size=NUM_ITEMS),
    })

    interactions = pd.DataFrame({
        "user_id": rng.integers(0, NUM_USERS, size=NUM_INTERACTIONS),
        "item_id": rng.integers(0, NUM_ITEMS, size=NUM_INTERACTIONS),
        "rating": rng.choice([1, 2, 3, 4, 5], size=NUM_INTERACTIONS, p=[0.05, 0.1, 0.15, 0.35, 0.35]),
        "clicked": rng.choice([0, 1], size=NUM_INTERACTIONS, p=[0.3, 0.7]),
        "purchased": rng.choice([0, 1], size=NUM_INTERACTIONS, p=[0.9, 0.1]),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
    items.to_csv(os.path.join(OUTPUT_DIR, "items.csv"), index=False)
    interactions.to_csv(os.path.join(OUTPUT_DIR, "interactions.csv"), index=False)

    return users, items, interactions


if __name__ == "__main__":
    generate_synthetic_data()
    print("Synthetic data generated in scalerec/data/")
