import numpy as np
import pandas as pd

np.random.seed(42)

NUM_USERS = 500
NUM_ITEMS = 1000
NUM_INTERACTIONS = 5000

USER_FEAT_DIM = 10   # fake user features
ITEM_FEAT_DIM = 10   # fake item features

# Random features
user_features = np.random.rand(NUM_USERS, USER_FEAT_DIM).astype("float32")
item_features = np.random.rand(NUM_ITEMS, ITEM_FEAT_DIM).astype("float32")

# Random interactions (user_id, item_id, clicked)
interactions = pd.DataFrame({
    "user_id": np.random.randint(0, NUM_USERS, NUM_INTERACTIONS),
    "item_id": np.random.randint(0, NUM_ITEMS, NUM_INTERACTIONS),
    "clicked": np.random.randint(0, 2, NUM_INTERACTIONS),
})

# Save
np.save("data/user_features.npy", user_features)
np.save("data/item_features.npy", item_features)
interactions.to_csv("data/interactions.csv", index=False)

print("✅ Data saved!")
print(interactions.head())
