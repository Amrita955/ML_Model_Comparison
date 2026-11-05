import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/breast_cancer.csv")

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].map({"B": 0, "M": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data["diagnosis"] = y_train

test_data = X_test.copy()
test_data["diagnosis"] = y_test

train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

print("\nTraining diagnosis distribution:")
print(train_data["diagnosis"].value_counts())

print("\nTesting diagnosis distribution:")
print(test_data["diagnosis"].value_counts())

print("\nTest data saved as test_data.csv")
