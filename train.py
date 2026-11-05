
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load training and testing datasets
# --------------------------------------------------

train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("test_data.csv")

X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]

X_test = test_data.drop(columns=["diagnosis"])
y_test = test_data["diagnosis"]

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Create model directory if it does not exist
os.makedirs("model", exist_ok=True)

# --------------------------------------------------
# 2. Define the five classifiers
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=42))
    ])
}

# --------------------------------------------------
# 3. Train and evaluate each classifier
# --------------------------------------------------

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict test labels
    y_pred = model.predict(X_test)

    # Get probability of malignant class (class 1)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probability)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("model", filename))

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")

# --------------------------------------------------
# 4. Display and save model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nAll five models saved successfully in the model folder.")
print("Comparison saved as model_comparison.csv")
