
# Machine Learning Model Comparison

## 1. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms on the same binary classification dataset.

The models are evaluated using six performance metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

The dataset contains 569 instances and 30 numerical input features computed from digitized images of breast mass samples.

The target variable is `diagnosis`:
- `0` – Benign (B)
- `1` – Malignant (M)

The dataset contains:
- 357 benign instances
- 212 malignant instances
- 30 numerical features
- No missing values

The ID column was removed because it is not a predictive feature.

The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

- Training samples: 455
- Testing samples: 114

## 3. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler was applied to Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes were trained without feature scaling.

## 4. Model Performance Comparison

All five models were evaluated on the same held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## 5. Observations

### Logistic Regression
- Achieved an accuracy of 96.49%.
- Obtained the highest AUC score of 0.9960.
- Demonstrated strong overall classification performance.

### Decision Tree
- Achieved an accuracy of 92.98%.
- Had the lowest AUC score among the five models.
- Provides an interpretable decision-making structure.

### K-Nearest Neighbors
- Achieved an accuracy of 95.61%.
- Produced strong precision and AUC scores.
- Feature scaling was applied because KNN relies on distance calculations.

### Gaussian Naive Bayes
- Achieved 100% precision.
- Had a recall of 83.33%, indicating that some malignant cases were classified incorrectly.
- Uses the Gaussian probability assumption for numerical features.

### Support Vector Machine
- Achieved the highest accuracy of 97.37%.
- Obtained the highest F1 Score of 0.9630 and MCC of 0.9442.
- Achieved 100% precision and 92.86% recall.

## 6. Overall Best Model

Based on the held-out test results, **Support Vector Machine (SVM)** performed best overall.

It achieved:
- Accuracy: 97.37%
- Precision: 100%
- Recall: 92.86%
- F1 Score: 0.9630
- MCC: 0.9442

Logistic Regression achieved the highest AUC score of 0.9960.

Therefore, SVM is the overall winner based on accuracy, F1 Score, and MCC, while Logistic Regression performed best in terms of AUC.

These results are based on one train-test split and should not be interpreted as evidence of clinical suitability.

## 7. Streamlit Application

The Streamlit application allows users to:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Evaluate all five models on the uploaded test dataset.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the selected model's confusion matrix and classification report.
- Compare model performance through tables and charts.

The application loads previously trained models. It does not retrain models using uploaded test data.

The uploaded CSV must contain all 30 input features and the `diagnosis` target column, encoded as 0 or 1.

## 8. Project Structure

```text
ML_Model_Comparison/
│
├── app.py
├── train.py
├── prepare_data.py
├── split_data.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
│
├── data/
│   ├── breast_cancer.csv
│   └── train_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
└── screenshots/
```

## 9. Installation and Execution

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ML_Model_Comparison
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 10. Repository and Deployment Links

**GitHub Repository:** Add the published GitHub repository URL here.

**Live Streamlit Application:** Add the deployed Streamlit application URL here.

## 11. Disclaimer

This project is developed for academic and educational purposes only. The dataset and model predictions must not be used for actual medical diagnosis or clinical decision-making.
