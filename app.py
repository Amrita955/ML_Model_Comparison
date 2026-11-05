
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ML Model Comparison Studio",
    page_icon="🧠",
    layout="wide"
)


# ==========================================================
# 2. CUSTOM CSS STYLING
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main heading */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
}

/* Section headings */

.section-heading {
    color: #172554;
    font-size: 23px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Metric cards */

.metric-card {
    background-color: white;
    padding: 20px 12px;
    border-radius: 12px;
    border: 1px solid #DCE3EF;
    text-align: center;
    box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04);
    margin-bottom: 10px;
}

.metric-value {
    font-size: 27px;
    font-weight: 750;
    color: #172554;
}

.metric-label {
    font-size: 13px;
    color: #64748B;
    margin-top: 5px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background-color: white;
    padding: 15px;
    border: 1px solid #DCE3EF;
    border-radius: 12px;
}

/* Data tables */

[data-testid="stDataFrame"] {
    border: 1px solid #DCE3EF;
    border-radius: 12px;
    overflow: hidden;
    background-color: white;
}

/* Buttons */

div.stButton > button {
    background-color: #1D4ED8;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #1E40AF;
    color: white;
}

/* Dividers */

hr {
    border-color: #DCE3EF;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. MODEL DEFINITIONS
# ==========================================================

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}


# ==========================================================
# 4. HEADER
# ==========================================================

st.markdown(
    '<p class="main-title">🧠 ML Model Comparison Studio</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Machine Learning Classification Dashboard | '
    'Breast Cancer Wisconsin Dataset'
    '</p>',
    unsafe_allow_html=True
)

st.info(
    "Compare five machine learning classifiers using "
    "standard classification evaluation metrics."
)


# ==========================================================
# 5. DATASET OVERVIEW
# ==========================================================

st.markdown(
    '<p class="section-heading">Dataset Overview</p>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Instances", "569")
col2.metric("Input Features", "30")
col3.metric("Classification Models", "5")
col4.metric("Target Classes", "2")

st.divider()


# ==========================================================
# 6. UPLOAD TEST DATA AND SELECT MODEL
# ==========================================================

st.markdown(
    '<p class="section-heading">Model Evaluation</p>',
    unsafe_allow_html=True
)

left, right = st.columns([1.2, 1])

with left:

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"]
    )

    st.caption(
        "Upload a CSV containing 30 numerical features "
        "and a diagnosis column (0 = Benign, 1 = Malignant)."
    )

with right:

    selected_model = st.selectbox(
        "Select Model for Detailed Evaluation",
        list(model_files.keys())
    )

evaluate_button = st.button(
    "Evaluate All Models",
    use_container_width=True
)


# ==========================================================
# 7. EVALUATE ALL FIVE MODELS ON UPLOADED DATA
# ==========================================================

if evaluate_button:

    if uploaded_file is None:

        st.error("Please upload a test CSV file.")

    else:

        try:

            data = pd.read_csv(uploaded_file)

            # Validate target column

            if "diagnosis" not in data.columns:

                st.error(
                    "The uploaded CSV must contain a diagnosis column."
                )

                st.stop()

            # Separate features and target

            X_test = data.drop(columns=["diagnosis"])
            y_test = data["diagnosis"]

            # Validate number of features

            if X_test.shape[1] != 30:

                st.error(
                    "The uploaded CSV must contain exactly 30 features."
                )

                st.stop()

            # Validate missing values

            if X_test.isnull().any().any() or y_test.isnull().any():

                st.error(
                    "Missing values found in the uploaded dataset."
                )

                st.stop()

            # Validate diagnosis values

            if not set(y_test.unique()).issubset({0, 1}):

                st.error(
                    "Diagnosis must contain only 0 (Benign) "
                    "and 1 (Malignant)."
                )

                st.stop()

            if y_test.nunique() != 2:

                st.error(
                    "Test data must contain both diagnosis classes."
                )

                st.stop()

            uploaded_results = []
            detailed_predictions = {}

            # Evaluate all five saved models

            for model_name, model_path in model_files.items():

                if not os.path.exists(model_path):

                    st.error(
                        f"Model file not found: {model_path}"
                    )

                    st.stop()

                # Load saved model without retraining

                model = joblib.load(model_path)

                # Match feature names and order

                expected_features = list(
                    model.feature_names_in_
                )

                if set(X_test.columns) != set(expected_features):

                    st.error(
                        f"Feature names do not match for {model_name}."
                    )

                    st.stop()

                X_aligned = X_test[expected_features]

                # Generate predictions

                predictions = model.predict(X_aligned)

                probabilities = model.predict_proba(
                    X_aligned
                )[:, 1]

                # Calculate all six metrics

                result = {
                    "Model": model_name,

                    "Accuracy": accuracy_score(
                        y_test, predictions
                    ),

                    "AUC": roc_auc_score(
                        y_test, probabilities
                    ),

                    "Precision": precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    "Recall": recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    "F1 Score": f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    "MCC": matthews_corrcoef(
                        y_test, predictions
                    )
                }

                uploaded_results.append(result)

                detailed_predictions[model_name] = {
                    "predictions": predictions,
                    "probabilities": probabilities
                }

            # Create comparison dataframe

            uploaded_comparison = pd.DataFrame(
                uploaded_results
            )

            st.success(
                "All five models evaluated successfully "
                "on the uploaded dataset!"
            )

            st.divider()


            # ==================================================
            # 8. UPLOADED DATASET MODEL COMPARISON
            # ==================================================

            st.markdown(
                '<p class="section-heading">'
                'Uploaded Dataset: Model Comparison'
                '</p>',
                unsafe_allow_html=True
            )

            # Highlight highest MCC

            def highlight_best_model(row):

                if row["MCC"] == uploaded_comparison["MCC"].max():

                    return [
                        "background-color: #DBEAFE; "
                        "color: #172554; font-weight: bold"
                    ] * len(row)

                return [""] * len(row)

            styled_results = (
                uploaded_comparison.style
                .format({
                    "Accuracy": "{:.4f}",
                    "AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC": "{:.4f}"
                })
                .apply(highlight_best_model, axis=1)
            )

            st.dataframe(
                styled_results,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Highlighted row: highest MCC on the uploaded test dataset."
            )


            # ==================================================
            # 9. SELECTED MODEL DETAILED EVALUATION
            # ==================================================

            st.divider()

            st.markdown(
                '<p class="section-heading">'
                'Detailed Model Evaluation'
                '</p>',
                unsafe_allow_html=True
            )

            st.write(
                f"Selected model: **{selected_model}**"
            )

            selected_predictions = detailed_predictions[
                selected_model
            ]

            y_pred = selected_predictions["predictions"]

            metrics = uploaded_comparison[
                uploaded_comparison["Model"] == selected_model
            ].iloc[0]

            # Display six metrics

            metric_columns = st.columns(3)

            metric_names = [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ]

            for index, metric_name in enumerate(metric_names):

                with metric_columns[index % 3]:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">
                                {metrics[metric_name]:.4f}
                            </div>
                            <div class="metric-label">
                                {metric_name}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # ==================================================
            # 10. CONFUSION MATRIX
            # ==================================================

            st.markdown(
                '<p class="section-heading">Confusion Matrix</p>',
                unsafe_allow_html=True
            )

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(
                figsize=(7, 4.5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                linewidths=1,
                linecolor="white",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                annot_kws={
                    "size": 14,
                    "weight": "bold"
                },
                ax=ax
            )

            ax.set_xlabel("Predicted Diagnosis")
            ax.set_ylabel("Actual Diagnosis")

            ax.set_title(
                selected_model,
                fontweight="bold"
            )

            st.pyplot(fig)
            plt.close(fig)


            # ==================================================
            # 11. CLASSIFICATION REPORT
            # ==================================================

            st.markdown(
                '<p class="section-heading">'
                'Classification Report'
                '</p>',
                unsafe_allow_html=True
            )

            report = classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Benign",
                    "Malignant"
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.style
                .format("{:.4f}")
                .background_gradient(
                    subset=[
                        "precision",
                        "recall",
                        "f1-score"
                    ],
                    cmap="Blues"
                ),
                use_container_width=True
            )


            # ==================================================
            # 12. DOWNLOAD UPLOADED DATA RESULTS
            # ==================================================

            csv_data = uploaded_comparison.to_csv(
                index=False
            )

            st.download_button(
                label="Download Uploaded Dataset Results",
                data=csv_data,
                file_name="uploaded_model_comparison.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(
                f"Evaluation failed: {error}"
            )


st.divider()


# ==========================================================
# 13. ORIGINAL TEST DATASET COMPARISON
# ==========================================================

st.markdown(
    '<p class="section-heading">'
    'Original Test Dataset: Model Comparison'
    '</p>',
    unsafe_allow_html=True
)

if os.path.exists("model_comparison.csv"):

    comparison = pd.read_csv(
        "model_comparison.csv"
    )

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Accuracy and F1 charts

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Accuracy Comparison**")

        st.bar_chart(
            comparison.set_index("Model")[["Accuracy"]]
        )

    with col2:

        st.markdown("**F1 Score Comparison**")

        st.bar_chart(
            comparison.set_index("Model")[["F1 Score"]]
        )

    # Overall winner based on MCC

    winner = comparison.loc[
        comparison["MCC"].idxmax()
    ]

    st.success(
        f"Best overall model by MCC: {winner['Model']} "
        f"| MCC: {winner['MCC']:.4f}"
    )

else:

    st.warning(
        "Model comparison file not found."
    )


st.divider()


# ==========================================================
# 14. DATASET INFORMATION
# ==========================================================

st.markdown(
    '<p class="section-heading">About the Dataset</p>',
    unsafe_allow_html=True
)

dataset_details = pd.DataFrame({

    "Property": [
        "Dataset Name",
        "Source",
        "Total Instances",
        "Input Features",
        "Target Variable",
        "Classification Type",
        "Benign Instances",
        "Malignant Instances"
    ],

    "Description": [
        "Breast Cancer Wisconsin (Diagnostic)",
        "UCI Machine Learning Repository",
        "569",
        "30 numerical features",
        "Diagnosis",
        "Binary Classification",
        "357",
        "212"
    ]
})

st.dataframe(
    dataset_details,
    use_container_width=True,
    hide_index=True
)

st.markdown("**Target Encoding**")

col1, col2 = st.columns(2)

with col1:
    st.info("0 = Benign")

with col2:
    st.warning("1 = Malignant")


# ==========================================================
# 15. FOOTER
# ==========================================================

st.divider()

st.warning(
    "This application is developed for academic and "
    "educational purposes only. It is not a medical "
    "diagnostic tool."
)

st.caption(
    "ML Model Comparison Studio | BITS Pilani Academic Assignment"
)
