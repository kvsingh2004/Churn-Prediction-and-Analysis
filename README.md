# Customer Churn Prediction: End-to-End ML Pipeline 📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project is an end-to-end Machine Learning pipeline designed to predict whether a customer will cancel their service. 

By analyzing demographics, account information, and service usage, this model identifies high-risk customers, allowing businesses to take proactive retention measures. The final model is deployed as a real-time, interactive web application.

## 🚀 Key Features & Business Impact
* **High-Accuracy Prediction:** Evaluated multiple ML models (Logistic Regression, KNN, SVC, Decision Trees) on 1,000+ customer records. Selected a **Random Forest Classifier** achieving **91% accuracy**.
* **Imbalanced Data Handling:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to balance class distribution, ensuring the model doesn't just default to the majority class and provides realistic business value.
* **Interactive Dashboard:** Deployed a live web interface using **Streamlit**, allowing non-technical stakeholders to input customer parameters and receive instant churn probabilities.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Deployment:** Streamlit, Joblib (Model Serialization)

## 📂 Project Structure
```text
├── EDA_and_Model_Comparison.ipynb  # Data exploration, visualization, and algorithm testing
├── Production_Pipeline.ipynb       # Final pipeline: SMOTE, Random Forest training, and asset export
├── app.py                          # Streamlit web application script
├── customer_churn_data.csv         # Raw dataset
├── rf_churn_model.pkl              # Serialized Random Forest model
├── rf_churn_scaler.pkl             # Serialized StandardScaler
├── rf_churn_columns.pkl            # Saved column headers for alignment
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
