# Diabetes Health Indicators Classification System

An experimental machine learning application that uses health and lifestyle indicators to classify whether a survey respondent reported having diabetes or prediabetes.

## Project Overview

This project explores how machine learning can identify patterns associated with diabetes-related survey responses using the CDC's Diabetes Health Indicators dataset.

Users can enter health and lifestyle information through a Streamlit interface and receive a model-generated likelihood score.

Important: This is an educational machine learning project, not a medical diagnostic tool. It does not predict future disease or replace professional medical advice.

## Dataset

- Source: UCI Machine Learning Repository
- Dataset: CDC Diabetes Health Indicators
- Original records: 253,680
- Features: 21
- Target: Diabetes_binary

The target combines respondents reporting prediabetes or diabetes into one positive class.

The dataset is based on US survey responses. Its performance and applicability in other populations have not been validated.

## Machine Learning Models

The project compares:

- Logistic Regression
- Random Forest
- Threshold-optimized Logistic Regression

The optimized Logistic Regression model is used in the Streamlit application.

## Final Model Evaluation

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.43% | 31.83% | 76.01% | 44.87% | 0.8106 |
| Random Forest | 82.53% | 42.62% | 41.15% | 41.87% | 0.8012 |
| Optimized Logistic Regression | 64.14% | 28.07% | 86.04% | 42.33% | 0.8106 |

The optimized model prioritizes recall, meaning it identifies a larger proportion of positive cases but also produces more false positives.

These results are experimental and should not be interpreted as clinical performance.

## Project Structure

`text
Healthcare-Risk-Prediction/
│
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
├── models/
├── notebooks/
└── reports/