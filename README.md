# 🕵️‍♂️ Credit Card Fraud Detection using Machine Learning  
A complete end-to-end project for detecting fraudulent credit card transactions using the popular **Kaggle Credit Card Fraud Dataset**.  
This project includes:

✅ Data preprocessing  
✅ Handling class imbalance  
✅ Feature scaling  
✅ Model training (XGBoost / Random Forest fallback)  
✅ Threshold tuning for best F1-score  
✅ Model evaluation (F1, Precision, Recall, ROC-AUC, PR-AUC, RMSE, MAE)  
✅ Predicting user-given transactions  
✅ Saving & loading trained models  

---

## 📌 Project Overview

Financial fraud is a major concern in the digital world. Fraudulent transactions are extremely rare (≈0.17%), making fraud detection an **imbalanced classification problem**.

This project trains a supervised machine learning model to classify:  
- `0` → Legitimate Transaction  
- `1` → Fraudulent Transaction  

The model uses **XGBoost** (if available) or **Random Forest** as fallback.

---

## 🚀 Features

✔ Processes raw dataset  
✔ Cleans missing & invalid values  
✔ Scales important columns (`Time`, `Amount`)  
✔ Handles imbalance using `scale_pos_weight`  
✔ Trains ML model using a clean pipeline  
✔ Finds the **best probability threshold** using F1 optimization  
✔ Evaluates model on multiple metrics  
✔ Saves model using `joblib`  
✔ Predicts user-provided transaction in real time  

---

## 📂 Dataset  
This project uses the public Kaggle dataset:

**Credit Card Fraud Detection Dataset**  
Link: https://www.kaggle.com/mlg-ulb/creditcardfraud

- 284,807 transactions  
- 492 frauds (0.172%)  
- Features `V1`–`V28` are PCA-transformed  
- Includes `Time`, `Amount`, and `Class`
- ## 🧠 Model

The training code automatically selects:

### ✅ **XGBoostClassifier**  
If installed, uses:
- 800 estimators  
- scale_pos_weight (for class imbalance)  
- learning rate: 0.05  
- max_depth: 4  

### ✅ **RandomForestClassifier**  
Fallback option with:
- 800 trees  
- Balanced subsample class weights  

---

## 📊 Metrics Used

The model prints all key metrics:

- **F1-score** (primary metric)  
- **Precision**
- **Recall**
- **Balanced Accuracy**
- **ROC-AUC**
- **PR-AUC** (Average Precision)
- **RMSE** (proba vs true label)
- **MAE** (proba vs true label)

These metrics ensure robustness for highly imbalanced fraud detection.

---

## 📈 Threshold Tuning

Instead of using a default 0.5 probability threshold, the model finds the **best threshold** that maximizes F1-score on the validation set.
