# 🏥 Healthcare Diagnosis Prediction Pipeline using CatBoost & SMOTE

This repository contains a complete end-to-end machine learning pipeline for predicting **medical conditions** (e.g., Diabetes, Hypertension, Cancer, etc.) from structured healthcare data. The project includes advanced **feature engineering**, **class balancing** using SMOTE, and **model training** with CatBoostClassifier, with clear evaluation metrics and visualizations.

---

## 📁 File Structure

- `healthcare_dataset.csv`: The dataset containing patient information and medical condition labels.
- `healthcare_diagnosis.py`: Jupyter notebook with all code.
- `requirements.txt`: Python packages required to run the notebook.
- `README.md`: Project documentation.

---

## 📊 Dataset Description

The dataset should include the following columns:
Name, Age, Gender, Blood Type, Medical Condition, Date of Admission,
Doctor, Hospital, Insurance Provider, Billing Amount, Room Number,
Admission Type, Discharge Date, Medication, Test Results


The **target column** is `Medical Condition`.

---

## 📌 Pipeline Steps

### 1. 📦 Importing Libraries

Essential libraries include:
- `pandas`, `numpy` for data handling
- `sklearn` for preprocessing, metrics, model splitting
- `CatBoostClassifier` for classification
- `imblearn.SMOTE` for class balancing
- `seaborn`, `matplotlib` for visualization

---

### 2. 📥 Data Loading & Feature Engineering

- Load the dataset using `pandas`.
- Encode the target variable using `LabelEncoder`.
- Generate class weights for handling class imbalance.

#### 🔧 Engineered Features:

- `Age_50plus`, `Age_Squared`
- `Blood_Rh`, `Blood_Group`
- `Billing_per_Day`, `High_Biller`, `Length_of_Stay`
- `Urgent_Admission` based on `Admission Type`
- `Test_Abnormal` based on pattern in `Test Results`

---

### 3. 📚 Data Preparation

- Selected features:
Age, Age_50plus, Age_Squared, Gender, Blood_Group, Blood_Rh,
Billing Amount, Billing_per_Day, High_Biller,
Test Results, Test_Abnormal, Urgent_Admission


- Performed a `train_test_split` with `stratify=y`.

- Used `SMOTE` to oversample minority classes if any class had fewer than 100 samples.

---

### 4. 🧠 Model Training (CatBoost)

- Used `CatBoostClassifier` with:
- `iterations=1500`, `learning_rate=0.05`, `depth=8`
- `auto_class_weights='Balanced'`
- `early_stopping_rounds=50`
- `loss_function='MultiClass'`, `eval_metric='TotalF1'`

- Categorical features were detected dynamically:
- `['Gender', 'Blood_Group', 'Test Results']`

---

### 5. 📈 Evaluation

- Predictions evaluated using:
- `Balanced Accuracy`
- `Classification Report`
- `Confusion Matrix` (Seaborn heatmap)
- `Feature Importance` (Barplot)

#### 📉 Example Output:
Balanced Accuracy: 0.17

Classification Report:
precision recall f1-score support
Arthritis 0.14 0.17 0.16
Asthma 0.21 0.16 0.18
Cancer 0.16 0.13 0.14
Diabetes 0.17 0.14 0.16
Hypertension 0.19 0.21 0.20
Obesity 0.17 0.22 0.19
