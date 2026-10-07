# ❤️ Heart Disease Prediction Using XGBoost

A machine learning web application that predicts the likelihood of heart disease based on patient health information.

The model is built using **XGBoost** and deployed as an interactive **Streamlit** application.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not a medical diagnostic system and should not be used to make medical decisions.

---

## 🚀 Live Application

After deploying on Streamlit Community Cloud, add your application URL here:

**Live Demo:** `https://heart-disease-xgboost.streamlit.app/`

---

## 📌 Project Overview

This project uses a Heart Disease dataset to train a binary classification model.

The model predicts two classes:

| Class | Meaning          |
| ----- | ---------------- |
| `0`   | No Heart Disease |
| `1`   | Heart Disease    |

The user enters patient information through a Streamlit web interface. The application sends the information to the trained XGBoost pipeline and displays the predicted condition and prediction probabilities.

---

## 🧠 Machine Learning Model

The project uses:

**XGBoost Classifier**

XGBoost is a gradient boosting algorithm that builds multiple decision trees sequentially. Each new tree attempts to improve the predictions made by the previous trees.

### Model Parameters

```text
n_estimators = 100
max_depth = 3
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
objective = binary:logistic
eval_metric = logloss
random_state = 42
```

---

## 📊 Dataset

The original dataset contained:

```text
Rows: 1025
Columns: 14
```

The dataset contained a large number of duplicate records.

### Duplicate Analysis

```text
Original rows:       1025
Duplicate rows:       723
Unique rows:          302
```

After removing duplicate records:

```text
Clean rows:           302
Features:              13
Target:                 1
```

### Target Distribution

```text
No Heart Disease:     138
Heart Disease:        164
```

Approximately:

```text
No Heart Disease:     45.70%
Heart Disease:        54.30%
```

---

## 📋 Dataset Features

The model uses 13 input features.

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `age`      | Patient age                       |
| `sex`      | Sex category                      |
| `cp`       | Chest pain type                   |
| `trestbps` | Resting blood pressure            |
| `chol`     | Cholesterol                       |
| `fbs`      | Fasting blood sugar               |
| `restecg`  | Resting ECG result                |
| `thalach`  | Maximum heart rate achieved       |
| `exang`    | Exercise-induced angina           |
| `oldpeak`  | ST depression                     |
| `slope`    | Slope of peak exercise ST segment |
| `ca`       | Number of major vessels           |
| `thal`     | Thalassemia category              |

The target column is:

```text
target
```

where:

```text
0 = No Heart Disease
1 = Heart Disease
```

---

## 🔄 Machine Learning Workflow

The project follows this workflow:

```text
Original Dataset
       │
       ▼
Data Inspection
       │
       ▼
Duplicate Detection
       │
       ▼
Remove Duplicates
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Separate Features and Target
       │
       ▼
Train/Test Split
       │
       ▼
One-Hot Encoding
       │
       ▼
XGBoost Training
       │
       ▼
Model Evaluation
       │
       ▼
Cross-Validation
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Final Pipeline
       │
       ▼
Pickle Model
       │
       ▼
Streamlit Application
```

---

## 🧹 Data Cleaning

The first major issue found in the dataset was duplicate data.

There were:

```text
1025 total records
723 duplicate records
```

After removing duplicates:

```text
302 unique records
```

The cleaning operation was:

```python
df_clean = df.drop_duplicates().reset_index(drop=True)
```

No missing values were found after cleaning.

---

## 🔀 Train/Test Split

The cleaned dataset was divided into:

```text
80% Training Data
20% Testing Data
```

Results:

```text
Training samples: 241
Testing samples:   61
```

Stratified splitting was used to maintain approximately the same target distribution in both datasets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## 🔤 Feature Encoding

Eight features were treated as categorical:

```text
sex
cp
fbs
restecg
exang
slope
ca
thal
```

Five features were treated as numerical:

```text
age
trestbps
chol
thalach
oldpeak
```

### One-Hot Encoding

The categorical features were transformed using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

The categorical features produced:

```text
25 encoded features
```

Together with the five numerical features:

```text
25 categorical features
+
5 numerical features
=
30 model features
```

---

## 🤖 Why Use a Pipeline?

The final model uses a Scikit-learn Pipeline.

The pipeline combines:

```text
Input Data
     │
     ▼
ColumnTransformer
     │
     ▼
OneHotEncoder
     │
     ▼
XGBoost
     │
     ▼
Prediction
```

This is important for deployment because the Streamlit application does not need to manually reproduce the encoding process.

The same preprocessing used during training is automatically applied to new user input.

---

## 📈 Model Performance

### Original Manual XGBoost

Test Accuracy:

```text
78.69%
```

Test ROC-AUC:

```text
87.45%
```

### Final Baseline Pipeline

Test Accuracy:

```text
77.05%
```

Test ROC-AUC:

```text
87.45%
```

### Confusion Matrix

```text
[[21, 7],
 [ 7,26]]
```

Meaning:

```text
True Negative  = 21
False Positive = 7
False Negative = 7
True Positive   = 26
```

---

## 📊 Classification Report

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------: | -----: | -------: |
| No Heart Disease |      0.75 |   0.75 |     0.75 |
| Heart Disease    |      0.79 |   0.79 |     0.79 |

Overall test accuracy:

```text
77.05%
```

---

## 📐 ROC-AUC

The final pipeline achieved:

```text
Test ROC-AUC = 87.45%
```

ROC-AUC evaluates how well the model separates the two classes across different classification thresholds.

---

## 🔁 Cross-Validation

A 5-fold cross-validation experiment was performed.

### Accuracy

```text
Fold 1: 89.80%
Fold 2: 81.25%
Fold 3: 83.33%
Fold 4: 91.67%
Fold 5: 75.00%
```

Mean accuracy:

```text
84.21%
```

Standard deviation:

```text
6.02%
```

### ROC-AUC

```text
Fold 1: 95.29%
Fold 2: 89.86%
Fold 3: 87.59%
Fold 4: 96.33%
Fold 5: 83.57%
```

Mean ROC-AUC:

```text
90.53%
```

Standard deviation:

```text
4.77%
```

---

## 🎯 Hyperparameter Tuning

GridSearchCV was used to test multiple XGBoost parameter combinations.

The best parameters found were:

```python
{
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
```

Best cross-validation ROC-AUC:

```text
91.04%
```

However, the tuned model did not perform better on the independent test set.

Therefore, the **baseline pipeline was selected as the final deployment model**.

---

## 💾 Model File

The trained model is stored as:

```text
heart_disease_xgb_pipeline.pkl
```

This pickle file contains the complete preprocessing and XGBoost model pipeline.

---

## 🌐 Streamlit Application

The application allows the user to enter:

```text
Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate
Exercise-Induced Angina
ST Depression
ST Slope
Major Vessels
Thalassemia
```

After clicking:

```text
🔍 Predict Heart Disease
```

the application displays:

```text
Prediction
Probability
Probability Chart
Patient Input Summary
Model Explanation
```

---

## 📁 Project Structure

The GitHub repository should contain:

```text
heart-disease-xgboost/
│
├── app.py
│
├── heart_disease_xgb_pipeline.pkl
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Requirements

The `requirements.txt` file contains:

```text
streamlit
pandas
scikit-learn
xgboost
```

---

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-xgboost.git
```

Move into the project:

```bash
cd heart-disease-xgboost
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:

```text
app.py
heart_disease_xgb_pipeline.pkl
requirements.txt
README.md
```

3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select the repository.
6. Select the `main` branch.
7. Select:

```text
app.py
```

8. Click **Deploy**.

Streamlit will install the packages from `requirements.txt` and run the application.

---

## 🧪 Example Prediction

Example input:

```text
Age: 55
Sex: Male
Chest Pain Type: Typical Angina
Resting BP: 140
Cholesterol: 250
Fasting Blood Sugar: No
Resting ECG: Normal
Maximum Heart Rate: 150
Exercise Angina: No
ST Depression: 1.0
ST Slope: Downsloping
Major Vessels: 0
Thalassemia: 2
```

The application will return a prediction such as:

```text
Prediction:
Heart Disease

Probability:
XX.XX%
```

The exact result depends on the trained model.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Streamlit charts

### Web Application

* Streamlit

### Model Serialization

* Pickle

### Deployment

* GitHub
* Streamlit Community Cloud

---

## 📚 What I Learned

This project demonstrates:

* Data cleaning
* Duplicate detection
* Exploratory Data Analysis
* Feature/target separation
* Train/test splitting
* Label Encoding
* One-Hot Encoding
* XGBoost classification
* Model evaluation
* Confusion matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* Cross-validation
* GridSearchCV
* Feature importance
* Scikit-learn Pipelines
* Pickle model saving
* Streamlit development
* GitHub deployment

---

## ⚠️ Medical Disclaimer

This application is a machine-learning demonstration.

It is **not a medical diagnosis tool**.

The predictions are generated from patterns learned from a dataset and may be inaccurate for individual patients.

Do not use the prediction to:

* Diagnose heart disease
* Start or stop medication
* Replace a doctor
* Make emergency medical decisions

If someone has concerning symptoms or health concerns, they should seek advice from a qualified healthcare professional.

---

## 👨‍💻 Author

**Amaan Ali**

Data Science | AI/ML | Web Development | Digital Marketing

GitHub:

`https://github.com/YOUR_USERNAME`

LinkedIn:

`https://www.linkedin.com/in/amaan-ali-21784824b/`

Portfolio:

`https://www.amaanali.rf.gd`

---

## ⭐ Project Goal

The goal of this project is to demonstrate how a machine-learning classification model can be trained, evaluated, serialized, and converted into an interactive web application using Streamlit.

If you find this project useful, consider giving the repository a ⭐.
