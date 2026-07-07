# 🩺 Diabetes Prediction using Machine Learning

A Machine Learning project built as part of my BSc Data Science and Artificial Intelligence coursework. This project predicts whether a patient is likely to have diabetes based on health parameters, using a Logistic Regression model, and provides an interactive web app for live predictions.

## 📌 Project Overview

This project uses the **Pima Indians Diabetes Dataset** to build a classification model that predicts diabetes based on features such as:
- Glucose level
- Blood Pressure
- BMI (Body Mass Index)
- Age
- Insulin level
- Number of Pregnancies
- Skin Thickness
- Diabetes Pedigree Function (family history factor)

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data loading and cleaning
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – machine learning model (Logistic Regression)
- **Streamlit** – interactive web application

## 📂 Files in this Repository

| File | Description |
|---|---|
| `diabetes_prediction.ipynb` | Jupyter Notebook with step-by-step analysis, model training, and evaluation |
| `app.py` | Streamlit web application with live prediction feature |
| `diabetes.csv` | Dataset used for training and testing |
| `requirements.txt` | Python libraries required to run the project |

## 🚀 How to Run This Project

1. Clone this repository
git clone https://github.com/vinay-data-code/diabetes-prediction.git
cd diabetes-prediction

2. Install the required libraries
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

4. The app will open in your browser. Enter patient details to get a live diabetes prediction.

## 🤖 Model Details

- **Algorithm used:** Logistic Regression
- **Train-Test Split:** 80% training, 20% testing
- **Feature Scaling:** StandardScaler was used to normalize the features
- **Accuracy achieved:** ~75%

## 📊 Key Insights

- Glucose level and BMI were found to be the features most strongly correlated with diabetes.
- The model can take new patient data as input and predict the likelihood of diabetes along with a probability score.

## 📁 Dataset Source

[Pima Indians Diabetes Database – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

