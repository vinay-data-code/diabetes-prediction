import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# ---- TITLE ----
st.title("🩺 Diabetes Prediction App")
st.write("This Machine Learning project predicts whether a patient is likely to have diabetes based on their health parameters.")

# ---- LOAD DATA ----
df = pd.read_csv("diabetes.csv")

# ---- SHOW RAW DATA (Optional toggle) ----
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

st.markdown("---")

# ---- TRAIN THE MODEL (runs once, cached for speed) ----
@st.cache_resource
def train_model(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, accuracy, cm, X.columns.tolist()

model, scaler, accuracy, cm, feature_names = train_model(df)

# ---- MODEL PERFORMANCE ----
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
col2.pyplot(fig_cm)

st.markdown("---")

# ---- DATA VISUALIZATION ----
st.subheader("📈 Data Insights")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Outcome", y="Glucose", palette="Set2", ax=ax1)
    ax1.set_title("Glucose Level by Diabetes Outcome")
    ax1.set_xlabel("Outcome (0 = No, 1 = Yes)")
    st.pyplot(fig1)

with viz_col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Outcome", y="BMI", palette="Set3", ax=ax2)
    ax2.set_title("BMI by Diabetes Outcome")
    ax2.set_xlabel("Outcome (0 = No, 1 = Yes)")
    st.pyplot(fig2)

st.markdown("---")

# ---- LIVE PREDICTION SECTION ----
st.subheader("🔮 Try It Yourself: Predict Diabetes")
st.write("Enter patient details below to get a live prediction from the trained model.")

pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

with pred_col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)

with pred_col2:
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with pred_col3:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

with pred_col4:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    new_patient = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    new_patient_scaled = scaler.transform(new_patient)
    prediction = model.predict(new_patient_scaled)
    probability = model.predict_proba(new_patient_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ This patient is LIKELY to have Diabetes (Probability: {probability*100:.1f}%)")
    else:
        st.success(f"✅ This patient is UNLIKELY to have Diabetes (Probability: {probability*100:.1f}%)")

st.markdown("---")

# ---- CONCLUSION ----
st.subheader("📌 Conclusion")
st.write("""
- We trained a Logistic Regression model to predict whether a patient has diabetes based on health parameters such as Glucose level, BMI, Age, Blood Pressure, and Insulin.
- Glucose and BMI were found to be the features most strongly correlated with diabetes.
- The model achieved an accuracy of approximately 75% on the test dataset.
- Users can enter their own health data above to get a live prediction from the model.
""")