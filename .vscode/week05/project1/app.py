import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# load model
with open("model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/features.pkl", "rb") as f:
    features = pickle.load(f)

# page config
st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")
st.divider()

# sidebar inputs
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    options=[1, 2, 3],
    format_func=lambda x: (
        f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else "3rd Class"
    ),
)

sex = st.sidebar.radio("Sex", ["Male", "Female"])

age = st.sidebar.slider("Age", min_value=1, max_value=80, value=30)

fare = st.sidebar.slider("Fare (£)", min_value=0, max_value=520, value=32)

sibsp = st.sidebar.number_input("Siblings / Spouses aboard", 0, 8, 0)

parch = st.sidebar.number_input("Parents / Children aboard", 0, 6, 0)

has_cabin = st.sidebar.checkbox("Has cabin number?", value=False)

title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

st.divider()

# build input features — mirrors train_model.py exactly
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_log = np.log1p(fare)
sex_encoded = 1 if sex == "Female" else 0

input_data = {
    "Pclass": pclass,
    "Age": age,
    "FareLog": fare_log,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "HasCabin": int(has_cabin),
    "Sex_encoded": sex_encoded,
    "Title_Mr": 1 if title == "Mr" else 0,
    "Title_Mrs": 1 if title == "Mrs" else 0,
    "Title_Miss": 1 if title == "Miss" else 0,
    "Title_Master": 1 if title == "Master" else 0,
}

input_df = pd.DataFrame([input_data])

# prediction
prob = model.predict_proba(input_df)[0][1]
pred = "Survived ✅" if prob >= 0.5 else "Died ❌"

# results
col1, col2, col3 = st.columns(3)
col1.metric("Prediction", pred)
col2.metric("Survival Probability", f"{prob*100:.1f}%")
col3.metric("Family Size", family_size)

st.divider()

# probability gauge
st.subheader("Survival Probability")
color = "green" if prob >= 0.6 else "orange" if prob >= 0.4 else "red"
st.progress(float(prob))
if prob >= 0.7:
    st.success(f"High survival chance — {prob*100:.1f}%")
elif prob >= 0.4:
    st.warning(f"Uncertain — {prob*100:.1f}% survival chance")
else:
    st.error(f"Low survival chance — {prob*100:.1f}%")

st.divider()

# passenger summary
st.subheader("Passenger Summary")
summary_cols = st.columns(4)
summary_cols[0].metric(
    "Class", f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}"
)
summary_cols[1].metric("Age", age)
summary_cols[2].metric("Fare", f"£{fare}")
summary_cols[3].metric("Alone?", "Yes" if is_alone else "No")

st.divider()

# SHAP explanation
st.subheader("Why this prediction?")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    shap_vals = shap_values[:, :, 1].values[0]

    shap_df = pd.DataFrame(
        {
            "Feature": features,
            "SHAP Value": shap_vals,
            "Feature Value": input_df.values[0],
        }
    ).sort_values("SHAP Value", key=abs, ascending=False)

    for _, row in shap_df.iterrows():
        direction = "↑ helps survival" if row["SHAP Value"] > 0 else "↓ hurts survival"
        color_indicator = "🟢" if row["SHAP Value"] > 0 else "🔴"
        st.write(
            f"{color_indicator} **{row['Feature']}** = {row['Feature Value']:.2f} "
            f"→ SHAP: {row['SHAP Value']:+.4f} {direction}"
        )
except Exception as e:
    st.write("SHAP explanation unavailable")

st.divider()
st.caption("Built with Phase 2 techniques — Days 16-29 | DS-AI-75D Project 1")
