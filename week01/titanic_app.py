import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(page_title="Titanic Explorer", page_icon="🚢", layout="wide")

# Title
st.title("🚢 Titanic Survival Explorer")
st.markdown("**Interactive data exploration app — built with Streamlit + Plotly**")
st.divider()


# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\DS-AI-75d\titanic.csv")


df = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────
st.sidebar.header("🔧 Filters")

# Class filter
classes = st.sidebar.multiselect(
    "Passenger Class",
    options=[1, 2, 3],
    default=[1, 2, 3],
    format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class",
)

# Gender filter
genders = st.sidebar.multiselect(
    "Gender", options=["male", "female"], default=["male", "female"]
)

# Age filter
age_min, age_max = st.sidebar.slider(
    "Age Range", min_value=0, max_value=80, value=(0, 80)
)

# Apply filters
filtered = df[
    (df["Pclass"].isin(classes))
    & (df["Sex"].isin(genders))
    & (df["Age"].between(age_min, age_max) | df["Age"].isna())
]

# ── KEY METRICS ───────────────────────────────────
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Passengers", len(filtered))
with col2:
    survivors = filtered["Survived"].sum()
    st.metric("Survivors", int(survivors))
with col3:
    rate = filtered["Survived"].mean()
    st.metric("Survival Rate", f"{rate:.1%}")
with col4:
    avg_age = filtered["Age"].mean()
    st.metric("Average Age", f"{avg_age:.1f}")

st.divider()

# ── CHARTS ───────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Survival by Class")
    class_data = filtered.groupby("Pclass")["Survived"].mean().reset_index()
    class_data.columns = ["Class", "Survival Rate"]
    class_data["Class"] = class_data["Class"].map({1: "1st", 2: "2nd", 3: "3rd"})
    fig1 = px.bar(
        class_data,
        x="Class",
        y="Survival Rate",
        color="Survival Rate",
        color_continuous_scale="RdYlGn",
        text="Survival Rate",
        template="plotly_dark",
    )
    fig1.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig1.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Survival by Gender")
    sex_data = filtered.groupby("Sex")["Survived"].mean().reset_index()
    fig2 = px.bar(
        sex_data,
        x="Sex",
        y="Survived",
        color="Sex",
        color_discrete_map={"male": "#38bdf8", "female": "#f472b6"},
        template="plotly_dark",
    )
    fig2.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# Age distribution
st.subheader("Age Distribution")
fig3 = px.histogram(
    filtered.dropna(subset=["Age"]),
    x="Age",
    color="Survived",
    color_discrete_map={0: "#ef4444", 1: "#22c55e"},
    barmode="overlay",
    nbins=30,
    opacity=0.7,
    template="plotly_dark",
)
fig3.update_layout(height=350)
st.plotly_chart(fig3, use_container_width=True)

# ── DATA TABLE ────────────────────────────────────
st.subheader("📋 Filtered Data")
st.dataframe(
    filtered[["Name", "Pclass", "Sex", "Age", "Fare", "Survived"]].reset_index(
        drop=True
    ),
    use_container_width=True,
    height=300,
)

st.caption(f"Showing {len(filtered)} passengers | Built with Streamlit + Plotly")
