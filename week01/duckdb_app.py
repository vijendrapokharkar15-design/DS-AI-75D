import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px

st.set_page_config(page_title="DuckDB SQL Explorer", page_icon="🦆", layout="wide")

st.title("🦆 DuckDB SQL Explorer")
st.markdown("**Run SQL queries directly on the Titanic dataset**")
st.divider()


# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\DS-AI-75d\titanic.csv")


df = load_data()
con = duckdb.connect()

# Schema info
with st.expander("📋 Dataset Schema — Click to expand"):
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
            use_container_width=True,
        )
    with col2:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
        st.metric("Missing Values", int(df.isnull().sum().sum()))

st.divider()

# Preset queries
st.subheader("⚡ Quick Queries")
preset = st.selectbox(
    "Choose a preset query or write your own below:",
    [
        "Custom query",
        "Survival rate by class",
        "Survival rate by gender and class",
        "Top 10 highest fares",
        "Average age per title",
        "Window function — fare rank in class",
    ],
)

preset_queries = {
    "Survival rate by class": """
SELECT
    Pclass,
    COUNT(*) AS total,
    SUM(Survived) AS survivors,
    ROUND(AVG(Survived) * 100, 1) AS survival_rate_pct
FROM df
GROUP BY Pclass
ORDER BY Pclass""",
    "Survival rate by gender and class": """
SELECT
    Pclass,
    Sex,
    COUNT(*) AS total,
    ROUND(AVG(Survived) * 100, 1) AS survival_rate_pct
FROM df
GROUP BY Pclass, Sex
ORDER BY Pclass, Sex""",
    "Top 10 highest fares": """
SELECT
    Name,
    Pclass,
    Sex,
    Age,
    Fare,
    Survived
FROM df
WHERE Fare IS NOT NULL
ORDER BY Fare DESC
LIMIT 10""",
    "Average age per title": """
SELECT
    CASE
        WHEN Name LIKE '%Mr.%' THEN 'Mr'
        WHEN Name LIKE '%Mrs.%' THEN 'Mrs'
        WHEN Name LIKE '%Miss.%' THEN 'Miss'
        WHEN Name LIKE '%Master.%' THEN 'Master'
        ELSE 'Other'
    END AS Title,
    COUNT(*) AS count,
    ROUND(AVG(Age), 1) AS avg_age,
    ROUND(AVG(Survived) * 100, 1) AS survival_pct
FROM df
WHERE Age IS NOT NULL
GROUP BY Title
ORDER BY count DESC""",
    "Window function — fare rank in class": """
SELECT
    Name,
    Pclass,
    Fare,
    RANK() OVER (PARTITION BY Pclass ORDER BY Fare DESC) AS rank_in_class,
    ROUND(AVG(Fare) OVER (PARTITION BY Pclass), 2) AS class_avg_fare
FROM df
WHERE Fare IS NOT NULL
ORDER BY Pclass, rank_in_class
LIMIT 15""",
}

# Query editor
default_query = preset_queries.get(preset, "SELECT * FROM df LIMIT 10")
query = st.text_area("✏️ SQL Query", value=default_query, height=180)

# Run query
if st.button("▶ Run Query", type="primary"):
    try:
        result = con.execute(query).df()
        st.success(f"✅ Query returned {len(result)} rows")

        # Show results
        st.dataframe(result, use_container_width=True, height=350)

        # Auto visualise if numeric columns exist
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        if len(result.columns) >= 2 and len(numeric_cols) >= 1:
            st.subheader("📊 Auto Visualisation")
            cat_cols = [c for c in result.columns if c not in numeric_cols]
            if cat_cols and numeric_cols:
                fig = px.bar(
                    result,
                    x=cat_cols[0],
                    y=numeric_cols[0],
                    color=numeric_cols[0],
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    title=f"{numeric_cols[0]} by {cat_cols[0]}",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Query error: {e}")

st.divider()
st.caption("Built with DuckDB + Streamlit + Plotly | Day 7 — 75-Day DS & AI Roadmap")
