# 75-Day Data Science & AI Roadmap

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-latest-orange?logo=numpy)
![Status](https://img.shields.io/badge/Status-Day%2017%20of%2075-brightgreen)

> A fully documented 75-day execution plan to become job-ready for Data Scientist, ML Engineer, and AI Engineer roles — one deployed project per phase, built in public.

---

## 🗺️ Roadmap Progress

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| 1 — Foundations | 1–15 | Python · NumPy · Pandas · Polars · Stats · SQL | ✅ Complete |
| 2 — Core ML | 16–35 | Scikit-learn · XGBoost · Feature Engineering · SHAP | 🔄 In Progress |
| 3 — Deep Learning & NLP | 36–50 | PyTorch · Transformers · Hugging Face | ⏳ Upcoming |
| 4 — GenAI & LLMs | 51–63 | RAG · LangGraph · Fine-tuning · RAGAS | ⏳ Upcoming |
| 5 — MLOps | 64–70 | Docker · FastAPI · MLflow · Evidently AI | ⏳ Upcoming |
| 6 — Portfolio & Jobs | 71–75 | GitHub Polish · Resume · Job Applications | ⏳ Upcoming |

---

## 📅 Daily Build Log

| Day | Date | What I Built | Link |
|-----|------|-------------|------|
| 1 | 20 Mar 2026 | Environment setup (pyenv + uv + VS Code). NumPy: arrays, vectorisation, broadcasting, 1M-element benchmark. | [Notebook](day01_numpy_practice.ipynb) |
| 2 | 21 Mar 2026 | Pandas: DataFrames, loc/iloc, boolean filtering, groupby, merging. Titanic survival analysis — 5 business questions answered. | [Notebook](week01/notebooks/day02_pandas_fundamentals.ipynb) |
| 3 | 22 Mar 2026 | Advanced Pandas: apply, map, missing data imputation, pivot tables, crosstab, datetime operations on Titanic. | [Notebook](week01/notebooks/day03_advanced_pandas.ipynb) |
| 4 | 24 Mar 2026 | Polars: expressions API, lazy evaluation, groupby, filtering, Pandas vs Polars benchmark — 14.9x speedup on 1M rows. | [Notebook](week01/notebooks/day04_polars.ipynb) |
| 5 | 25 Mar 2026 | Matplotlib + Seaborn: bar chart, histogram, scatter, subplots, heatmap, boxplot, violin, FacetGrid — 7 chart types on Titanic. | [Notebook](week01/notebooks/day05_visualisation.ipynb) |
| 6 | 26 Mar 2026 | Plotly interactive charts (bar, scatter, histogram, boxplot, heatmap, dashboard) + Streamlit Titanic Explorer app with live filters. | [Notebook](week01/notebooks/day06_plotly_streamlit.ipynb) |
| 7 | 27 Mar 2026 | DuckDB: SQL on DataFrames, CSV queries, window functions, CTEs, 4.5x faster than Pandas. DuckDB SQL Explorer Streamlit app. | [Notebook](week01/notebooks/day07_duckdb.ipynb) |
| 8 | 01 Apr 2026 | Statistics & Probability: descriptive stats, distributions, hypothesis testing (t-test, p-value), correlation heatmap on Titanic. | [Notebook](.vscode/week02/day08_statistics.ipynb) |
| 9 | 02 Apr 2026 | Probability fundamentals: conditional probability, Bayes' theorem, Normal, Binomial & Poisson distributions with Titanic simulations. | [Notebook](.vscode/week02/day09_probability.ipynb) |
| 10 | 02 Apr 2026 | Inferential statistics: one-sample t-test, two-sample t-test, chi-square test, hypothesis testing framework on Titanic. | [Notebook](.vscode/week02/day10_inferential_statistics.ipynb) |
| 11 | 03 Apr 2026 | A/B Testing: confidence intervals, Type I & II errors, statistical power, sample size calculation, full A/B test simulation. | [Notebook](.vscode/week02/day11_ab_testing.ipynb) |
| 12 | 04 Apr 2026 | SQL Fundamentals: SELECT, WHERE, GROUP BY, HAVING, all JOIN types, subqueries, CASE WHEN — 15 queries on Titanic using DuckDB. | [Notebook](.vscode/week02/day12_sql_fundamentals.ipynb) |
| 13 | 04 Apr 2026 | Advanced SQL: window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD), CTEs, NTILE, running totals, moving averages on Titanic. | [Notebook](.vscode/week02/day13_advanced_sql.ipynb) |
| 14 | 05 Apr 2026 | SQL for Data Science: cohort analysis, funnel analysis, retention queries, date arithmetic, month-over-month trends on simulated e-commerce data. | [Notebook](.vscode/week02/day14_sql_for_ds.ipynb) |
| 15 | 05 Apr 2026 | Phase 1 Review: statistics quiz, SQL sprint, visual summary of all Phase 1 insights, knowledge checklist — 20% of journey complete! | [Notebook](.vscode/week03/day15_phase1_review.ipynb) |
| 16 | 06 Apr 2026 | Scikit-learn ML Workflow: data preparation, train/test split, Pipeline, ColumnTransformer, 5-fold cross-validation, Logistic Regression — 80.4% accuracy on Titanic. | [Notebook](.vscode/week03/day16_sklearn_ml_workflow.ipynb) |
| 17 | 06 Apr 2026 | Linear Regression: OLS from scratch in NumPy, Ridge/Lasso/ElasticNet regularisation, regression metrics (MSE, RMSE, MAE, R²), residuals analysis on Titanic fares. | [Notebook](.vscode/week03/day17_linear_regression.ipynb) |
| 18 | 07 Apr 2026 | Logistic Regression: sigmoid function, all classification metrics (accuracy, precision, recall, F1, ROC-AUC, log-loss), confusion matrix, threshold tuning — 81% accuracy, AUC=0.853. | [Notebook](.vscode/week03/day18_logistic_regression.ipynb) |
| 19 | 07 Apr 2026 | Decision Trees: Gini impurity, information gain, depth control, overfitting visualisation, feature importance — tree independently discovered "women and children first"! | [Notebook](.vscode/week03/day19_decision_trees.ipynb) |
| 20 | 08 Apr 2026 | Random Forests: bagging, bootstrap aggregation, OOB score, n_estimators analysis, feature importance vs single tree — RF gives Fare 75x more importance than single tree! | [Notebook](.vscode/week03/day20_random_forests.ipynb) |
---

## 🚀 Projects (Building Throughout the Journey)

| # | Project | Stack | Status | Demo |
|---|---------|-------|--------|------|
| 1 | Predictive Analytics App | XGBoost · SHAP · Streamlit | ⏳ Day 30–35 | — |
| 2 | NLP Classifier API | DistilBERT · FastAPI · Docker | ⏳ Day 48–49 | — |
| 3 | AI-Powered RAG Assistant | LlamaIndex · Pinecone · Claude API | ⏳ Day 63 | — |
| 4 | Production ML Service | FastAPI · MLflow · GitHub Actions | ⏳ Day 70 | — |

---

## 🛠️ Tech Stack

**Data & ML**
`Python 3.12` `NumPy` `Pandas` `Polars` `DuckDB` `Scikit-learn` `XGBoost` `LightGBM` `Optuna` `SHAP`

**Deep Learning & NLP**
`PyTorch` `Hugging Face Transformers` `Sentence Transformers`

**GenAI & LLMs**
`OpenAI API` `Anthropic API` `LangGraph` `LlamaIndex` `ChromaDB` `Pinecone` `RAGAS`

**MLOps & Engineering**
`FastAPI` `Docker` `MLflow` `Weights & Biases` `GitHub Actions` `Evidently AI`

**Visualisation & Apps**
`Matplotlib` `Seaborn` `Plotly` `Streamlit`

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/vijendrapokharkar15-design/DS-AI-75D.git
cd DS-AI-75D
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

---

## 👤 About Me

Building from Python foundations to production GenAI systems — one commit at a time.

🔗 [LinkedIn](https://www.linkedin.com/in/vijendra-pokharkar-394380115) · 📧 vijendrapokharkar15@gmail.com
