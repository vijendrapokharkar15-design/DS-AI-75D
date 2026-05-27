# 75-Day Data Science & AI Roadmap

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-latest-orange?logo=numpy)
![Status](https://img.shields.io/badge/Status-Day%2035%20of%2075-brightgreen)
> A fully documented 75-day execution plan to become job-ready for Data Scientist, ML Engineer, and AI Engineer roles — one deployed project per phase, built in public.

---

## 🗺️ Roadmap Progress

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| 1 — Foundations | 1–15 | Python · NumPy · Pandas · Polars · Stats · SQL | ✅ Complete |
| 2 — Core ML | 16–35 | Scikit-learn · XGBoost · Feature Engineering · SHAP | ✅ Complete|
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
| 21 | 08 Apr 2026 | XGBoost & Gradient Boosting: boosting vs bagging, GradientBoosting/XGBoost/LightGBM comparison, hyperparameter tuning — Logistic Regression wins on small Titanic dataset! | [Notebook](.vscode/week03/day21_xgboost.ipynb) |
| 22 | 09 Apr 2026 | Feature Engineering: feature creation (Title, FareLog, HasCabin, Age²), encoding (Label/OHE/Ordinal/Frequency), scaling (Standard/MinMax/Robust), feature selection — Logistic Regression improved +3.9% with engineered features! | [Notebook](.vscode/week04/day22_feature_engineering.ipynb) |
| 23 | 09 Apr 2026 | Clustering & Unsupervised Learning: K-Means (elbow method, silhouette score), DBSCAN (outlier detection, arbitrary shapes), PCA (explained variance, loadings) — found same patterns as supervised learning without labels! | [Notebook](.vscode/week04/day23_clustering.ipynb) |
| 24 | 10 Apr 2026 | Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV, Optuna (Bayesian optimisation) — improved RF AUC from 0.833 to 0.852 (+1.9%) with best params found in 34 seconds! | [Notebook](.vscode/week04/day24_hyperparameter_tuning.ipynb) |
| 25 | 10 Apr 2026 | Model Evaluation & Selection: bias-variance tradeoff, learning curves, StratifiedKFold, model comparison framework — Gradient Boosting best CV AUC (0.890) but Random Forest best generalisation! | [Notebook](.vscode/week04/day25_model_evaluation.ipynb) |
| 26 | 11 Apr 2026 | Imbalanced Data & SMOTE: accuracy paradox, class weights, SMOTE oversampling, undersampling, SMOTETomek, threshold tuning — improved minority recall from 9.1% to 81.8%! | [Notebook](.vscode/week04/day26_imbalanced_data.ipynb) |
| 27 | 11 Apr 2026 | SHAP & Model Explainability: TreeExplainer, SHAP values, bar plot, beeswarm, waterfall plots — gender swing of 23.5% confirmed, SHAP more honest than RF feature importance! | [Notebook](.vscode/week04/day27_shap_explainability.ipynb) |
| 28 | 12 Apr 2026 | Time Series Basics: datetime indexing, rolling statistics, seasonal decomposition, ADF stationarity test, ARIMA(2,1,2) forecasting — seasonal pattern recovered (April +22.9, October -19.9). | [Notebook](.vscode/week04/day28_time_series.ipynb) |
| 29 | 13 Apr 2026 | Phase 2 Review: complete ML pipeline comparison across all Days 16-28 — LR wins with Test AUC=0.872, 8 key lessons learned, Phase 2 complete! | [Notebook](.vscode/week05/day29_phase2_review.ipynb) |
| 30 | 13 Apr 2026 | Project 1 — Titanic Survival Predictor: full ML pipeline, Streamlit web app with SHAP explanations — 93.9% survival for 1st class female, 15.9% for 3rd class male! | [🌐 Live App](https://vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app) · [Code](.vscode/week05/project1/app.py) · [Notebook](.vscode/week05/day30_project1.ipynb) |
| 31 | 14 Apr 2026 | Project 1 Enhanced: 4-tab Streamlit app — Predict, Data Insights, Model Comparison (LR wins AUC=0.872), Batch Predict (305/891 survivors predicted, download CSV)! | [🌐 Live App](https://vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app) · [Code](.vscode/week05/project1/app.py) · [Notebook](.vscode/week05/day31_project1_enhanced.ipynb) |
| 32 | 14 Apr 2026 | Deployment: pushed to Streamlit Cloud — app live at vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app — accessible worldwide, auto-deploys on git push! | [🌐 Live App](https://vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app) · [Notebook](.vscode/week05/day32_deployment.ipynb) |
| 33 | 15 Apr 2026 | Project 1 Polish: professional README with badges, model metrics table, how to run locally — CV AUC=0.8852, Test AUC=0.8487, Precision=80%, Recall=69.6%! | [🌐 Live App](https://vijendrapokharkar15-design-titanic-survival-predicto-app-wbdy2y.streamlit.app) · [Notebook](.vscode/week05/day33_project1_polish.ipynb) |
| 34 | 15 Apr 2026 | Kaggle Titanic Submission: feature engineering pipeline on test set, RF + LR + GB + Ensemble predictions — Kaggle public score 0.76794 (beats naive baseline 0.766)! | [Notebook](.vscode/week05/day34_kaggle_submission.ipynb) |
| 35 | 16 Apr 2026 | Phase 2 Wrap-up: complete scorecard Days 16-35, 10 key lessons, Phase 3 preview — PyTorch, CNNs, NLP, Transformers, Hugging Face coming next! | [Notebook](.vscode/week05/day35_phase2_wrapup.ipynb) |
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
