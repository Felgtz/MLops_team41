# MLOps Team 41 – Online News Popularity

## Project Overview
This repository contains the complete work developed during **Phase 1** of the MLOps course at **Tecnológico de Monterrey (Applied Artificial Intelligence – MNA Program)**.  
The dataset analyzed corresponds to **Online News Popularity**, focused on predicting the number of social media shares of news articles published by Mashable.

The goal of this phase was to:
1. **Clean and validate** a noisy dataset.
2. **Explore and preprocess** relevant features.
3. **Version artifacts and models** using DVC.
4. **Train and evaluate baseline models** with full reproducibility.

---

## Repository Structure

```
.
├── data/
│   ├── df_final_validated.csv           # Cleaned dataset (output of notebook 01)
│   ├── features_used.csv                # Feature list used during model training
│   ├── online_news_original.csv         # Original dataset from UCI ML Repository
│   └── online_news_modified.csv         # Modified dataset provided by TA’s (with injected inconsistencies)
│
├── notebooks/
│   ├── V2_01_EDA_and_Data_Cleaning.ipynb                # Data repair, validation, and export (author: Steven)
│   ├── V2_02_Data_Exploration_and_Preprocessing.ipynb   # Exploratory analysis, feature scaling, encoding (author: Felipe)
│   └── V2_03_Model_Construction_and_Evaluation.ipynb    # Modeling, evaluation, and model selection (author: Felipe & Steven)
│
├── reports/
│   ├── Executive_Deck_Final.pdf         # Final stakeholder presentation (Phase 1 summary)
│   ├── Executive_Deck_v2.pdf            # Early version of the presentation
│   ├── MLOps_team41_presentation.mp4    # Video presentation of the Phase 1 results
│
├── Machine Learning Canvas - Online News Popularity.pdf  # ML Canvas and value proposition
│
├── dvc.yaml                              # DVC pipeline definition
├── dvc.lock                              # DVC lockfile
├── .dvc/                                 # DVC metadata folder
└── README.md
```

---

## Roles and Responsibilities (Phase 1)

| Role | Member | Main Responsibilities |
|------|--------|------------------------|
| **Data Scientist** | **Steven Sebastian Brutscher Cortez (A01732505)** | Data repair, imputation, domain validation, export of `df_final_validated.csv`; documentation and reproducibility. |
| **ML Engineer** | **Felipe de Jesús Gutiérrez Dávila (A01360023)** | Preliminary EDA, preprocessing references, initial baselines. |
| **Software Engineer** | **Ana Karen Estupiñán Pacheco (A01796893)** | Data and model versioning using **DVC**; integration with Git for experiment reproducibility. |
| **Data Engineer** | **Ángel Iván Ahumada Arguelles (A00398508)** | Executive presentation and pipeline documentation for Phase 1 deliverables. |

---

## Tools and Technologies

- **Python**, **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, **Seaborn**
- **Google Colab** (development and experimentation)
- **DVC** for artifact and model versioning
- **GitHub** for collaborative version control and documentation

---

## Deliverables (Phase 1)

1. **V2_01_EDA_and_Data_Cleaning.ipynb**  
   - Repairs invalid values and enforces domain constraints (proportions, polarities, binaries, LDA topics).  
   - Exports the final validated dataset `data/df_final_validated.csv`.

2. **V2_02_Data_Exploration_and_Preprocessing.ipynb**  
   - Generates descriptive statistics, correlations, and distribution insights.  
   - Scales and encodes features; exports final ready-to-model dataset.

3. **V2_03_Model_Construction_and_Evaluation.ipynb**  
   - Compares multiple regression baselines: Linear, Ridge, KNN, Random Forest, Decision Tree, and XGBoost.  
   - Evaluates results using **MAE**, **RMSE**, and **R²** metrics on original scale.  
   - Selects XGBoost as the most consistent and generalizable model.

4. **Executive Deck (Final PDF)**  
   - Summarizes project scope, key findings, and methodology for academic presentation.

5. **Video Presentation (MP4)**  
   - 5–10 minute explanation of the project and Phase 1 workflow.

6. **DVC Repository**  
   - Tracks datasets, features, and model artifacts to ensure full reproducibility of the workflow.

---

## Results Summary

**Model Comparison (Test Set Performance)**

| Model | MAE_test | RMSE_test | R²_test |
|:------|----------:|-----------:|---------:|
| **XGBoost** | **1851.43** | **4018.48** | **0.0116** |
| Random Forest | 1878.84 | 4028.64 | 0.0066 |
| KNN (k=15) | 1924.58 | 4111.83 | -0.0348 |
| Ridge Regression | 1911.31 | 4114.96 | -0.0364 |
| Linear Regression | 1911.31 | 4114.95 | -0.0364 |
| Decision Tree | 2712.27 | 4930.57 | -0.4880 |

**Interpretation:**
- The **XGBoost Regressor** achieved the best performance overall, with the lowest MAE/RMSE and the only positive R², confirming slight predictive power over random baselines.  
- The **Random Forest** performed similarly, validating the strength of ensemble methods on this dataset.  
- Simpler models (Linear, Ridge) could not capture non-linear relationships, while the **Decision Tree** overfitted the data.  
- Overall, the models confirm the high noise and non-linearity in predicting article virality, consistent with prior research (Fernandes et al., 2015).

**Outcome:**  
Phase 1 successfully delivered a **fully reproducible pipeline** capable of training and evaluating machine learning baselines on the *Online News Popularity* dataset.  
The **XGBoost model** is selected as the foundation for **Phase 2 hyperparameter tuning** and **feature explainability (SHAP/LIME)** analysis.

---

## How to Run Locally

1. **Create and activate a Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux/Mac)
   venv\Scripts\activate     # (Windows)
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **If using DVC, pull versioned artifacts:**
   ```bash
   dvc pull
   ```

4. **Run notebooks sequentially:**
   - `notebooks/V2_01_EDA_and_Data_Cleaning.ipynb`
   - `notebooks/V2_02_Data_Exploration_and_Preprocessing.ipynb`
   - `notebooks/V2_03_Model_Construction_and_Evaluation.ipynb`

---

## DVC Quick Reference

| Action | Command |
|:--------|:---------|
| Initialize DVC | `dvc init` |
| Track file | `dvc add data/df_final_validated.csv` |
| Commit to Git | `git add data/df_final_validated.csv.dvc .gitignore && git commit -m "Track cleaned dataset with DVC"` |
| Set remote storage | `dvc remote add -d origin <remote-url>` |
| Push artifacts | `dvc push` |
| Reproduce pipeline | `dvc repro` |

---

## References

- **UCI Machine Learning Repository:** Online News Popularity dataset.  
- **Fernandes, Vinagre, & Cortez (2015):** *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News.*  
- Course materials and MLOps Phase 1 rubric (Tecnológico de Monterrey).  
- Internal documentation and DVC logs.

---

## License and Credits

This work includes a structured **Machine Learning Canvas** adapted from Louis Dorard, Ph.D., under **CC BY-SA 4.0**.  
All other content authored by **MLOps Team 41** for academic and non-commercial purposes.

---

## Video Presentations

- **Version 1:** [https://youtu.be/dAoLZClsZGE] 
- **Version 2 (Final):** *(link pending upload)*

---
