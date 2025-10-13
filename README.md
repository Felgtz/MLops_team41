# MLOps Team 41 – Online News Popularity

## Project Overview
This repository contains the complete work developed during **Phase 1** of the MLOps course at Tecnológico de Monterrey (Applied Artificial Intelligence MNA program).  
The dataset analyzed corresponds to **Online News Popularity**, focused on predicting the number of social media shares of news articles.
The goal of this phase is to clean and prepare data, explore and preprocess features, version artifacts, and train baseline models with documented methodology.

## Repository structure

```
.
├── data/
│   ├── df_final_validated.csv           # cleaned dataset (Phase 1 output of 01 notebook)
│   └── online_news_original             # original dataset retrieved from UCI ML Repository
│   └── online_news_modified             # modified dataset provided by the TA's with intentional noise
├── notebooks/
│   ├── 01_EDA_and_Data_Cleaning.ipynb                   # First version of data repair, validation, export (author: Steven, DS)
│   ├── EDA_MLops_team41 ML.ipynb                        # First version of EDA + preprocessing + modeling + evaluation (author: Felipe, MLE)
│   ├── V2_01_EDA_and_Data_Cleaning.ipynb                # Second version of data repair, validation, export (author: Steven, DS)
│   ├── V2_02_Data_Exploration_and_Preprocessing.ipynb   # 2nd version of EDA + preprocessing (author: Felipe, MLE)
│   └── V2_03_Model_Construction_and_Evaluation.ipynb    # 2nd version of modeling + evaluation (author: Felipe, MLE)
├── Machine Learning Canvas - Online News Popularity     # ML Canvas and value proposition
├── reports/
│   ├── Executuve Deck v2.pdf             # 2nd version of executive Deck, a presentation of the Phase 1 for stakeholders
│   ├── MLOps team 41 presentation.mp4    # Video presentation of Executive Deck v2
├── dvc.yaml                              # DVC pipeline definition (if applicable)
├── dvc.lock                              # DVC lockfile (generated)
├── .dvc/                                 # DVC metadata
└── README.md
```

## Roles and responsibilities (Phase 1)

| Role | Member | Main responsibilities |
|------|--------|-----------------------|
| **Data Scientist** | Steven Sebastian Brutscher Cortez Brutscher (A01732505) | Data repair, domain enforcement, imputations, validation and export of `df_final_validated.csv`; documentation. |
| **ML Engineer** | Felipe de Jesús Gutiérrez Dávila (A01360023) | EDA, preprocessing, feature preparation; baseline and tuned models; evaluation. |
| **Software Engineer** | Ana Karen Estupiñán Pacheco (A01796893) | Data and model versioning using DVC; coordination with Git for reproducibility. |
| **Data Engineer** | Ángel Iván Ahumada Arguelles (A00398508) | Executive presentation and pipeline narrative for Phase 1 deliverables. |

##Tools and Technologies
- **Python**, **Pandas**, **Scikit-learn**, **Matplotlib**
- **Google Colab** for notebooks
- **DVC** for data and model versioning
- **GitHub** for collaborative version control

## Deliverables (Phase 1)

1. **01_EDA_and_Data_Cleaning.ipynb**  
   - Repairs domains and logical constraints across all feature families (proportions, polarities, binaries, LDA topics).  
   - Stabilizes distributions and exports `data/df_final_validated.csv`.
2. **02_Data_Exploration_and_Preprocessing.ipynb**  
   - Descriptive statistics, visual EDA, correlation analysis.  
   - Feature preprocessing: scaling, encoding, and dataset split export.
3. **03_Model_Construction_and_Evaluation.ipynb**  
   - Baseline and tuned models; metrics and error analysis; model selection rationale.
4. **Executive deck (PDF)**  
   - Summary of approach, key evidence, and handoff notes for Phase 2.
5. **Short video (5–10 min)**  
   - Team explanation of Phase 1 work aligned with the rubric.
6. **DVC repository**  
   - Tracks data and model artifacts; enables exact reproducibility.

## Results Summary
- **Mean correlation difference (Δ):** 0.0297 → strong feature preservation.
- **Best Model:** KNN (RMSE ≈ 3975, MAE ≈ 2350).
- **Outcome:** Solid baseline for predictive modeling of online news popularity.

## How to run locally

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If using DVC, pull versioned artifacts:
   ```bash
   dvc pull
   ```
4. Open the notebooks in order:
   - `notebooks/01_EDA_and_Data_Cleaning.ipynb`
   - `notebooks/02_Data_Exploration_and_Preprocessing.ipynb`
   - `notebooks/03_Model_Construction_and_Evaluation.ipynb`

## DVC quick reference

- Initialize: `dvc init`  
- Track a file: `dvc add data/df_final_validated.csv`  
- Commit to Git: `git add data/df_final_validated.csv.dvc .gitignore && git commit -m "Track cleaned dataset with DVC"`  
- Set remote storage: `dvc remote add -d origin <remote-url>`  
- Push artifacts: `dvc push`  
- Reproduce pipeline: `dvc repro` (if `dvc.yaml` is defined)

## References

- UCI Machine Learning Repository: Online News Popularity.  
- Fernandes, Vinagre, Cortez (2015). *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News*.
- Course materials and Phase 1 rubric.
- Team notebooks and DVC logs.

## License and credits
- This work includes a structured ML Canvas adapted from Louis Dorard, Ph.D., under CC BY-SA 4.0. Attribution retained per license. Course deliverables authored by Team 41 for academic purposes.

##VIDEO
-https://youtu.be/dAoLZClsZGE 
