import pandas as pd
csv_path = r"C:/Users/FGUTIE63/Desktop/MLOps/Fase 2/MLops_team41-main/MLops_team41-main/MCE/Model_construction_and_evaluation/data/raw/df_ready_for_modeling.csv"

df = pd.read_csv(csv_path, nrows=5)      # use small head for speed
print(df.columns.tolist())               # full column list
print(df["shares"].head())               # sample values
print(df["shares"].dtype)                # dtype check