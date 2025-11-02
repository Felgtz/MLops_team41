import pandas as pd, pathlib, textwrap, numpy as np

path = pathlib.Path("data/processed/df_ready_for_modeling.csv")  # adjust if needed
df = pd.read_csv(path)

nan_mask   = df.isna()                     # True where pandas put NaN
blank_mask = (df == "") & df.dtypes.eq("object")  # True where empty string (object cols)

nan_cols   = nan_mask.any()
blank_cols = blank_mask.any()

print(textwrap.dedent(f"""
    File................: {path}
    Shape...............: {df.shape}
    Numeric NaN columns.: {list(df.columns[nan_cols])}
    Blank-string columns: {list(df.columns[blank_cols])}
    Total NaNs..........: {nan_mask.sum().sum()}
    Total empty strings.: {blank_mask.sum().sum()}
"""))