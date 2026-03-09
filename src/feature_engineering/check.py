import pandas as pd

df = pd.read_csv("data/processed/tox21_clean.csv")

print(df.columns)
print(df.head())
print(df.iloc[0])