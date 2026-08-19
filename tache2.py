import pandas as pd
df =pd.read_csv("dataset.csv")
df.shape
df.info()
df.describe()
df.head(10)
df.columns.tolist()
