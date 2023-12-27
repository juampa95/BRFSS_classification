import pandas as pd
import pyarrow.parquet as pq

df_processed = pd.read_parquet("data/data.parquet")
df_transformed = pd.read_parquet("data/data_preprocessed.parquet")
