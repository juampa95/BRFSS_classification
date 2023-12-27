import pandas as pd
import json

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)

df = pd.read_csv("data/LLCP2021.csv")
with open("data/features.json","r") as file:
    selected_columns = json.load(file)

selected_columns = list(selected_columns.keys())

df2 = df[selected_columns]
df2.shape

df2.to_csv("data/data.csv", index=False)
