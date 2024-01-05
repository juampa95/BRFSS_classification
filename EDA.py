import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functions import hist_box, graf_cat, confusion_matrix



df_processed = pd.read_parquet("data/data.parquet")
df_transformed = pd.read_parquet("data/data_preprocessed.parquet")

"""
df_processed es el dataframe que solo tiene las columnas seleccionadas. 
df_transformed es el dataframe resultante del archivo preprocessing en donde se modificaron variables categoricas 
aplicando conceptos de dicotomía y encoders como OHE. 
"""
dfp = df_processed.copy()
dfp["DIABETE4"] = dfp["DIABETE4"].apply(lambda x: 1 if x == 1 else 0)
dft = df_transformed.copy().round(0).astype(int)

dfp_columns = dfp.columns.to_list()

# fig, axes = plt.subplots(10,4,figsize=(30,35))
# plt.suptitle('HISTOGRAMAS DE TODAS LAS VARIABLES, DISCRIMINADOS POR TARGET DIABETE4', fontsize=16, y=0.9)
#
# i = 0
# for fil in range(10):
#     for col in range(4):
#         if i <=37:
#             sns.histplot(data=dfp,
#                          ax=axes[fil,col],
#                          x=dfp_columns[i],
#                          hue='DIABETE4'
#                          )
#             i += 1
# plt.show()

target = 'DIABETE4'
# Voy a hacer graficos diferentes para variables categoricas nominales que para variables numericas discretas.
dfp_col_num = ['_BMI5']
dfp_col_cat = [x for x in dfp_columns if x not in dfp_col_num]

graf_cat(dfp,dfp_col_cat[26],1,target)
graf_cat(dft,dfp_col_cat[24],1,target)

dft.head()

hist_box(dfp,dfp_col_cat[14],1)
hist_box(dft,dfp_col_cat[36],1,1)


for i in range(30,40):
    try:
        graf_cat(dft, dfp_col_cat[i], 1, target)
    except:
        try:
            graf_cat(dfp, dfp_col_cat[i], 1, target)
        except:
            pass


