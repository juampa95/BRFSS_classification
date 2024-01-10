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

target = 'DIABETE4'
# Voy a hacer graficos diferentes para variables categoricas nominales que para variables numericas discretas.
dfp_col_num = ['_BMI5']
dfp_col_cat = [x for x in dfp_columns if x not in dfp_col_num]

# graf_cat(dfp,dfp_col_cat[26],1,target)
# graf_cat(dft,dfp_col_cat[24],1,target)
#
# dft.head()
#
# hist_box(dfp,dfp_col_cat[14],1)
# hist_box(dft,dfp_col_cat[36],1,bins=1)

path = 'data/graphs/EDA/'

# VARIABLE NUMERICA

hist_box(dft,dfp_col_num[0],1,path=path)

# CATEGORICAS

# for i in range(len(dfp_col_cat)):
for i in range(10):
    try:
        graf_cat(dft, dfp_col_cat[i], 1, target,path=path)
    except:
        try:
            graf_cat(dfp, dfp_col_cat[i], 1, target,path=path)
        except:
            pass

# VARIABLES CON 2do GRAFICO

hist_box(dft,dfp_col_cat[dfp_col_cat.index('MENTHLTH')],1, bins=1, path=(path+'2'))
hist_box(dft,dfp_col_cat[dfp_col_cat.index('PHYSHLTH')],1, bins=1, path=(path+'2'))
hist_box(dft,dfp_col_cat[dfp_col_cat.index('POTADA1_')],1, path=(path+'2'))


# ANALISIS INDIVIDAL DE VARIABLES

# _RFHYPE6

conteodf = dft['_RFHYPE6'].groupby(dft[target]).value_counts().unstack(fill_value=0).T
conteodf['% con diabetes'] = conteodf[1]/conteodf.sum(axis=1)*100
print(conteodf)

# El 5,8% de la gente sin presión tiene diabetes y el 24,4% de la gente con presión alta tiene diabetes



