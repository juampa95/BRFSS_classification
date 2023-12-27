import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from transformers import transform_1_1_other_0, transform_list_1_other_0, transform_days, transform_numbers, MyFunctionTransformer
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)

df = pd.read_csv("data/data.csv")
df = df.fillna(0)

column_names = df.columns.tolist()

columns_1_1_other_0 = ["DIABETE4", "TOLDHI3", "_CHOLCH3", "SMOKE100", "CVDSTRK3", "_MICHD", "_TOTINDA", "_FRTLT1A",
                       "_VEGLT1A", "_HLTHPLN", "MEDCOST1", "DIFFWALK", "_SEX", "EXERANY2", "BPMEDS", "CHOLMED3",
                       "ASTHNOW", "ADDEPEV3", "HAVARTH5", "DEAF", "BLIND", "DECIDE", "PNEUVAC4", "FLUSHOT7", ]
# Columnas en donde los valores que valen 1 quedan como 1 y los demas valdran 0

columns_2_1_other_0 = ["_RFHYPE6", "_RFDRHV7", "_LTASTH1"]
# Columnas en donde los valores que valen 2 quedan como 1 y los demas valdran 0

columns_ohe = ["GENHLTH", "EDUCA", "INCOME3", "_AGEG5YR", "PERSDOC3"]
# Columnas en donde se hace un OHE. Falta hacer el drop de alguna de ellas para evitar multicolinealidad

columns_different = ["MARITAL", "_URBSTAT", "MENTHLTH", "PHYSHLTH", "_BMI5", "POTADA1_"]
# Columnas que deberan tratarse de formas diferentes

ohe_transformer = OneHotEncoder(drop=None, sparse_output=False)
binary_transformer = MyFunctionTransformer(transform_1_1_other_0)
binary_transformer_2 = MyFunctionTransformer(transform_list_1_other_0, kw_args={'values': [2, 2.0]})
binary_transformer_3 = MyFunctionTransformer(transform_list_1_other_0, kw_args={'values': [1, 1.0]})
days_transformer = MyFunctionTransformer(transform_days)
numbers_not_null = MyFunctionTransformer(transform_numbers)

column_transformer = ColumnTransformer(
    transformers=[
        ('ohe', ohe_transformer, columns_ohe),
        ('tr_1', binary_transformer, columns_1_1_other_0),
        ('tr_2', binary_transformer_2, columns_2_1_other_0),
        ('tr_3', binary_transformer_3, ['MARITAL', '_URBSTAT']),
        ('transform_days', days_transformer, ['MENTHLTH', 'PHYSHLTH']),
        ('not_null', numbers_not_null, ['_BMI5', 'POTADA1_'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('colum_transform', column_transformer),
])

df_transformed = pipeline.fit_transform(df)

columnas_despues_ohe = list(column_transformer.named_transformers_['ohe'].get_feature_names_out(columns_ohe))

columnas_finales = columnas_despues_ohe + columns_1_1_other_0 + columns_2_1_other_0 + columns_different

df_resultado = pd.DataFrame(df_transformed, columns=columnas_finales)

print(df_resultado.head(100))