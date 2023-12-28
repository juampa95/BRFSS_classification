import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer


class MyFunctionTransformer(FunctionTransformer):
    """
    Se uso para intentar extraer el nombre de las columnas al aplicar una funcion de transformacion propia
    No funciono.
    """
    def get_feature_names_out(self, input_features=None):
        return input_features

def transform_1_1_other_0(data):
    return data.apply(lambda column: pd.Series(np.where((column == 1) | (column == 1.0), 1, 0), index=column.index))


def transform_list_1_other_0(data, values: list):
    return data.apply(lambda column: pd.Series(np.where(column.isin(values), 1, 0), index=column.index))


def transform_days(data):
    return data.apply(lambda column: pd.Series(np.where((column >= 1) & (column <= 30), column, 0), index=column.index))


def transform_numbers(data):
    return data.apply(lambda column: pd.Series(np.where((column.notnull()), column, 0), index=column.index))