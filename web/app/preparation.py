# Импорт библиотек
import numpy as np # для работы с массивами
import pandas as pd # для работы с DataFrame 
from sklearn import preprocessing #предобработка
import sklearn.preprocessing as pp # импорт для работы с кодировщиком


def normalize_data(df):
        
    scaler = preprocessing.RobustScaler()
    scaler.fit_transform(df)
    df_scaler = scaler.transform(df)
    
    return df_scaler