# src/preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import holidays

def load_data(filepath="data/train.csv"):
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(" Data loaded with shape:", df.shape)
    return df

def handle_missing_values(df):
    # Fill 'onpromotion' NaNs with 0
    if 'onpromotion' in df.columns:
        df['onpromotion'].fillna(0, inplace=True)
    
    # Check for other missing values
    missing_report = df.isnull().sum()
    print("🧹 Missing Values Report:\n", missing_report[missing_report > 0])
    
    return df

def add_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

def add_holiday_feature(df):
    ecuador_holidays = holidays.Ecuador()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in ecuador_holidays else 0)
    return df

def add_lag_features(df):
    df = df.sort_values(by=['store_nbr', 'family', 'date'])
    for lag in [1, 7, 14]:
        df[f'sales_lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)
    return df

def scale_sales(df):
    df['sales_scaled'] = df.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    )
    return df

def save_preprocessed_data(df, path="data/preprocessed.csv"):
    df.to_csv(path, index=False)
    print(f" Preprocessed data saved to: {path}")

def preprocess_pipeline():
    df = load_data()
    df = handle_missing_values(df)
    df = add_date_features(df)
    df = add_holiday_feature(df)
    df = add_lag_features(df)
    df = scale_sales(df)

    # Drop rows with NaNs from lag features
    df.dropna(inplace=True)

    save_preprocessed_data(df)
    return df

if __name__ == "__main__":
    preprocess_pipeline()
