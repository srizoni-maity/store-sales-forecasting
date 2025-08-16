# src/train_xgboost.py

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Custom safe MAPE function (handles divide by zero)
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_xgboost(df, store_id=1, family='GROCERY I'):
    # Filter specific store and product family
    df_store = df[(df['store_nbr'] == store_id) & (df['family'] == family)]
    
    # Drop missing values (from lag features)
    df_store = df_store.dropna()

    # Select features and target
    features = [
        'onpromotion', 'is_holiday', 'is_weekend', 'dayofweek',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14'
    ]
    target = 'sales'

    X = df_store[features]
    y = df_store[target]

    # Time-aware split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model training
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)

    print(f"✅ XGBoost Results for Store {store_id}, {family}")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

    # 📊 Plot forecast
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:100], label='Actual', color='teal')
    plt.plot(y_pred[:100], label='Predicted', color='orange')
    plt.title(f'XGBoost Sales Forecast: Store {store_id} - {family}')
    plt.xlabel('Sample Days')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)

    # Save output
    os.makedirs("models", exist_ok=True)
    plot_path = f"models/xgb_forecast_store{store_id}_{family.replace(' ', '_')}.png"
    plt.savefig(plot_path)

    # Save prediction data
    result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    result_csv_path = f"models/xgb_forecast_store{store_id}_{family.replace(' ', '_')}.csv"
    result_df.to_csv(result_csv_path, index=False)

    return model

# Main execution
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed.csv", parse_dates=['date'])
    model = train_xgboost(df, store_id=1, family='GROCERY I')
