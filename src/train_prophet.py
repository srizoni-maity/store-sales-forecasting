# src/train_prophet.py

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def train_prophet(df, store_id=1, family='GROCERY I'):
    # Filter for one store and product family
    store_df = df[(df['store_nbr'] == store_id) & (df['family'] == family)]
    
    # Rename columns for Prophet
    prophet_df = store_df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Initialize and fit model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Plot & Save
    fig = model.plot(forecast)
    plt.title(f"Prophet Forecast - Store {store_id}, {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    
    os.makedirs("models", exist_ok=True)
    fig.savefig(f"models/prophet_forecast_store{store_id}_{family.replace(' ', '_')}.png")

    # Save forecast
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f"models/prophet_forecast_store{store_id}_{family.replace(' ', '_')}.csv", index=False)
    
    print(f" Prophet forecast saved for Store {store_id}, {family}")
    return forecast

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("data/preprocessed.csv", parse_dates=['date'])

    # Train Prophet on one store/family
    forecast = train_prophet(df, store_id=1, family='GROCERY I')
