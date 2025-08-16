# src/train_lstm.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 📦 Prepare sequence data for LSTM
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(df, store_id=1, family='GROCERY I'):
    # Filter the store/family
    store_df = df[(df['store_nbr'] == store_id) & (df['family'] == family)].copy()
    store_df = store_df.sort_values('date')
    store_df = store_df.dropna()

    # Use sales_scaled for LSTM
    sales_data = store_df['sales_scaled'].values

    # Define sequence length
    SEQ_LEN = 30
    X, y = create_sequences(sales_data, SEQ_LEN)

    # Reshape for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train with early stopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=50, batch_size=16, callbacks=[es], verbose=1)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/lstm_store{store_id}_{family.replace(' ', '_')}.h5")

    # 📉 Plot dark-themed loss curve
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss', color='cyan')
    plt.plot(history.history['val_loss'], label='Val Loss', color='magenta')
    plt.title('🧠 LSTM Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='dotted', alpha=0.3)
    plt.savefig(f"models/lstm_loss_curve_store{store_id}_{family.replace(' ', '_')}.png")

    print(f"✅ LSTM trained and saved for Store {store_id}, {family}")

    return model

# Run it
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed.csv", parse_dates=['date'])
    model = train_lstm(df, store_id=1, family='GROCERY I')
