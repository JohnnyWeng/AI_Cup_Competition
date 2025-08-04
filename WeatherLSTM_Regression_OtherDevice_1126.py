
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import joblib

import numpy as np
import pandas as pd
import os

# Load the provided CSV files to examine their structure
file_A = 'D066_L08_0715_06.csv'  # 預測資料
file_B = 'D066_L08_0715_06.csv'  # 完整資料


# Read the files
data_A = pd.read_csv(file_A)
data_B = pd.read_csv(file_B)

# 截取前 12 個字元作為時間戳
data_A['Serial'] = data_A['Serial'].astype(str).str[:12]
data_B['Serial'] = data_B['Serial'].astype(str).str[:12]

# Ensure the 'Serial' column is treated as datetime for time series alignment
data_A['Serial'] = pd.to_datetime(data_A['Serial'], format='%Y%m%d%H%M%S')
data_B['Serial'] = pd.to_datetime(data_B['Serial'], format='%Y%m%d%H%M%S')

# Align the two datasets based on their time stamps and identify missing indices in L04
data_A = data_A.set_index('Serial')
data_B = data_B.set_index('Serial')


# Identify rows where Power(mW) in L04 is missing
missing_indices = data_A[data_A['Power(mW)'].isna()].index

# Confirm alignment and check for missing data
# data_L02_aligned.head(), data_L04_aligned.head(), missing_indices.tolist()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 在資料中加入時間相關特徵，如: day_of_week, hour, minute
data_A['day_of_week'] = data_A.index.dayofweek
data_A['hour'] = data_A.index.hour
data_A['minute'] = data_A.index.minute
data_B['day_of_week'] = data_B.index.dayofweek
data_B['hour'] = data_B.index.hour
data_B['minute'] = data_B.index.minute

# Merge L02 features with L04 where available
features_A = data_A[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'day_of_week', 'hour', 'minute']]
features_B = data_B[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'day_of_week', 'hour', 'minute']]
# features_A = data_A[['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'day_of_week', 'hour', 'minute']]
# features_B = data_B[['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'day_of_week', 'hour', 'minute']]
# features_A = data_A[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']]
# features_B = data_B[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']]
targets_A = data_A['Power(mW)']

# Fill missing values in L04 with corresponding L02 data for consistent input size
features_AA_filled = features_A.fillna(features_B)




# 設定 look_back
# look_back = 1
# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_AA_filled)
targets_scaled = scaler.fit_transform(targets_A.dropna().values.reshape(-1, 1))

# # Create training data
X = features_scaled[~targets_A.isna()]
y = targets_scaled



# # 創建包含 look_back 歷史步數的數據
# def create_dataset(data, look_back):
#     X, y = [], []
#     for i in range(len(data) - look_back):
#         X.append(data[i:i + look_back])
#         y.append(data[i + look_back])
#     return np.array(X), np.array(y)

# X, y = create_dataset(features_scaled, look_back)

# Prepare missing data for prediction
X_missing = features_scaled[targets_A.isna()]

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 確認形狀
print(X_train.shape)  # 應為 (樣本數, look_back, 特徵數)

print(np.isnan(X_train).sum(), np.isnan(y_train).sum())  # 檢查訓練標籤數據中的 NaN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector
from tensorflow.keras.optimizers import Adam

# Reshape data for LSTM input (samples, timesteps, features)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_missing_reshaped = X_missing.reshape((X_missing.shape[0], 1, X_missing.shape[1]))

print(f"X_missing_reshaped nan:{np.isnan(X_missing_reshaped).sum()}")  # 檢查是否有 NaN

# Build LSTM model
# Define the regressor model
lstm_model = Sequential()

# 1st LSTM Layer with 256 units and 'relu' activation
lstm_model.add(LSTM(units=512, activation='relu', return_sequences=True, input_shape=(1, X_train_reshaped.shape[2])))
# lstm_model.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], 5)))

# 2nd LSTM Layer with 128 units and 'relu' activation
lstm_model.add(LSTM(units=256, activation='relu'))
# lstm_model.add(LSTM(units=128))

# Add Dropout for regularization
lstm_model.add(Dropout(0.3))

# Output Layer
lstm_model.add(Dense(units=1, activation='relu'))

# 設定學習率
optimizer = Adam(learning_rate=0.001)  # 可調整學習率值

# Compile the model
lstm_model.compile(optimizer=optimizer, loss='mse')

from tensorflow.keras.callbacks import ReduceLROnPlateau

# 定義回調
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # 監測的指標
    factor=0.5,           # 每次降低學習率的倍數
    patience=10,          # 多少個 epoch 沒有改善後降低學習率
    min_lr=1e-6           # 最低學習率
)

# 訓練模型時添加回調
history = lstm_model.fit(
    X_train_reshaped, y_train,
    epochs=200, batch_size=16,
    validation_data=(X_val_reshaped, y_val),
    callbacks=[reduce_lr],  # 加入回調
    verbose=2
)

# import matplotlib.pyplot as plt

# # 繪製訓練和驗證損失
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()

print("Before prediction:", X_missing_reshaped.shape)

# Predict missing values
y_missing_scaled = lstm_model.predict(X_missing_reshaped)

print(f"y_missing_scaled_nan:{np.isnan(y_missing_scaled).sum()}")

print("Predicted values (scaled):", y_missing_scaled[:5])  # 查看前幾個預測結

# Inverse scale the predictions to original range
y_missing = scaler.inverse_transform(y_missing_scaled)

print("Inverse transformed missing values:", y_missing[:5])

# 'missing_indices' 是之前確定的 L04 缺失資料的索引
data_A.loc[missing_indices, 'Power(mW)'] = y_missing.flatten()

# 只提取 'Power(mW)' 欄位，轉為一維數組
output_power = data_A['Power(mW)'].values

# 保存結果到 CSV（無時間索引，僅電量）
output_file = f'{file_A[:8]}_filled.csv'
np.savetxt(output_file, output_power, fmt='%.2f', header='', comments='')

print(f"補全結果已儲存至 {output_file}")


