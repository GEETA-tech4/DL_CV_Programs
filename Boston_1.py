import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load and scale data
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build DNN model (Linear Regression style)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Linear output (no activation)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train & Evaluate
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)
