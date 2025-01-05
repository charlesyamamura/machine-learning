import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load the input and target data
X = pd.read_excel('data1319.xlsx', sheet_name=0, usecols='D:S', skiprows=1, nrows=765).values
y = pd.read_excel('data1319.xlsx', sheet_name=0, usecols='T', skiprows=1, nrows=765).values

# Expand dimensions of target data if necessary
t = np.expand_dims(y, axis=1) if y.ndim == 1 else y

# Normalize input and target data
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

x_scaled = input_scaler.fit_transform(X)
t_scaled = target_scaler.fit_transform(t)

# Split data into training, validation, and testing sets
x_train, x_temp, t_train, t_temp = train_test_split(x_scaled, t_scaled, test_size=0.3, random_state=42)
x_val, x_test, t_val, t_test = train_test_split(x_temp, t_temp, test_size=0.5, random_state=42)

# Create a fitting network
hidden_layer_size = 10
model = Sequential([
    Dense(hidden_layer_size, activation='relu', input_shape=(x_scaled.shape[1],)),
    Dense(t_scaled.shape[1], activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['mae'])

# Train the network
history = model.fit(x_train, t_train, epochs=100, batch_size=10,
                    validation_data=(x_val, t_val), verbose=1)

# Test the network
y_pred = model.predict(x_test)
performance = model.evaluate(x_test, t_test, verbose=0)

# Display performance metrics
print("Test Performance (MSE):", performance)

# Inverse transform predictions and targets for interpretation
y_pred_original = target_scaler.inverse_transform(y_pred)
t_test_original = target_scaler.inverse_transform(t_test)

# Visualize training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training History')
plt.show()

# Optional: Generate a simple function for deployment
def my_neural_network_function(new_x):
    new_x_scaled = input_scaler.transform(new_x)
    y_scaled = model.predict(new_x_scaled)
    return target_scaler.inverse_transform(y_scaled)
