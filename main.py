import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import os
import sys

# Suppress TensorFlow logs to prevent unwanted messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set default encoding to UTF-8 for console output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Load the dataset
df = pd.read_excel("export.xlsx")


# Drop columns with all NaN values
df = df.drop(columns=['snow', 'wpgt', 'tsun'])


# Assuming the 'duration' column exists for trip duration prediction (replace with actual if different)
# Here we will simulate a duration column for illustration purposes
df['duration'] = np.random.randint(10, 60, size=len(df))  # Random trip duration for this example

# Preprocess the data
features = df.drop(columns=["time", "duration"])  # Drop time and label column
labels = df["duration"]






# Scale the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the data into training and validation sets (80%/20% split)
X_train, X_val, y_train, y_val = train_test_split(scaled_features, labels, test_size=0.2, shuffle=False)

# Define the MLP Model (Multi-Layer Perceptron)
mlp_model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Define input shape with an Input layer
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
mlp_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the MLP Model
mlp_history = mlp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Define the Linear Regression Model (No hidden layers)
linear_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(1)
])

linear_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the Linear Regression Model
linear_history = linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Define the DNN Model (Deep Neural Network with 2 hidden layers)
dnn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

dnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the DNN Model
dnn_history = dnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Plot training and validation loss for all models
plt.plot(mlp_history.history['loss'], label='MLP Train Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')

plt.plot(linear_history.history['loss'], label='Linear Train Loss')
plt.plot(linear_history.history['val_loss'], label='Linear Validation Loss')

plt.plot(dnn_history.history['loss'], label='DNN Train Loss')
plt.plot(dnn_history.history['val_loss'], label='DNN Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the models and compare
mlp_loss, mlp_mae = mlp_model.evaluate(X_val, y_val)
linear_loss, linear_mae = linear_model.evaluate(X_val, y_val)
dnn_loss, dnn_mae = dnn_model.evaluate(X_val, y_val)

# Print the evaluation results (use UTF-8 encoding for printing)
print(f"MLP Model Loss: {mlp_loss}, MLP Model MAE: {mlp_mae}".encode('utf-8'))
print(f"Linear Model Loss: {linear_loss}, Linear Model MAE: {linear_mae}".encode('utf-8'))
print(f"DNN Model Loss: {dnn_loss}, DNN Model MAE: {dnn_mae}".encode('utf-8'))

# Make predictions using the best model (example here: MLP)
predictions = mlp_model.predict(X_val)

# Optionally, compare predictions with actual values
plt.scatter(y_val, predictions)
plt.xlabel('Actual Trip Duration')
plt.ylabel('Predicted Trip Duration')
plt.title('Actual vs Predicted Trip Duration')
plt.show()
