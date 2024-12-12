import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import random
from sklearn.metrics import confusion_matrix

# Fissa il seed per la riproducibilità
np.random.seed(42)                  # Fissa il seed per NumPy
random.seed(42)                     # Fissa il seed per il modulo random

# Define directory where the data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the preprocessed training and test data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Reshape the data for FFNN (flatten image data into vectors)
x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten for FFNN
x_test = x_test.reshape(x_test.shape[0], -1)

# Function to create FFNN model
def create_ffnn(lambda_value=0.0):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))  # First hidden layer
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(lambda_value)))  # Second hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Heatmap 1: Accuracy vs epochs and batch_size ---

epochs_range = [5, 10, 15, 20]  # Values for epochs (x-axis)
batch_sizes = [8, 16, 32, 64]  # Values for batch_size (y-axis)
accuracy_epochs_batch_train = np.zeros((len(batch_sizes), len(epochs_range)))
accuracy_epochs_batch_test = np.zeros((len(batch_sizes), len(epochs_range)))

# Loop through each combination of epochs and batch_size
for i, batch_size in enumerate(batch_sizes):
    for j, epochs in enumerate(epochs_range):
        print(f"Calculating: Epochs={epochs}, Batch Size={batch_size}")
        model = create_ffnn()
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)  # Train
        train_accuracy = history.history['accuracy'][-1]  # Last training accuracy
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # Evaluate on test data
        accuracy_epochs_batch_train[i, j] = train_accuracy
        accuracy_epochs_batch_test[i, j] = test_accuracy

# Plot heatmaps for train and test accuracy
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training accuracy heatmap
sns.heatmap(accuracy_epochs_batch_train, annot=True, fmt=".2f", ax=ax[0], cmap="viridis",
            xticklabels=epochs_range, yticklabels=batch_sizes)
ax[0].set_title('Training Accuracy')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Batch Size')

# Test accuracy heatmap
sns.heatmap(accuracy_epochs_batch_test, annot=True, fmt=".2f", ax=ax[1], cmap="viridis",
            xticklabels=epochs_range, yticklabels=batch_sizes)
ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Batch Size')

plt.tight_layout()
plt.show()

# --- Heatmap 2: Accuracy vs lambda and batch_size ---

lambda_values = [0.1, 0.01, 0.001, 0.0001]  # Values for lambda (x-axis)
accuracy_lambda_batch_train = np.zeros((len(batch_sizes), len(lambda_values)))
accuracy_lambda_batch_test = np.zeros((len(batch_sizes), len(lambda_values)))

# Loop through each combination of lambda and batch_size
for i, batch_size in enumerate(batch_sizes):
    for j, lambda_value in enumerate(lambda_values):
        print(f"Calculating: Lambda={lambda_value}, Batch Size={batch_size}")
        model = create_ffnn(lambda_value=lambda_value)
        history = model.fit(x_train, y_train, epochs=15, batch_size=batch_size, verbose=0)  # Use fixed epochs
        train_accuracy = history.history['accuracy'][-1]  # Last training accuracy
        _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # Evaluate on test data
        accuracy_lambda_batch_train[i, j] = train_accuracy
        accuracy_lambda_batch_test[i, j] = test_accuracy

# Plot heatmaps for train and test accuracy
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training accuracy heatmap
sns.heatmap(accuracy_lambda_batch_train, annot=True, fmt=".2f", ax=ax[0], cmap="viridis",
            xticklabels=lambda_values, yticklabels=batch_sizes)
ax[0].set_title('Training Accuracy')
ax[0].set_xlabel('Lambda (Ridge Regularization)')
ax[0].set_ylabel('Batch Size')

# Test accuracy heatmap
sns.heatmap(accuracy_lambda_batch_test, annot=True, fmt=".2f", ax=ax[1], cmap="viridis",
            xticklabels=lambda_values, yticklabels=batch_sizes)
ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Lambda (Ridge Regularization)')
ax[1].set_ylabel('Batch Size')

plt.tight_layout()
plt.show()

#-------------- plot confusion matrix ---------------

# Fissa i parametri
fixed_epochs = 15
fixed_batch_size = 64
fixed_lambda = 0.1

# Creazione e allenamento del modello con i parametri fissati
print(f"Training model with fixed parameters: Epochs={fixed_epochs}, Batch Size={fixed_batch_size}, Lambda={fixed_lambda}")
model = create_ffnn(lambda_value=fixed_lambda)
model.fit(x_train, y_train, epochs=fixed_epochs, batch_size=fixed_batch_size, verbose=0)

# Predizioni
y_pred_probs = model.predict(x_test)  # Probabilità previste
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Converte in valori binari (0 o 1)

# Creazione della confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot della confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d',
    cbar=False, 
    cmap='viridis', 
    xticklabels=["Predicted Pneumonia", "Predicted Normal"], 
    yticklabels=["Actual Pneumonia", "Actual Normal"], 
    annot_kws={"size": 16}
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

