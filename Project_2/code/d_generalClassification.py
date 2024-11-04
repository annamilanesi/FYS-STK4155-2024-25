import autograd.numpy as np
from autograd import grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from NeuralNetwork import *
from functions import *
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, LeakyReLU

cancer = load_breast_cancer()

# Download the data for inputs and targets
inputs = cancer.data
targets = cancer.target
targets = targets.reshape(targets.shape[0], 1) # Reshape the targets into a 1D array

# Print the correltion matrix
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
correlation_matrix = cancerpd.corr().round(1)

# Set up the figure with a larger size and tight layout
plt.figure(figsize=(14, 12))
sns.heatmap(data=correlation_matrix, annot=True, cmap="viridis", annot_kws={"size": 8}, fmt=".1f",
            square=True, cbar_kws={"shrink": .8})  # Adjust font size and color bar size

# Display the plot
plt.tight_layout()  # Ensure everything fits within the figure
plt.show()

#CONFUSION MATRIX

# Split the datas into train and test
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)

# Scale datas
scaler = StandardScaler(with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create the NN model
cost_fun = cross_entropy
cost_der = cross_entropy_der

network_input_size = X_train.shape[1]
output_size = 1

hidden_activation_func = sigmoid # Choose activation function and explain why
hidden_activation_der = sigmoid_der
output_activation_func = sigmoid 
output_activation_der = sigmoid_der

n_epochs = 100
batch_size = 32 
learning_rate = 0.05

layer_output_sizes = [40, 40, 1]
initial_layers = create_layers(network_input_size, layer_output_sizes)

# Define the neural network
FFNN = NeuralNetwork(
    network_input_size,
    layer_output_sizes,
    hidden_activation_func,
    hidden_activation_der,
    output_activation_func,
    output_activation_der,
    cost_fun=cost_fun,
    cost_der=cost_der,
    initial_layers=initial_layers
)

# Make predictions
trained_layers = train_layers_SGD(X_train, y_train, FFNN, M=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

y_tilde = FFNN.predict(X_train)
y_pred = FFNN.predict(X_test)

# Apply a threshold to convert continuous predictions to binary (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, 
            xticklabels=["Predicted Negative", "Predicted Positive"], 
            yticklabels=["Actual Negative", "Actual Positive"],
            annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix-Our FFNN")
plt.show()

#NEURAL NETWORK WITH TENSOR

# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Add Input layer
    model.add(Dense(40, activation='sigmoid'))  # First Dense layer
    model.add(Dense(40, activation='sigmoid'))  # Second Dense layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression
    return model

# Create and build the model
input_shape = (X_train.shape[1],)  # Number of features in your input data
model = build_model(input_shape)

# Compile the model
model.compile(optimizer='sgd',  # Using SGD optimizer
              loss='binary_crossentropy',  # Cross entropy for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0) 

# Make predictions on the test set
y_pred_proba = model.predict(X_test)  # Get probabilities
y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, 
            xticklabels=["Predicted Negative", "Predicted Positive"], 
            yticklabels=["Actual Negative", "Actual Positive"],
            annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix-Keras FFNN")
plt.show()

#LOGISTIC REGRESSION BY SCIKIT-LEARN

# Logistic Regression model setup and training
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train.ravel())

# Logistic Regression predictions
y_pred_logreg = log_reg.predict(X_test)

# Confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

# Plot the confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="viridis", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"], 
            yticklabels=["Actual Negative", "Actual Positive"],
            annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Scikit Logistic Regression")
plt.show()

from LogisticRegression import *

# MY LOGISTIC REGRESSION METHOD

learning_rate = 0.05
momentum = 0.9 
lmd = 0
n_epochs = 100
batch_size = 32

beta_initial = np.random.rand(X_train.shape[1], 1)

LogReg = LogisticRegression(gradient_mode='autograd', momentum=momentum, learning_rate=learning_rate,
                            lmd=lmd, n_epochs=n_epochs, batch_size=batch_size, beta_initial= beta_initial)

beta, _ = LogReg.beta_SGD_history(X_train, y_train)

y_pred = LogReg.predict(X_test, beta)

# Confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="viridis", cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"], 
            yticklabels=["Actual Negative", "Actual Positive"],
            annot_kws={"size": 16})
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Our Logistic Regression")
plt.show()