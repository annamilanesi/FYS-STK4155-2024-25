import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from functions import *

def mse_epochs_vs_batchsize(model, X_train, y_train, X_test, y_test, n_epochs_list, minibatch_sizes, learning_rate=0.01, momentum=0.0):
    mse_train = np.zeros((len(n_epochs_list), len(minibatch_sizes)))
    mse_test = np.zeros((len(n_epochs_list), len(minibatch_sizes)))

    # Save initial weights to reset the model for each configuration
    initial_weights = model.get_weights()

    for i, n_epochs in enumerate(n_epochs_list):
        for j, batch_size in enumerate(minibatch_sizes):
            # Reset model weights
            model.set_weights(initial_weights)
            
            # Compile the model with SGD optimizer
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                          loss='mse',
                          metrics=['mse'])
            
            # Train the model with current epochs and batch size
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

            # Evaluate MSE for both train and test sets
            train_mse = model.evaluate(X_train, y_train, verbose=0)[1]
            test_mse = model.evaluate(X_test, y_test, verbose=0)[1]

            mse_train[i, j] = train_mse
            mse_test[i, j] = test_mse

    # Plot heatmap for MSE
    plot_heatmaps(mse_train, mse_test, minibatch_sizes, n_epochs_list, xlabel='Batch size', ylabel='Epochs')

    return mse_train, mse_test


    # Plot heatmap for MSE
    plot_heatmaps(mse_train, mse_test, minibatch_sizes, n_epochs_list, xlabel='Batch size', ylabel='Epochs')

    return mse_train, mse_test

def mse_learningrate_vs_lmd(model, X_train, y_train, X_test, y_test, learning_rate_values, lmd_values, n_epochs=100, batch_size=32, momentum=0.0):
    # Initialize matrices for MSE values with learning rates on the rows and lambda values on the columns
    mse_train = np.zeros((len(learning_rate_values), len(lmd_values)))
    mse_test = np.zeros((len(learning_rate_values), len(lmd_values)))

    # Save initial weights to reset the model for each configuration
    initial_weights = model.get_weights()

    for i, learning_rate in enumerate(learning_rate_values):
        for j, lmd in enumerate(lmd_values):
            # Reset model weights
            model.set_weights(initial_weights)

            # Adjust the regularization term for each layer that has it
            for layer in model.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(lmd)

            # Compile model with updated SGD optimizer and regularization
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                          loss='mse',
                          metrics=['mse'])

            # Train the model with the specified learning rate and lambda regularization
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

            # Evaluate MSE for both train and test sets
            train_mse = model.evaluate(X_train, y_train, verbose=0)[1]
            test_mse = model.evaluate(X_test, y_test, verbose=0)[1]

            mse_train[i, j] = train_mse
            mse_test[i, j] = test_mse

    # Plot heatmap for MSE with learning rate on the y-axis and lambda on the x-axis
    plot_heatmaps(mse_train, mse_test, x_labels=lmd_values, y_labels=learning_rate_values, xlabel='Lambda', ylabel='Learning Rate')

    return mse_train, mse_test

def plot_heatmaps(mse_train, mse_test, x_labels, y_labels, xlabel, ylabel):
    plt.figure(figsize=(14, 6))
    plt.suptitle('MSE Analysis', fontsize=20)

    # Train MSE heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(mse_train, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Train Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Test MSE heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(mse_test, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Test Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
   

# NeuralNetwork using Keras/TensorFlow

np.random.rand(2024)

# Create the dataset
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 0.1)

# Flatten the data into arrays
x = x.ravel()
y = y.ravel()
z = z.ravel()

# Create the input matrix with x and y as columns
inputs = np.column_stack((x, y))

# Define the targets
targets = z.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, z_train, z_test = train_test_split(inputs, targets, test_size=0.2)

# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Add Input layer
    model.add(Dense(30, activation=LeakyReLU(negative_slope=0.01)))  # First Dense layer
    model.add(Dense(40, activation=LeakyReLU(negative_slope=0.01)))  # Second Dense layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression
    return model

# Create and build the model
input_shape = (X_train.shape[1],)  # Number of features in your input data
model = build_model(input_shape)

# Define parameters for MSE search
n_epochs_list = [10, 50, 100, 200, 500, 1000]
minibatch_sizes = [8, 16, 32, 64, 128, 256]
learning_rate_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
lmd_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
"""
# Run MSE analysis for different epochs and batch sizes using SGD
mse_train_epochs_batchsize, mse_test_epochs_batchsize = mse_epochs_vs_batchsize(
    model, X_train, z_train, X_test, z_test, n_epochs_list, minibatch_sizes, learning_rate=0.01, momentum=0.9
)
"""
# Run MSE analysis for different learning rates and lambda values using SGD
mse_train_lr_lmd, mse_test_lr_lmd = mse_learningrate_vs_lmd(
    model, X_train, z_train, X_test, z_test, learning_rate_values, lmd_values, momentum=0.9, batch_size=16
)
