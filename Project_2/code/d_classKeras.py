import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from functions import *
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def accuracy_epochs_vs_batchsize(model, X_train, y_train, X_test, y_test, n_epochs_list, minibatch_sizes, learning_rate=0.01, momentum=0.0):
    # Initialize matrices for accuracy values
    accuracy_train = np.zeros((len(n_epochs_list), len(minibatch_sizes)))
    accuracy_test = np.zeros((len(n_epochs_list), len(minibatch_sizes)))

    # Save initial weights to reset the model for each configuration
    initial_weights = model.get_weights()

    for i, n_epochs in enumerate(n_epochs_list):
        for j, batch_size in enumerate(minibatch_sizes):
            # Reset model weights
            model.set_weights(initial_weights)
            
            # Compile the model with SGD optimizer and binary cross-entropy loss
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            
            # Train the model with current epochs and batch size
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

            # Evaluate accuracy for both train and test sets
            train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
            test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

            accuracy_train[i, j] = train_accuracy
            accuracy_test[i, j] = test_accuracy

    # Plot heatmap for accuracy
    plot_heatmaps(accuracy_train, accuracy_test, minibatch_sizes, n_epochs_list, xlabel='Batch size', ylabel='Epochs', title='Accuracy Analysis')

    return accuracy_train, accuracy_test

def accuracy_learningrate_vs_lmd(model, X_train, y_train, X_test, y_test, learning_rate_values, lmd_values, n_epochs=100, batch_size=32, momentum=0.0):
    # Initialize matrices for accuracy values
    accuracy_train = np.zeros((len(learning_rate_values), len(lmd_values)))
    accuracy_test = np.zeros((len(learning_rate_values), len(lmd_values)))

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
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            # Train the model with the specified learning rate and lambda regularization
            model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

            # Evaluate accuracy for both train and test sets
            train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
            test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

            accuracy_train[i, j] = train_accuracy
            accuracy_test[i, j] = test_accuracy

    # Plot heatmap for accuracy with learning rate on the y-axis and lambda on the x-axis
    plot_heatmaps(accuracy_train, accuracy_test, x_labels=lmd_values, y_labels=learning_rate_values, xlabel='Lambda', ylabel='Learning Rate', title='Accuracy Analysis')

    return accuracy_train, accuracy_test

def plot_heatmaps(train_data, test_data, x_labels, y_labels, xlabel, ylabel, title):
    plt.figure(figsize=(14, 6))
    plt.suptitle(title, fontsize=20)

    # Train accuracy heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(train_data, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title(f'{title} (Train Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Test accuracy heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(test_data, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title(f'{title} (Test Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# NeuralNetwork using Keras/TensorFlow for classification

np.random.rand(2024)

cancer = load_breast_cancer()

# Download the data for inputs and targets
inputs = cancer.data
targets = cancer.target
targets = targets.reshape(targets.shape[0], 1) # Reshape the targets into a 1D array

# Split the data into training and testing sets
X_train, X_test, z_train, z_test = train_test_split(inputs, targets, test_size=0.2)

# Scale datas
scaler = StandardScaler(with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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

# Define parameters for MSE search
n_epochs_list = [10, 50, 100, 200, 500, 1000]
minibatch_sizes = [8, 16, 32, 64, 128, 256]
learning_rate_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
lmd_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Run accuracy analysis for different epochs and batch sizes using SGD
accuracy_train_epochs_batchsize, accuracy_test_epochs_batchsize = accuracy_epochs_vs_batchsize(
    model, X_train, z_train, X_test, z_test, n_epochs_list, minibatch_sizes, learning_rate=0.01, momentum=0.9
)

# Run accuracy analysis for different learning rates and lambda values using SGD
accuracy_train_lr_lmd, accuracy_test_lr_lmd = accuracy_learningrate_vs_lmd(
    model, X_train, z_train, X_test, z_test, learning_rate_values, lmd_values, momentum=0.9, batch_size=16
)


