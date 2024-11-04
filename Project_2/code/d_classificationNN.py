import autograd.numpy as np
from autograd import grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from NeuralNetwork import *
from functions import *
import pandas as pd

cancer = load_breast_cancer()

# Download the data for inputs and targets
inputs = cancer.data
targets = cancer.target
targets = targets.reshape(targets.shape[0], 1) # Reshape the targets into a 1D array

# Split the datas into train and test
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)

# Scale datas
scaler = StandardScaler(with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Choose the cost function and its derivative
cost_fun = cross_entropy
cost_der = cross_entropy_der

network_input_size = X_train.shape[1]
output_size = 1

hidden_activation_func = sigmoid
hidden_activation_der = sigmoid_der
output_activation_func = sigmoid 
output_activation_der = sigmoid_der

# Define parameters for accuracy analysis
hidden_layers = 1  # Number of hidden layers
n_nodes_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]  # List of node counts to test
n_epochs = 100
batch_size = 32 
learning_rate = 0.05
output_size = 1  # =1 for regression tasks, change to >1 for classification tasks

# Call the mse_n_nodes function to evaluate MSE
accuracy_train, accuracy_test = accuracy_n_nodes(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    hidden_layers=hidden_layers, 
    n_nodes_list=n_nodes_list, 
    n_epochs=n_epochs, 
    batch_size=batch_size, 
    learning_rate=learning_rate, 
    output_size=output_size,
    hidden_activation_func=hidden_activation_func,
    hidden_activation_der=hidden_activation_der,
    output_activation_func=output_activation_func,
    output_activation_der=output_activation_der,
    cost_fun=cost_fun,
    cost_der=cost_der
)

n_hidden_layers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Different numbers of hidden layers to evaluate
n_nodes = 15  # Number of nodes in each hidden layer

# Call the mse_n_hidden_layers function 
accuracy_train, accuracy_test = accuracy_n_hidden_layers(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    n_nodes=n_nodes, 
    n_hidden_layers_list=n_hidden_layers_list, 
    n_epochs=n_epochs, 
    batch_size=batch_size, 
    learning_rate=learning_rate, 
    output_size=output_size,
    hidden_activation_func=hidden_activation_func,
    hidden_activation_der=hidden_activation_der,
    output_activation_func=output_activation_func,
    output_activation_der=output_activation_der,
    cost_fun=cost_fun,
    cost_der=cost_der
)

# Create initial layers for the neural network
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

n_epochs_list = [10, 50, 100, 200, 500, 1000]
batch_sizes = [8, 16, 32, 64, 128, 256]
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 1]
lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Calcolare l'accuratezza rispetto a epoche e dimensione del batch
acc_train, acc_test = accuracy_epochs_vs_batchsize(
    X_train, X_test, y_train, y_test, n_epochs_list, batch_sizes, FFNN, learning_rate=0.01
)

# Calcolare l'accuratezza rispetto al tasso di apprendimento e al valore di lambda
acc_train, acc_test = accuracy_learning_rate_vs_lambda(
    X_train, X_test, y_train, y_test, learning_rates, lambdas, n_epochs=100, neural_network_model=FFNN, batch_size=64
)

# Calcolare l'accuratezza rispetto alla dimensione del batch e al tasso di apprendimento
acc_train, acc_test = accuracy_batchsize_vs_learning_rate(
    X_train, X_test, y_train, y_test, batch_sizes, learning_rates, n_epochs=100, neural_network_model=FFNN
)
