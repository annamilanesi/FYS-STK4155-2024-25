import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import *
from NeuralNetwork import *

#leakyReLU FUNCTION

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

# Choose the cost function and its derivative
cost_fun = mse
cost_der = mse_der

# Define network structure and activation functions
network_input_size = inputs.shape[1]
#layer_output_sizes = [8, 1]
hidden_activation_func = leaky_ReLU
hidden_activation_der = leaky_ReLU_der
output_activation_func = sigmoid
output_activation_der = sigmoid_der

# Split the data into training and testing sets
X_train, X_test, z_train, z_test = train_test_split(inputs, targets, test_size=0.2)

# Define parameters for MSE analysis
hidden_layers = 1  # Number of hidden layers
n_nodes_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # List of node counts to test
n_epochs = 100
batch_size = 20 
learning_rate = 0.01
output_size = 1  # =1 for regression tasks, change to >1 for multi-classification tasks

# Call the mse_n_nodes function to evaluate MSE
mse_train, mse_test = mse_n_nodes(
    X_train, 
    X_test, 
    z_train, 
    z_test, 
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
n_nodes = 40  # Number of nodes in each hidden layer

# Call the mse_n_hidden_layers function 
mse_train, mse_test = mse_n_hidden_layers(
    X_train, 
    X_test, 
    z_train, 
    z_test, 
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
layer_output_sizes = [30, 40, 1] 
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

# Define parameters for MSE search
n_epochs_list = [10, 50, 100, 200, 500, 1000]
batch_sizes = [8, 16, 32, 64, 128, 256]
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 1]
lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# Call the functions to evaluate MSE
mse_train, mse_test = mse_epochs_vs_batchsize(
    X_train, X_test, z_train, z_test, n_epochs_list, batch_sizes, FFNN, learning_rate=0.01
)

mse_train, mse_test = mse_learning_rate_vs_lambda(
    X_train, X_test, z_train, z_test, learning_rates, lambdas, n_epochs=100, neural_network_model=FFNN, batch_size=16
)

mse_train, mse_test = mse_batchsize_vs_learning_rate(
    X_train, X_test, z_train, z_test, batch_sizes, learning_rates, n_epochs=100, neural_network_model=FFNN
)