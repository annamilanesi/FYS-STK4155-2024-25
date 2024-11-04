import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import *

class NeuralNetwork:
    def __init__(self,
                 network_input_size,
                 layer_output_sizes,
                 hidden_activation,
                 hidden_activation_der,
                 output_activation,
                 output_activation_der,
                 cost_fun,
                 cost_der,
                 lmd=0,
                 initial_layers=None):
        
        # Assign parameters to class attributes
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.hidden_activation = hidden_activation
        self.hidden_activation_der = hidden_activation_der
        self.output_activation = output_activation
        self.output_activation_der = output_activation_der
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lmd = lmd
        
        # Initialize layers (weights and biases)
        if initial_layers is None:
            self.layers = self.create_layers()
        else:
            self.layers = [(np.copy(W), np.copy(b)) for W, b in initial_layers]

    def create_layers(self):
        # Create layers as a list of (weights, bias) tuples
        layers = []
        input_size = self.network_input_size

        for output_size in self.layer_output_sizes:
            W = np.random.randn(input_size, output_size)  # Initialize weights
            b = np.random.randn(output_size)              # Initialize biases
            layers.append((W, b))
            input_size = output_size

        return layers

    def feed_forward(self, inputs):
        # Forward propagation with appropriate activation functions
        a = inputs
        for idx, (W, b) in enumerate(self.layers):
            z = np.dot(a, W) + b
            # Use output activation only for the last layer
            if idx == len(self.layers) - 1:
                a = self.output_activation(z)
            else:
                a = self.hidden_activation(z)
        return a

    def feed_forward_saver(self, inputs):
        # Save intermediate values for backpropagation during forward propagation
        layer_inputs = []
        zs = []
        a = inputs

        for idx, (W, b) in enumerate(self.layers):
            layer_inputs.append(a)
            z = np.dot(a, W) + b
            # Use output activation only for the last layer
            if idx == len(self.layers) - 1:
                a = self.output_activation(z)
            else:
                a = self.hidden_activation(z)
            zs.append(z)

        return layer_inputs, zs, a

    def backpropagation(self, inputs, target):
        # Backpropagation calculation with saved activations
        layer_inputs, zs, predict = self.feed_forward_saver(inputs)
        layer_grads = [() for _ in self.layers]

        for i in reversed(range(len(self.layers))):
            W, b = self.layers[i]
            layer_input = layer_inputs[i]
            z = zs[i]

            # Use output derivative only for the last layer
            if i == len(self.layers) - 1:
                dC_da = self.cost_der(predict, target)
                activation_der = self.output_activation_der
            else:
                # Backpropagate the gradient
                W_next, _ = self.layers[i + 1]
                dC_da = np.dot(dC_dz, W_next.T)
                activation_der = self.hidden_activation_der

            # Calculate gradients for weights and biases
            dC_dz = dC_da * activation_der(z)
            dC_dW = np.dot(layer_input.T, dC_dz) + 2 * self.lmd * W
            dC_db = np.sum(dC_dz, axis=0, keepdims=True)

            # Save gradients
            layer_grads[i] = (dC_dW, dC_db)
        
        return layer_grads

    def predict(self, inputs):
        # Prediction function based on the trained network
        return self.feed_forward(inputs)

    def reset_weights(self):
        # Reinitialize self.layers with new random values
        self.layers = self.create_layers()

# Functions outside of the class

def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)

def train_layers_SGD(inputs, targets, NNmodel, M=10, n_epochs=100, lmd=0, momentum=0.9, learning_rate=0.01):
    n_samples, n_features = inputs.shape  # Number of samples and features
    
    # Initialize velocity for weights and biases
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in NNmodel.layers]

    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        random_index = np.random.permutation(n_samples)
        inputs_shuffled = inputs[random_index]
        targets_shuffled = targets[random_index]

        # Mini-batch gradient descent
        for i in range(0, n_samples, M):
            inputs_i = inputs_shuffled[i:i + M]
            targets_i = targets_shuffled[i:i + M]

            # Compute gradients using backpropagation in NNmodel
            layers_grads = NNmodel.backpropagation(inputs_i, targets_i)
            # Update weights and biases with momentum
            new_layers = []
            new_v = []
            for (W, b), (W_grad, b_grad), (v_W, v_b) in zip(NNmodel.layers, layers_grads, v):
                # Update velocities for weights and biases
                v_W_new = momentum * v_W + learning_rate * W_grad
                v_b_new = momentum * v_b + learning_rate * b_grad

                # Update weights and biases using velocities
                W_new = W - v_W_new
                b_new = b - v_b_new

                # Save updated weights, biases, and velocities
                new_layers.append((W_new, b_new))
                new_v.append((v_W_new, v_b_new))

            # Update NNmodel layers and velocities
            NNmodel.layers = new_layers
            v = new_v

    return NNmodel.layers

def create_layers(network_input_size, layer_output_sizes):
    # Create layers as a list of (weights, biases)
    layers = []
    i_size = network_input_size

    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

# Function to visualize heatmaps
def plot_heatmaps(mse_train, mse_test, x_labels, y_labels, method, xlabel='X-axis', ylabel='Y-axis'):
    # Create heatmap for visualization
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'MSE Analysis using {method} Method', fontsize=20)

    # Heatmap for training MSE
    plt.subplot(1, 2, 1)
    sns.heatmap(mse_train, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Train Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Heatmap for test MSE
    plt.subplot(1, 2, 2)
    sns.heatmap(mse_test, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Test Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title and text
    plt.show()

def mse_epochs_vs_batchsize(X_train, X_test, z_train, z_test, n_epochs_list, batch_sizes, neural_network_model, learning_rate=0.01):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(n_epochs_list), len(batch_sizes)))
    mse_test = np.zeros((len(n_epochs_list), len(batch_sizes)))

    for i, epoch in enumerate(n_epochs_list):
        for j, batch_size in enumerate(batch_sizes):
            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=epoch, learning_rate=learning_rate)
            
            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, batch_sizes, n_epochs_list, method="Neural Network", xlabel='Batch Size', ylabel='Epochs')

    return mse_train, mse_test

def mse_learning_rate_vs_lambda(X_train, X_test, z_train, z_test, learning_rates, lambdas, n_epochs, batch_size, neural_network_model):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(learning_rates), len(lambdas)))
    mse_test = np.zeros((len(learning_rates), len(lambdas)))

    for i, lr in enumerate(learning_rates):
        for j, l in enumerate(lambdas):
            # Initialize the neural network with the current learning rate and lambda
            neural_network_model.lmd = l

            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=n_epochs)

            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, lambdas, learning_rates, method="Neural Network", xlabel='Lambda', ylabel='Learning Rate')

    return mse_train, mse_test

def mse_batchsize_vs_learning_rate(X_train, X_test, z_train, z_test, batch_sizes, learning_rates, n_epochs, neural_network_model):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(batch_sizes), len(learning_rates)))
    mse_test = np.zeros((len(batch_sizes), len(learning_rates)))

    for i, batch_size in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=n_epochs, learning_rate=lr)

            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, batch_sizes, learning_rates, method="Neural Network", xlabel='Batch Size', ylabel='Learning Rate')

    return mse_train, mse_test

# Function that fix a number of hidden layers and varies the number of nodes
def mse_n_nodes(X_train, X_test, z_train, z_test, hidden_layers, n_nodes_list, 
                n_epochs=100, batch_size=10, learning_rate=0.01, output_size=1, 
                hidden_activation_func=sigmoid, hidden_activation_der=sigmoid_der, 
                output_activation_func=sigmoid, output_activation_der=sigmoid_der,
                cost_fun=mse, cost_der=mse_der):
    
    # Initialize arrays to store MSE values
    mse_train = []
    mse_test = []

    for n_nodes in n_nodes_list:
        # Define the layer output sizes
        layer_output_sizes = [n_nodes] * hidden_layers + [output_size]
        
        # Create initial layers for the neural network
        initial_layers = create_layers(X_train.shape[1], layer_output_sizes)
        
        # Define the neural network
        FFNN = NeuralNetwork(
            network_input_size=X_train.shape[1],
            layer_output_sizes=layer_output_sizes,
            hidden_activation=hidden_activation_func,
            hidden_activation_der=hidden_activation_der,
            output_activation=output_activation_func,
            output_activation_der=output_activation_der,
            cost_fun=cost_fun,
            cost_der=cost_der,
            initial_layers=initial_layers
        )
        
        # Train the neural network
        train_layers_SGD(X_train, z_train, FFNN, M=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

        # Predict on training and testing data
        z_train_pred = FFNN.predict(X_train)
        z_test_pred = FFNN.predict(X_test)

        # Calculate MSE for training and testing data
        mse_train.append(mse(z_train, z_train_pred))
        mse_test.append(mse(z_test, z_test_pred))

        # Reset weights for the next iteration
        FFNN.reset_weights()

    # Plot the MSE results
    plt.figure(figsize=(10, 6))
    plt.plot(n_nodes_list, mse_train, label='Train MSE', marker='o')
    plt.plot(n_nodes_list, mse_test, label='Test MSE', marker='x')
    plt.title('MSE vs Number of Nodes in Hidden Layers')
    plt.xlabel('Number of Nodes in Hidden Layers')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid()
    plt.show()

    return np.array(mse_train), np.array(mse_test)

import matplotlib.pyplot as plt
import numpy as np

def mse_n_hidden_layers(X_train, X_test, z_train, z_test, n_nodes, n_hidden_layers_list,
                         n_epochs=100, batch_size=10, learning_rate=0.01, output_size=1, 
                         hidden_activation_func=sigmoid, hidden_activation_der=sigmoid_der, 
                         output_activation_func=sigmoid, output_activation_der=sigmoid_der, 
                         cost_fun=mse, cost_der=mse_der):

    mse_train = []
    mse_test = []

    for n_hidden_layers in n_hidden_layers_list:
        # Create the layer structure
        layer_output_sizes = [n_nodes] * n_hidden_layers + [output_size]  # List of layer sizes

        # Initialize the neural network
        initial_layers = create_layers(X_train.shape[1], layer_output_sizes)
        neural_network_model = NeuralNetwork(
            network_input_size=X_train.shape[1],
            layer_output_sizes=layer_output_sizes,
            hidden_activation=hidden_activation_func,
            hidden_activation_der=hidden_activation_der,
            output_activation=output_activation_func,
            output_activation_der=output_activation_der,
            cost_fun=cost_fun,
            cost_der=cost_der,
            initial_layers=initial_layers
        )

        # Train the neural network and evaluate MSE
        trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=n_epochs, learning_rate=learning_rate)

        # Predict on training and testing data
        z_tilde = neural_network_model.predict(X_train)
        z_pred = neural_network_model.predict(X_test)

        # Calculate MSE for training and testing data
        mse_train.append(cost_fun(z_train, z_tilde))  # Use cost function to calculate MSE
        mse_test.append(cost_fun(z_test, z_pred))      # Use cost function to calculate MSE

        # Reset the neural network's weights for the next iteration
        neural_network_model.reset_weights()

    # Plotting the MSE results
    plt.figure(figsize=(10, 6))
    plt.plot(n_hidden_layers_list, mse_train, label='Training MSE', marker='o')
    plt.plot(n_hidden_layers_list, mse_test, label='Testing MSE', marker='o')
    plt.title('MSE vs Number of Hidden Layers')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(n_hidden_layers_list)
    plt.legend()
    plt.grid()
    plt.show()

    return np.array(mse_train), np.array(mse_test)









"""import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import *

np.random.rand(2024)

class NeuralNetwork: 

    # Class initialization
    def __init__(self,
                 network_input_size,
                 layer_output_sizes,
                 activation_funcs,
                 activation_ders,
                 cost_fun,
                 cost_der,
                 lmd=0,
                 initial_layers=None):
        
        # Assign parameters to class attributes
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lmd = lmd
        
        # Initialize layers (weights and biases)
        if initial_layers is None:
            self.layers = self.create_layers()
        else:
            self.layers = [(np.copy(W), np.copy(b)) for W, b in initial_layers]

    def create_layers(self):
        # Create layers as a list of (weights, biases)
        layers = []
        input_size = self.network_input_size

        for output_size in self.layer_output_sizes:
            W = np.random.randn(input_size, output_size)  # Initialize weights
            b = np.random.randn(output_size)              # Initialize biases
            layers.append((W, b))
            input_size = output_size

        return layers

    def feed_forward(self, inputs):
        # Feed forward through the network with stored activation functions
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation_func(z)
        return a

    def feed_forward_saver(self, inputs):
        # Save intermediate values (for backpropagation) during feed forward
        layer_inputs = []
        zs = []
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = np.dot(a, W) + b
            a = activation_func(z)
            zs.append(z)

        return layer_inputs, zs, a

    def backpropagation(self, inputs, target):

        # Backpropagation to calculate gradients with saved layer activations
        layer_inputs, zs, predict = self.feed_forward_saver(inputs)

        layer_grads = [() for layer in self.layers]

        for i in reversed(range(len(self.layers))):
            
            W, b = self.layers[i]  # Get weights and biases for the current layer

            layer_input = layer_inputs[i]
            z = zs[i]
            activation_der = self.activation_ders[i]

            if i == len(self.layers) - 1:
                # For the last layer, use the cost derivative directly
                dC_da = self.cost_der(predict, target)
            else:
                # Propagate gradient backwards
                W_next, _ = self.layers[i + 1]
                dC_da = np.dot(dC_dz, W_next.T)

            # Compute gradients for weights and biases
            dC_dz = dC_da * activation_der(z)
            dC_dW = np.dot(layer_input.T, dC_dz) + 2 * self.lmd * W
            dC_db = np.sum(dC_dz, axis=0, keepdims=True)

            # Store the gradients
            layer_grads[i] = (dC_dW, dC_db)
        
        return layer_grads

    def predict(self, inputs):
        # Function to predict based on trained network
        return self.feed_forward(inputs)

    def reset_weights(self):
        # Reinizializza self.layers con nuovi valori casuali
        self.layers = self.create_layers()


#outside of the class define the training method
def learning_schedule(t, t0=5 , t1=50):
    return t0/(t+t1)

def train_layers_SGD(inputs, targets, NNmodel, M=10, n_epochs=100, lmd=0, momentum=0.9, learning_rate=0.01):
    
    n_samples, n_features = inputs.shape  # Number of samples and features
    
    # Initialize velocity for weights and biases
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in NNmodel.layers]

    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        random_index = np.random.permutation(n_samples)
        inputs_shuffled = inputs[random_index]
        targets_shuffled = targets[random_index]

        # Mini-batch gradient descent
        for i in range(0, n_samples, M):
            inputs_i = inputs_shuffled[i:i + M]
            targets_i = targets_shuffled[i:i + M]

            """ """# Adjust learning rate with schedule if necessary
            adjusted_learning_rate = learning_schedule(epoch * (n_samples // M) + i)""" """

            # Compute gradients using backpropagation in NNmodel
            layers_grads = NNmodel.backpropagation(inputs_i, targets_i)
            # Update weights and biases with momentum
            new_layers = []
            new_v = []
            for (W, b), (W_grad, b_grad), (v_W, v_b) in zip(NNmodel.layers, layers_grads, v):
                # Update velocities for weights and biases
                v_W_new = momentum * v_W + learning_rate * W_grad
                v_b_new = momentum * v_b + learning_rate * b_grad

                # Update weights and biases using velocities
                W_new = W - v_W_new
                b_new = b - v_b_new

                # Save updated weights, biases, and velocities
                new_layers.append((W_new, b_new))
                new_v.append((v_W_new, v_b_new))

            # Update NNmodel layers and velocities
            NNmodel.layers = new_layers
            v = new_v

    return NNmodel.layers

def create_layers(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size

    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

#print and mse function

# Funzione per visualizzare le heatmap
def plot_heatmaps(mse_train, mse_test, x_labels, y_labels, method, xlabel='X-axis', ylabel='Y-axis'):
    # Crea heatmap per la visualizzazione
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'MSE Analysis using {method} Method', fontsize=20)

    # Heatmap per MSE di addestramento
    plt.subplot(1, 2, 1)
    sns.heatmap(mse_train, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Train Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Heatmap per MSE di test
    plt.subplot(1, 2, 2)
    sns.heatmap(mse_test, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
    plt.title('MSE Heatmap (Test Set)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Regola layout per titolo e testo
    plt.show()

""" """
def mse_polydegree(x, y, z, maxdegree, neural_network_model, M=10, n_epochs=100, lmd=0, momentum=0.9, learning_rate=0.01):
    # Inizialize vectors for MSE values
    mse_train = np.zeros(maxdegree + 1)
    mse_test = np.zeros(maxdegree + 1)

    # Dividi il dataset in train e test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    # Loop over polynomial degrees
    for degree in range(maxdegree + 1):
        # Create design matrices for train and test
        X_train = np.column_stack((x, y))
        X_test = np.column_stack((x, y))

        # Train the neural network
        neural_network_model.initial_layers = None
        trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=M, n_epochs=n_epochs, lmd=lmd, momentum=momentum, learning_rate=learning_rate )

        # Predict on training and testing data
        z_tilde = neural_network_model.predict(X_train)
        z_pred = neural_network_model.predict(X_test)

        # Calculate MSE for training and testing data
        mse_train[degree] = mse(z_train, z_tilde)
        mse_test[degree] = mse(z_test, z_pred)

    # Crea il grafico dell'andamento dell'MSE
    plt.figure(figsize=(10, 6))
    plt.plot(range(maxdegree + 1), mse_train, label='MSE Train', marker='o')
    plt.plot(range(maxdegree + 1), mse_test, label='MSE Test', marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE for Train and Test sets by Polynomial Degree using NeuralNetwork')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mse_train, mse_test
""" """
        
def mse_epochs_vs_batchsize(X_train, X_test, z_train, z_test, n_epochs_list, batch_sizes, neural_network_model, learning_rate=0.01):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(n_epochs_list), len(batch_sizes)))
    mse_test = np.zeros((len(n_epochs_list), len(batch_sizes)))

    for i, epoch in enumerate(n_epochs_list):
        for j, batch_size in enumerate(batch_sizes):

            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=epoch, learning_rate=learning_rate)
            
            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, batch_sizes, n_epochs_list, method="Neural Network", xlabel='Batch Size', ylabel='Epochs')

    return mse_train, mse_test

def mse_learning_rate_vs_lambda(X_train, X_test, z_train, z_test, learning_rates, lambdas, n_epochs, batch_size, neural_network_model):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(learning_rates), len(lambdas)))
    mse_test = np.zeros((len(learning_rates), len(lambdas)))

    for i, lr in enumerate(learning_rates):
        for j, l in enumerate(lambdas):
            # Initialize the neural network with the current learning rate and lambda
            neural_network_model.lmd = l

            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=n_epochs)

            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, lambdas, learning_rates, method="Neural Network", xlabel='Lambda', ylabel='Learning Rate')

    return mse_train, mse_test

def mse_batchsize_vs_learning_rate(X_train, X_test, z_train, z_test, batch_sizes, learning_rates, n_epochs, neural_network_model):
    # Initialize matrices for MSE values
    mse_train = np.zeros((len(batch_sizes), len(learning_rates)))
    mse_test = np.zeros((len(batch_sizes), len(learning_rates)))

    for i, batch_size in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):

            # Train the neural network
            trained_layers = train_layers_SGD(X_train, z_train, neural_network_model, M=batch_size, n_epochs=n_epochs, learning_rate=lr)

            # Predict on training and testing data
            z_tilde = neural_network_model.predict(X_train)
            z_pred = neural_network_model.predict(X_test)

            # Calculate MSE for training and testing data
            mse_train[i, j] = mse(z_train, z_tilde)
            mse_test[i, j] = mse(z_test, z_pred)

            neural_network_model.reset_weights()

    plot_heatmaps(mse_train, mse_test, batch_sizes, learning_rates, method="Neural Network", xlabel='Batch Size', ylabel='Learning Rate')

    return mse_train, mse_test """
