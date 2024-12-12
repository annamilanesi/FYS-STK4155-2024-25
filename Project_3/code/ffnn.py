import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ------------------- LOAD DATA -------------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))

x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# ------------------- DATA PREPROCESSING -------------------

# Flatten the images to a 1D array for Feed-Forward Neural Network (FFNN)
x_train = x_train.reshape(x_train.shape[0], -1)  # -1 ensures the shape adapts automatically
x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten all images into a vector

# ------------------- BUILD THE MODEL -------------------

# Function to create a Feed-Forward Neural Network (FFNN) model
def create_ffnn_model(input_shape, n_nodes, n_layers):
    model = Sequential()
    
    # Add the first layer (Flattening is already done on the data)
    model.add(Dense(n_nodes, activation='relu', input_shape=(input_shape,)))
    
    # Add hidden layers
    for _ in range(n_layers - 1):  # Since the first layer is already added
        model.add(Dense(n_nodes, activation='relu'))
    
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Fixed parameters
batch_size = 16
epochs = 20

# ------------------- ANALYSIS WITH VARYING NUMBER OF NODES -------------------

# Plotting accuracy vs number of nodes in the hidden layer
nodes_range = [3, 5, 7, 10, 15, 20, 30]
train_accuracies = []
test_accuracies = []

for n_nodes in nodes_range:
    print(f"Training FFNN with {n_nodes} nodes")
    
    # Create the model
    model = create_ffnn_model(input_shape=x_train.shape[1], n_nodes=n_nodes, n_layers=1)
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Save the training accuracy
    train_accuracies.append(history.history['accuracy'][-1])
    
    # Evaluate the accuracy on the test data
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_accuracies.append(test_accuracy)

# Plotting the accuracy vs number of nodes
plt.figure(figsize=(10, 6))
plt.plot(nodes_range, train_accuracies, label='Train Accuracy', marker='o', linestyle='-')
plt.plot(nodes_range, test_accuracies, label='Test Accuracy', marker='o', linestyle='--')
plt.title('Accuracy vs Number of Nodes in the Hidden Layer')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------- ANALYSIS WITH VARYING NUMBER OF HIDDEN LAYERS -------------------

# Plotting accuracy vs number of hidden layers
layers_range = [1, 2, 3, 4, 5]
train_accuracies = []
test_accuracies = []

for n_layers in layers_range:
    print(f"Training FFNN with {n_layers} hidden layers")
    
    # Create the model
    model = create_ffnn_model(input_shape=x_train.shape[1], n_nodes=10, n_layers=n_layers)
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Save the training accuracy
    train_accuracies.append(history.history['accuracy'][-1])
    
    # Evaluate the accuracy on the test data
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_accuracies.append(test_accuracy)

# Plotting the accuracy vs number of hidden layers
plt.figure(figsize=(10, 6))
plt.plot(layers_range, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='deepskyblue')
plt.plot(layers_range, test_accuracies, label='Test Accuracy', marker='o', linestyle='--', color='orange')
plt.title('Accuracy vs Number of Hidden Layers')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
