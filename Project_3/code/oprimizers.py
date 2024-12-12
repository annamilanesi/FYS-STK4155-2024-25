import numpy as np
import os
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.regularizers import l2

#------------------- OPTIMIZERS ---------------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))

x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Fixed parameters
n_filters = 8                  # Number of filters !!!!!!
kernel_size = (5, 5)           # Kernel size
pool_size = (3, 3)             # Pooling size
stride = 1                     # Stride
batch_size = 16                # Batch size
lmbda = 0.01                   # L2 regularization factor
n_nodes = 20                   # Number of nodes in dense layer
epochs = 20                    # Fixed number of epochs for analysis

# Optimizers to test
optimizers = {
    "SGD": SGD(),
    "Adam": Adam(),
    "RMSprop": RMSprop(),
    "Adagrad": Adagrad()
}

# Store test accuracy at each epoch for each optimizer
test_accuracies = {opt_name: [] for opt_name in optimizers.keys()}

# Loop over optimizers
for opt_name, optimizer in optimizers.items():
    print(f"Testing optimizer: {opt_name}")
    
    # Build the model
    model = Sequential()
    model.add(layers.Input(shape=(150, 150, 1)))
    model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(units=n_nodes, activation='relu', kernel_regularizer=l2(lmbda)))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model with the current optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model and store accuracy at each epoch
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(x_test, y_test))
    
    # Store test accuracy for each epoch
    test_accuracies[opt_name] = history.history['val_accuracy']

# Plot the test accuracies over epochs for each optimizer
plt.figure(figsize=(10, 6))
for opt_name, accuracies in test_accuracies.items():
    plt.plot(range(1, epochs + 1), accuracies, label=opt_name)

# Add labels, legend, and title
plt.title("Test Accuracy vs Epochs for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)
plt.show()
