import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, regularizers

#----------------- ANALYSIS OF FILTER SIZE WITH DIFFERENT STRIPES -------------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))

x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Define the model creation function
def create_model(kernel_size, stride, num_filters, lambda_val):
    model = Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(150, 150, 1)))
    
    # Convolutional Layer with kernel size and stride as parameters
    model.add(layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        activation='relu',
        strides=stride,
        padding='same',
    ))
    
    # Max Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # Dense Layer with L2 regularization
    model.add(layers.Dense(
        units=20, 
        activation='relu',
        kernel_regularizer=regularizers.l2(lambda_val)
    ))
    
    # Output Layer (Binary classification)
    model.add(layers.Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Parameters
kernel_sizes = [3, 5, 7]  # Sizes of the kernels to test
strides = [1, 2]  # Strides to test
num_filters = 32  # Fixed number of filters
lambda_val = 0.01  # Fixed regularization
epochs = 15  # Fixed number of epochs
batch_size = 16  # Fixed batch size

# Create dictionaries to store results
train_accuracies = {stride: [] for stride in strides}
test_accuracies = {stride: [] for stride in strides}

# Train and evaluate models for each kernel size and stride
for stride in strides:
    for kernel_size in kernel_sizes:
        print(f"Training model with kernel_size={kernel_size}, stride={stride}.")
        
        # Create the model
        model = create_model(kernel_size, stride, num_filters, lambda_val)
        
        # Train the model
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Get train and test accuracies
        train_accuracy = model.history.history['accuracy'][-1]  # Last epoch accuracy on training data
        test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]  # Accuracy on test data
        
        # Store the accuracies
        train_accuracies[stride].append(train_accuracy)
        test_accuracies[stride].append(test_accuracy)

# Plot the results
plt.figure(figsize=(10, 6))

# Training accuracies
plt.plot(kernel_sizes, train_accuracies[1], marker='o', color='blue', linestyle='-', label='Training Accuracy (Stride=1)')
plt.plot(kernel_sizes, train_accuracies[2], marker='o', color='red', linestyle='-', label='Training Accuracy (Stride=2)')

# Test accuracies
plt.plot(kernel_sizes, test_accuracies[1], marker='s', color='deepskyblue', linestyle='--', label='Test Accuracy (Stride=1)')
plt.plot(kernel_sizes, test_accuracies[2], marker='s', color='orange', linestyle='--', label='Test Accuracy (Stride=2)')

# Customize the plot
plt.title('Accuracy vs Kernel Size')
plt.xlabel('Kernel Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(kernel_sizes)  # Ensure x-axis shows only the kernel sizes
plt.tight_layout()
plt.show()
