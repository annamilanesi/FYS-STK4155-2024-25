import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, regularizers

#--------------------- ANALYSIS CHANGEING THE NUMBER OF FILTERS -----------------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))

# Load the test and validation data
x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))
x_val = np.load(os.path.join(output_dir, 'x_val_augmented.npy'))
y_val = np.load(os.path.join(output_dir, 'y_val_augmented.npy'))

# Define the model creation function
def create_model(num_filters, lambda_val):
    model = Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(150, 150, 1)))
    
    # Convolutional Layer with variable number of filters
    model.add(layers.Conv2D(
        filters=num_filters, 
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='same',
    ))
    
    # Max Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # Hidden Dense Layer with L2 regularization
    model.add(layers.Dense(
        units=20,  # Fixed number of nodes in the dense layer
        activation='relu',
        kernel_regularizer=regularizers.l2(lambda_val)  # L2 regularization
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
filters_list = [8, 16, 32, 64, 128]  # Number of filters to test
batch_size = 16  # Fixed batch size
epochs = 15  # Fixed number of epochs
lambda_val = 0.01  # Fixed L2 regularization

# Arrays to store accuracies
train_accuracies = []
test_accuracies = []

# Loop over the number of filters
for num_filters in filters_list:
    print(f"Training model with num_filters={num_filters}.")
    
    # Create the model
    model = create_model(num_filters, lambda_val)
    
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
    
    # Append the accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(filters_list, train_accuracies, label='Training Accuracy', marker='o', linestyle='-', color='deepskyblue')
plt.plot(filters_list, test_accuracies, label='Test Accuracy', marker='o', linestyle='--', color='orange')

# Labels and title
plt.title('Accuracy vs Number of Filters')
plt.xlabel('Number of Filters')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
