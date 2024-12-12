import numpy as np
import os
import pandas as pd
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, AveragePooling2D
from keras.regularizers import l2

#--------------------- POOLING LAYER -------------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))

x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Define parameters for the analysis
pooling_types = ["max", "average", "none"]  # Pooling types to test
pool_sizes = [(2, 2), (3, 3)]              # Pool sizes to test
n_filters = 8                             # Fixed number of filters
kernel_size = (5, 5)                       # Fixed kernel size
stride = 1                                 # Fixed stride
epochs = 15                                 # Fixed number of epochs
batch_size = 16                            # Fixed batch size
lmbda = 0.01                              # L2 regularization factor

# Store results in a table
results = []

# Loop over pooling types and sizes
for pooling_type in pooling_types:
    for pool_size in pool_sizes:
        print(f"Testing {pooling_type} pooling with pool size {pool_size}.")
        
        # Build the model
        model = Sequential()
        model.add(layers.Input(shape=(150, 150, 1)))
        model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, 
                         activation='relu', padding='same'))
        
        # Add pooling layer if not "none"
        if pooling_type == "max":
            model.add(MaxPooling2D(pool_size=pool_size))
        elif pooling_type == "average":
            model.add(AveragePooling2D(pool_size=pool_size))
        # If "none", skip adding a pooling layer

        # Add Flatten and Dense layers
        model.add(Flatten())
        model.add(Dense(units=20, activation='relu', kernel_regularizer=l2(lmbda)))  # Dense layer with 20 nodes
        model.add(Dense(units=1, activation='sigmoid'))  # Output layer
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model and evaluate accuracy
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        train_accuracy = history.history['accuracy'][-1]  # Last training epoch accuracy
        test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]  # Test accuracy
        
        # Append results to the table
        results.append({
            "Pooling Type": pooling_type,
            "Pool Size": f"{pool_size[0]}x{pool_size[1]}",
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy
        })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display the table
print(results_df)
