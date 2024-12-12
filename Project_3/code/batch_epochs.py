import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras import layers

#-------------------- ANALYSIS FOR BATCHSIZE VS EPOCHS ---------------------

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

print(x_train.shape)

# Define the model creation function
def create_model(n_nodes):
    model = Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(150, 150, 1)))
    
    # Convolutional Layer
    model.add(layers.Conv2D(
        filters=16, 
        kernel_size=(3, 3),
        activation='relu',
        strides=1,
        padding='same',
    ))
    
    # Max Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # Hidden Dense Layer
    model.add(layers.Dense(units=n_nodes, activation='relu'))
    
    # Output Layer (Binary classification)
    model.add(layers.Dense(units=1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define ranges for batch sizes and number of epochs
batch_sizes = [8, 16, 32, 64]
epochs_list = [5, 10, 20, 30]
n_nodes = 20  # Number of nodes in the dense layer

# Create empty arrays to store accuracies
train_accuracies = np.zeros((len(batch_sizes), len(epochs_list)))
test_accuracies = np.zeros((len(batch_sizes), len(epochs_list)))

# Loop over all combinations of batch sizes and epochs
for i, batch_size in enumerate(batch_sizes):
    for j, epochs in enumerate(epochs_list):
        print(f"Training model with batch_size={batch_size} and epochs={epochs}.")
        
        # Create the model
        model = create_model(n_nodes)
        
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
        
        # Store the accuracies in the arrays
        train_accuracies[i, j] = train_accuracy
        test_accuracies[i, j] = test_accuracy

# Plot heatmaps for training and test accuracies
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training accuracy heatmap
sns.heatmap(train_accuracies, annot=True, fmt=".2f", ax=ax[0], cmap="viridis",
            xticklabels=epochs_list, yticklabels=batch_sizes)
ax[0].set_title('Training Accuracy')
ax[0].set_xlabel('Number of Epochs')
ax[0].set_ylabel('Batch Size')

# Test accuracy heatmap
sns.heatmap(test_accuracies, annot=True, fmt=".2f", ax=ax[1], cmap="viridis",
            xticklabels=epochs_list, yticklabels=batch_sizes)
ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Number of Epochs')
ax[1].set_ylabel('Batch Size')

plt.tight_layout()
plt.show()
