import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, Adagrad
from keras.regularizers import l2
from keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

# Fissa il seed per la riproducibilità
np.random.seed(42)                  # Fissa il seed per NumPy
random.seed(42)                     # Fissa il seed per il modulo random

#------------------- DEPHT OF CNN --------------

# Define directory where the augmented data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented training data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Fixed parameters
n_filters = 8                 # Number of filters
kernel_size = (5, 5)           # Kernel size
pool_size = (2, 2)             # Pooling size
stride = 2                     # Stride
batch_size = 16                # Batch size
lmbda = 0.01                   # L2 regularization factor
n_nodes = 20                   # Number of nodes in dense layer
epochs = 10                    # Fixed number of epochs for analysis

# Function to create the CNN model with variable depth
def create_cnn_model(depth):
    model = Sequential()
    model.add(layers.Input(shape=(150, 150, 1)))
    
    # Add convolution and pooling layers based on the depth
    for i in range(depth):
        model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, 
                         activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Flatten())
    model.add(Dense(units=n_nodes, activation='relu', kernel_regularizer=l2(lmbda)))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    optimizer = Adagrad()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create a function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cbar=False, 
        cmap='viridis', 
        xticklabels=["Predicted Pneumonia", "Predicted Normal"], 
        yticklabels=["Actual Pneumonia", "Actual Normal"], 
        annot_kws={"size": 16}
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Train and evaluate CNN models with different depths (1, 2, 3, 4)
depths = [1, 2, 3]
train_accuracies = []
test_accuracies = []

for depth in depths:
    print(f"Training CNN with depth {depth}")
    
    # Create the model
    model = create_cnn_model(depth)
    
    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Record train accuracy
    train_accuracy = history.history['accuracy'][-1]
    train_accuracies.append(train_accuracy)
    
    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_accuracies.append(test_accuracy)
    
    # Predict on the test data
    y_pred = (model.predict(x_test) > 0.5).astype('int32')  # Convert probabilities to 0 or 1
    
    # Plot confusion matrix for this model
    plot_confusion_matrix(y_test, y_pred, title=f'Confusion Matrix for CNN with {depth} Convolution Layers')

# Plot the accuracy vs depth graph
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='deepskyblue')
plt.plot(depths, test_accuracies, label='Test Accuracy', marker='o', linestyle='--', color='orange')
plt.title('Accuracy vs Depth of CNN')
plt.xlabel('Depth of CNN (Number of Convolution Layers)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
