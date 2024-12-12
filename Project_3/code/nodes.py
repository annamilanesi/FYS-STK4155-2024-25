import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

#--------------------- ANALYSIS TO FIND THE NUMBER OF NODES FOR DENSE LAYER ------------------------

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

#---------------------- inizia modello cnn ------------------------

# Define the model
def create_model(hidden_units):
    model = Sequential()

    # Input Layer
    model.add(layers.Input(shape=(150, 150, 1)))  # Input layer with the shape of the images
    
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
    
    # Hidden Dense Layer with variable number of nodes
    model.add(layers.Dense(
        units=hidden_units, 
        activation='relu', 
        #kernel_regularizer=regularizers.l2(0.01)  # Ridge regularization (L2)
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

epochs = 25
batch_size = 8

# List of hidden units to test
hidden_units_list = [1,5,10,15,20,30,50]

# Store accuracy values for plotting
train_accuracies = []
test_accuracies = []

for hidden_units in hidden_units_list:
    print(f"Training model with {hidden_units} hidden units.")
    
    # Create the model with a variable hidden layer size
    model = create_model(hidden_units)

    early_stopping = EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

    # Fit the model (training it for 'epochs' number of epochs)
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping]
    )

    # After training, get the accuracy for the last epoch
    train_accuracy = model.history.history['accuracy'][-1]  # Accuracy after all epochs
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]  # Test accuracy

    # Save the final accuracies for plotting
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot accuracy
plt.figure(figsize=(10, 6))

# Plot Training Accuracy vs Number of Hidden Units
plt.plot(hidden_units_list, train_accuracies, label='Training Accuracy', marker='o', linestyle='-', color='deepskyblue')

# Plot Test Accuracy vs Number of Hidden Units
plt.plot(hidden_units_list, test_accuracies, label='Test Accuracy', marker='o', linestyle='--', color='orange')

# Labels and title
plt.title('Training and Test Accuracy for Different Hidden Layer Sizes')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show plot
plt.show()