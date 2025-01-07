import os
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers
from sklearn.metrics import classification_report

# Define the directory where the augmented data is stored
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the augmented data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Create the CNN model to extract features
def create_cnn_model(input_shape):
    model = models.Sequential([
        # First Conv Layer with 8 filters, 5x5 kernel, stride 2, padding 'same'
        layers.Conv2D(8, (5, 5), strides=2, padding='same', activation='relu', input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization with lambda = 0.01
        layers.MaxPooling2D((2, 2)),  # MaxPooling Layer with pool size 2x2
        
        # Second Conv Layer (similar to first)
        layers.Conv2D(8, (5, 5), strides=2 , activation='relu', padding='same', 
                        kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),  # MaxPooling Layer with pool size 2x2
        
        # Flatten the output
        layers.Flatten(),
        
        # Dense layer with 20 nodes
        layers.Dense(20, activation='relu'),
        
        # Output layer (you can change the number of units based on your use case)
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    return model


# Define the CNN model
cnn_model = create_cnn_model(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# Compile the CNN model (without the final classification layer)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model for feature extraction
cnn_model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))

# Extract features from the training and test images
cnn_features_train = cnn_model.predict(x_train)  # Features for the training data
cnn_features_test = cnn_model.predict(x_test)   # Features for the test data

# Reshape the features to be 1D vectors for input to XGBoost
cnn_features_train = cnn_features_train.reshape(cnn_features_train.shape[0], -1)
cnn_features_test = cnn_features_test.reshape(cnn_features_test.shape[0], -1)

# Define the base directory for saving features
base_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3'  # Project_3 base directory

# Define the directory to save the extracted features
features_dir = os.path.join(base_dir, 'XGboost_features')  # Save features in 'XGboost_features' under Project_3

# Create the directory if it does not exist
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# Save the extracted features to .npy files
np.save(os.path.join(features_dir, 'cnn_features_train.npy'), cnn_features_train)
np.save(os.path.join(features_dir, 'cnn_features_test.npy'), cnn_features_test)

print("Features saved in 'XGboost_features/cnn_features_train.npy' and 'XGboost_features/cnn_features_test.npy'")