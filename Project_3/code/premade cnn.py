import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop

# --- Loading the data ---
# Define directory where the data is saved
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Load the preprocessed training and test data
x_train = np.load(os.path.join(output_dir, 'x_train_augmented.npy'))
y_train = np.load(os.path.join(output_dir, 'y_train_augmented.npy'))
x_test = np.load(os.path.join(output_dir, 'x_test_augmented.npy'))
y_test = np.load(os.path.join(output_dir, 'y_test_augmented.npy'))

# Add a channel dimension if necessary (e.g., grayscale images)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 1)  # Ensure the dataset has 4 dimensions
x_test = x_test.reshape(x_test.shape[0], 150, 150, 1)

# --- Creating the CNN model ---
model = Sequential()

# Adding the layers to the model
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# --- Training the model ---
epochs = 10
batch_size = 32

print("Training the model...")
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)

# --- Predictions and confusion matrix ---
print("Calculating predictions and confusion matrix...")
y_pred_probs = model.predict(x_test)  # Predicted probabilities
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Convert to binary values (0 or 1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
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
plt.title('Confusion Matrix from pre-made model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#---------------- plot confusion matrix ------------

# Fixed parameters
learning_rate = 0.01
n_estimators = 50
max_depth = 3
min_child_weight = 2
scale_pos_weight = 5  

# Create the XGBoost model
print(f"Training XGBoost model with scale_pos_weight={scale_pos_weight}...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_test_pred = model.predict(x_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix
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
plt.title('Confusion Matrix for XGBoost Model (scale_pos_weight=5)', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
