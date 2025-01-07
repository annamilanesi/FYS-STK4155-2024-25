import numpy as np
import os
import tensorflow as tf

import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import transforms
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.geometric.resize import Resize

from functions import *

# Define the directory where the .npy files are stored
data_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/chest_xray'

# Load the preprocessed data
train_data = np.load(os.path.join(data_dir, 'train_data.npy'), allow_pickle=True)
test_data = np.load(os.path.join(data_dir, 'test_data.npy'), allow_pickle=True)
val_data = np.load(os.path.join(data_dir, 'val_data.npy'), allow_pickle=True)

print("Data loaded successfully from .npy files.")

x_train, y_train = split_features_and_labels(train_data)
x_val, y_val = split_features_and_labels(val_data)
x_test, y_test = split_features_and_labels(test_data)

# Define image size
img_size = 150

# Manteniamo i dati nel range 0-255 per l'augmentation
x_train = np.array(x_train)  # Non normalizziamo ancora
x_val = np.array(x_val) / 255  # Solo validazione e test rimangono normalizzati
x_test = np.array(x_test) / 255

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

#--------------- conta i dati prima dell'augmentation ----------------

# Count the number of images for each class after augmentation
unique_labels, counts = np.unique(y_train, return_counts=True)

# Create a dictionary mapping labels to their counts
label_counts = dict(zip(unique_labels, counts))

# Print the results
print(f"Number of normal images (label 1) before augmentation: {label_counts.get(1, 0)}")
print(f"Number of pneumonia images (label 0): {label_counts.get(0, 0)}")


#--------------- fai augmentation ---------------------

# Define a function for data augmentation
def augment_images(images):
    """Applies augmentation to a list of images."""
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])
    augmented_images = [augment(image=img)['image'] for img in images]
    return augmented_images

# Separate normal and pneumonia cases
normal_indices = np.where(y_train == 1)[0]  # Assuming label 1 is for normal cases
pneumonia_indices = np.where(y_train == 0)[0]  # Assuming label 0 is for pneumonia cases

x_train_normal = x_train[normal_indices]
y_train_normal = y_train[normal_indices]

x_train_pneumonia = x_train[pneumonia_indices]
y_train_pneumonia = y_train[pneumonia_indices]

# Perform data augmentation on the normal cases
augmented_normal_images = augment_images(x_train_normal)

# Convert the augmented images and original normal labels back to numpy arrays
augmented_normal_images = np.array(augmented_normal_images)
augmented_normal_labels = np.full(len(augmented_normal_images), 1)  # Label 1 for normal

# Combine the augmented normal images with the original dataset
x_train = np.concatenate([x_train, augmented_normal_images], axis=0)
y_train = np.concatenate([y_train, augmented_normal_labels], axis=0)

x_train = np.array(x_train) / 255 

#------------------ conta i dati dopo augmentation -----------------------

# Count the number of images for each class after augmentation
unique_labels, counts = np.unique(y_train, return_counts=True)

# Create a dictionary mapping labels to their counts
label_counts = dict(zip(unique_labels, counts))

# Print the results
print(f"Number of normal images (label 1) after augmentation: {label_counts.get(1, 0)}")
print(f"Number of pneumonia images (label 0): {label_counts.get(0, 0)}")

# ---------------- Reshape images and prepare for saving ----------------

# Reshape training data
x_train, y_train = prepare_data_for_model(x_train, y_train, img_size)

# Reshape validation data
x_val, y_val = prepare_data_for_model(x_val, y_val, img_size)

# Reshape test data
x_test, y_test = prepare_data_for_model(x_test, y_test, img_size)

# ---------------- Save the data in .npy format -------------------------

# Define directory to save augmented data
output_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/augmented_data'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Save the augmented and reshaped training data
np.save(os.path.join(output_dir, 'x_train_augmented.npy'), x_train)
np.save(os.path.join(output_dir, 'y_train_augmented.npy'), y_train)
np.save(os.path.join(output_dir, 'x_test_augmented.npy'), x_test)
np.save(os.path.join(output_dir, 'y_test_augmented.npy'), y_test)
np.save(os.path.join(output_dir, 'x_val_augmented.npy'), x_val)
np.save(os.path.join(output_dir, 'y_val_augmented.npy'), y_val)

print(f"Augmented and reshaped data saved successfully in: {output_dir}")
