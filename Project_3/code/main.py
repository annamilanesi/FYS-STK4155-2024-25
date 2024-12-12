import os
import zipfile
import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# Download dataset using Kaggle API
def download_and_extract_kaggle_dataset(dataset_name, dest_dir):
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", dest_dir], check=True)
    zip_path = os.path.join(dest_dir, f"{dataset_name.split('/')[-1]}.zip")
    
    # Unzip the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    
    os.remove(zip_path)  # Remove the zip file

# Define paths and labels
dataset_name = "paultimothymooney/chest-xray-pneumonia"
dest_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3/chest_xray'
data_dir = os.path.join(dest_dir, "chest_xray")
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

# Download and extract the dataset if it doesn't already exist
if not os.path.exists(data_dir):
    os.makedirs(dest_dir, exist_ok=True)
    download_and_extract_kaggle_dataset(dataset_name, dest_dir)
    print("Dataset downloaded and extracted.")

# Function to load and preprocess images into a single unified array
def get_training_data(data_dir, subset):
    subset_path = os.path.join(data_dir, subset)
    data = []
    
    for label in labels:
        path = os.path.join(subset_path, label)
        class_num = labels.index(label)
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)  # Define img_path here
            
            try:
                # Read and process the image
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print(f"Warning: Failed to load image {img_path}")
                    continue
                
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                
                # Append the resized image and label as a single list element
                data.append([resized_arr, class_num])
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    # Convert to a numpy array of objects (heterogeneous structure)
    data = np.array(data, dtype=object)
    return data

# Load train, test, and validation data
train_data = get_training_data(data_dir, 'train')
test_data = get_training_data(data_dir, 'test')
val_data = get_training_data(data_dir, 'val')

print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples, {len(val_data)} validation samples.")

# Save the arrays for future use
np.save(os.path.join(dest_dir, 'train_data.npy'), train_data)
np.save(os.path.join(dest_dir, 'test_data.npy'), test_data)
np.save(os.path.join(dest_dir, 'val_data.npy'), val_data)

print("Data saved as .npy files.")

#------------------ stampa alcune foto --------------------

# Initialize counters for each class
normal_count = 0
pneumonia_count = 0

# Set up a figure with 4 subplots
plt.figure(figsize=(5, 5))

# Loop through images in train_data
for i, (img, label) in enumerate(train_data):
    if label == 0 and normal_count < 2:  # If NORMAL and less than 2 shown
        title = "NORMAL"
        normal_count += 1
    elif label == 1 and pneumonia_count < 2:  # If PNEUMONIA and less than 2 shown
        title = "PNEUMONIA"
        pneumonia_count += 1
    else:
        continue  # Skip this image if we already have 2 of this type

    # Plot the image in the next subplot
    plt.subplot(2, 2, normal_count + pneumonia_count)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

    # Stop if we have plotted 4 images in total
    if normal_count + pneumonia_count == 4:
        break

plt.tight_layout()
plt.show()
