import numpy as np

def split_features_and_labels(data):
    features = []
    labels = []
    
    for feature, label in data:
        features.append(feature)
        labels.append(label)
        
    return features, labels


def prepare_data_for_model(features, labels, img_size):
    
    # Reshape the features array to include the grayscale channel (1)
    reshaped_features = np.array(features).reshape(-1, img_size, img_size, 1)
    
    # Convert labels list to a NumPy array
    np_labels = np.array(labels)
    
    return reshaped_features, np_labels
