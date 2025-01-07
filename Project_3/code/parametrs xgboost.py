import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Define directories
base_dir = '/Users/annamilanesi/Desktop/Machine Learning/Project_3'  # Base directory
features_dir = os.path.join(base_dir, 'XGboost_features')  # XGBoost features directory
labels_dir = os.path.join(base_dir, 'augmented_data')  # Augmented data directory

# Load features and labels
x_train = np.load(os.path.join(features_dir, 'cnn_features_train.npy'))
x_test = np.load(os.path.join(features_dir, 'cnn_features_test.npy'))
y_train = np.load(os.path.join(labels_dir, 'y_train_augmented.npy'))
y_test = np.load(os.path.join(labels_dir, 'y_test_augmented.npy'))


### FIRST ANALYSIS: Vary max_depth and min_child_weight ###
# Fixed parameters
learning_rate = 0.01
n_estimators = 50

# Hyperparameter ranges for heatmap
max_depth_range = [2, 3, 5, 7, 9]  # Depth of the trees
min_child_weight_range = [1, 2, 3, 5, 10]  # Minimum sum of instance weight (hessian) for a child

# Initialize arrays to store train and test accuracies
train_accuracies = np.zeros((len(max_depth_range), len(min_child_weight_range)))
test_accuracies = np.zeros((len(max_depth_range), len(min_child_weight_range)))

print("Starting first analysis: varying max_depth and min_child_weight...")
# Loop through combinations of max_depth and min_child_weight
for i, max_depth in enumerate(max_depth_range):
    for j, min_child_weight in enumerate(min_child_weight_range):
        print(f"Training XGBoost with max_depth={max_depth}, min_child_weight={min_child_weight}...")
        # Create the XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            eval_metric='logloss'
        )
        
        # Fit the model on the training data
        model.fit(x_train, y_train)
        
        # Predict on the training and test sets
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        print("Train Classification Report:")
        print(classification_report(y_train, y_train_pred))

        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Calculate accuracies
        train_accuracies[i, j] = accuracy_score(y_train, y_train_pred)
        test_accuracies[i, j] = accuracy_score(y_test, y_test_pred)

# Plot heatmaps for training and test accuracies
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training accuracy heatmap
sns.heatmap(train_accuracies, annot=True, fmt=".2f", ax=ax[0], cmap="viridis",
            xticklabels=min_child_weight_range, yticklabels=max_depth_range)
ax[0].set_title('Training Accuracy')
ax[0].set_xlabel('Min Child Weight')
ax[0].set_ylabel('Max Depth')

# Test accuracy heatmap
sns.heatmap(test_accuracies, annot=True, fmt=".2f", ax=ax[1], cmap="viridis",
            xticklabels=min_child_weight_range, yticklabels=max_depth_range)
ax[1].set_title('Test Accuracy')
ax[1].set_xlabel('Min Child Weight')
ax[1].set_ylabel('Max Depth')

plt.tight_layout()
plt.show()


### SECOND ANALYSIS: Vary learning_rate and n_estimators ###
# Fixed parameters
max_depth = 3
min_child_weight = 2

# Hyperparameter ranges
learning_rate_range = [0.001, 0.01, 0.05, 0.1, 1.0]  # Learning rates
n_estimators_range = [10, 50, 100, 150, 200]  # Number of estimators (trees)

# Initialize arrays to store train and test accuracies
train_accuracies_lr = np.zeros((len(learning_rate_range), len(n_estimators_range)))
test_accuracies_lr = np.zeros((len(learning_rate_range), len(n_estimators_range)))

print("Starting second analysis: varying learning_rate and n_estimators...")
# Loop through combinations of learning_rate and n_estimators
for i, learning_rate in enumerate(learning_rate_range):
    for j, n_estimators in enumerate(n_estimators_range):
        print(f"Training XGBoost with learning_rate={learning_rate}, n_estimators={n_estimators}...")
        # Create the XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            eval_metric='logloss'
        )
        
        # Fit the model on the training data
        model.fit(x_train, y_train)
        
        # Predict on the training and test sets
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        # Calculate accuracies
        train_accuracies_lr[i, j] = accuracy_score(y_train, y_train_pred)
        test_accuracies_lr[i, j] = accuracy_score(y_test, y_test_pred)

# Plot heatmaps for training and test accuracies (learning_rate vs n_estimators)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training accuracy heatmap
sns.heatmap(train_accuracies_lr, annot=True, fmt=".2f", ax=ax[0], cmap="viridis",
            xticklabels=n_estimators_range, yticklabels=learning_rate_range)
ax[0].set_title('Training Accuracy (Learning Rate vs N Estimators)')
ax[0].set_xlabel('N Estimators')
ax[0].set_ylabel('Learning Rate')

# Test accuracy heatmap
sns.heatmap(test_accuracies_lr, annot=True, fmt=".2f", ax=ax[1], cmap="viridis",
            xticklabels=n_estimators_range, yticklabels=learning_rate_range)
ax[1].set_title('Test Accuracy (Learning Rate vs N Estimators)')
ax[1].set_xlabel('N Estimators')
ax[1].set_ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# Fixed parameters for this analysis
max_depth = 3  # Fixed depth of the trees
learning_rate = 0.01  # Fixed learning rate
n_estimators = 50  # Fixed number of estimators
scale_pos_weight_range = [1, 2, 5, 10, 20]  # Range of scale_pos_weight values to test

# Initialize arrays to store train and test accuracies
train_accuracies_spw = []
test_accuracies_spw = []

print("Starting first analysis: varying scale_pos_weight...")
# Loop through scale_pos_weight values
for scale_pos_weight in scale_pos_weight_range:
    print(f"Training XGBoost with scale_pos_weight={scale_pos_weight}...")
    # Create the XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    # Fit the model on the training data
    model.fit(x_train, y_train)
    
    # Predict on the training and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate and store accuracies
    train_accuracies_spw.append(accuracy_score(y_train, y_train_pred))
    test_accuracies_spw.append(accuracy_score(y_test, y_test_pred))

# Plot training and test accuracy vs scale_pos_weight
plt.figure(figsize=(8, 6))
plt.plot(scale_pos_weight_range, train_accuracies_spw, label='Train Accuracy', marker='o')
plt.plot(scale_pos_weight_range, test_accuracies_spw, label='Test Accuracy', marker='o')
plt.title('Accuracy vs Scale Pos Weight')
plt.xlabel('Scale Pos Weight')
plt.ylabel('Accuracy')
plt.xticks(scale_pos_weight_range)
plt.legend()
plt.grid()
plt.tight_layout()
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
