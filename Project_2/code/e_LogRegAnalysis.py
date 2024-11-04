import autograd.numpy as np
from autograd import grad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from NeuralNetwork import *
from LogisticRegression import *
from functions import *
import pandas as pd

cancer = load_breast_cancer()

# Download the data for inputs and targets
inputs = cancer.data
targets = cancer.target
targets = targets.reshape(targets.shape[0], 1) # Reshape the targets into a 1D array

# Split the datas into train and test
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)

# Scale datas
scaler = StandardScaler(with_std=True)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

learning_rate = 0.01
momentum = 0.9
lmd = 0
n_epochs = 100
batch_size = 32

beta_initial = np.random.rand(X_train.shape[1], 1)

LogReg = LogisticRegression(gradient_mode='autograd', momentum=momentum, learning_rate=learning_rate,
                            lmd=lmd, n_epochs=n_epochs, batch_size=batch_size, beta_initial= beta_initial)

# Define parameters for analysis
learning_rate_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1]
lmd_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
n_epochs_list = [10, 50, 100, 200, 500, 1000]
minibatch_sizes = [8, 16, 32, 64, 128, 256]
batch_sizes = [8, 16, 32, 64, 128, 256]

# Accuracy for epochs vs. batch size
"""accuracy_train_epochs_batchsize, accuracy_test_epochs_batchsize = LogReg.accuracy_epochs_vs_batchsize(
    X_train, y_train, X_test, y_test, method='SGD', n_epochs_list=n_epochs_list, minibatch_sizes=minibatch_sizes
)"""

# Accuracy for learning rate vs. lambda
accuracy_train_learningrate_lmd, accuracy_test_learningrate_lmd = LogReg.accuracy_learningrate_vs_lmd(
    X_train, y_train, X_test, y_test, method='SGD', learning_rate_values=learning_rate_values, lmd_values=lmd_values
)

# Accuracy for batch size vs. learning rate
accuracy_train_batchsize_learningrate, accuracy_test_batchsize_learningrate = LogReg.accuracy_batchsize_vs_learningrate(
    X_train, y_train, X_test, y_test, method='SGD', batch_sizes=batch_sizes, learning_rate_values=learning_rate_values
)
