import autograd.numpy as np 
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from functions import *
from LinearRegression import *

#ANLYSIS OF LINEAR REGRESSION WITH DIFFERENT GRADIENT DESCENT METHODS (varying different hyperparameters)

np.random.rand(2024)

#create the data set
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) + np.random.normal(0, 0.1)

#flatten the data into arrays
x = x.ravel()
y = y.ravel()
z = z.ravel()

#create the design matrix with a fixed degree
degree = 11
X = design_matrix(x, y, degree)

print(X.shape)

# Split into training and testing sets
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

# Define parameters for analysis
learning_rate_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1]
lmd_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
n_epochs_list = [10, 50, 100, 200, 500, 1000]
minibatch_sizes = [8, 16, 32, 64, 128, 256]
batch_sizes = [8, 16, 32, 64, 128, 256]

# Set an initial value for beta
beta_initial = np.random.rand(X.shape[1], 1)

# Instantiate the LinearRegression model
model = LinearRegression(beta_initial=beta_initial, learning_rate=0.05)

# Call the MSE functions 
mse_train_epochs_batchsize, mse_test_epochs_batchsize = model.mse_epochs_vs_batchsize(
    X_train, z_train, X_test, z_test, method='SGD', n_epochs_list=n_epochs_list, minibatch_sizes=minibatch_sizes
)

mse_train_learningrate_lmd, mse_test_learningrate_lmd = model.mse_learningrate_vs_lmd(
    X_train, z_train, X_test, z_test, method='SGD', learning_rate_values=learning_rate_values, lmd_values=lmd_values
)

mse_train_batchsize_learningrate, mse_test_batchsize_learningrate = model.mse_batchsize_vs_learningrate(
    X_train, z_train, X_test, z_test, method='SGD', batch_sizes=batch_sizes, learning_rate_values=learning_rate_values
)

# Instantiate the LinearRegression model
model2 = LinearRegression(gradient_mode='autograd')

# Graph for mse varying polydegree
maxdegree = 15
mse_train_poly, mse_test_poly = model2.mse_polydegree(x, y, z, maxdegree, method='SGD')

plt.savefig("a_mse_vs_polydegree.png", format="png")