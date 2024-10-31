import autograd.numpy as np 
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from functions import *
from LinearRegression import *

#LINEAR REGRESSION WITH GRADIENT DESCENT (comparison of convergence with or without mini-batches)

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

beta_initial = np.random.rand(X.shape[1], 1)

#common parameters
n_epochs = 100
batch_size = 10
momentum = 0.0

# Model inizialization
model = LinearRegression(beta_initial=beta_initial, gradient_mode="autograd", learning_rate=0.01, n_epochs=n_epochs, batch_size=batch_size, momentum=momentum)

# list of different methods
methods = {
    'GD': model.beta_GD_history,
    'SGD': model.beta_SGD_history,
    'ADAM': model.beta_ADAM_history,
    'ADAM_SGD': model.beta_ADAM_SGD_history,
    'RMSprop': model.beta_RMS_history,
    'RMSprop_SGD': model.beta_RMS_SGD_history,
    'AdaGrad': model.beta_AdaGrad_history,
    'AdaGrad_SGD': model.beta_AdaGrad_SGD_history
}

# colors configuartion
colors = {
    'GD': 'b', 'SGD': 'b--',
    'ADAM': 'r', 'ADAM_SGD': 'r--',
    'RMSprop': 'g', 'RMSprop_SGD': 'g--',
    'AdaGrad': 'm', 'AdaGrad_SGD': 'm--'
}

# Plotting
plt.figure(figsize=(12, 8))

for method_name, method_function in methods.items():

    beta, mse_history = method_function(X, z)
    
    style = colors[method_name]
    plt.plot(mse_history, style, label=method_name)


plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epochs for Various Optimization Methods')
plt.legend()
plt.grid(True)
plt.show()

#fix the method and vary the size of mini_batches. See how, as the size of the minibatch approches the total numner of datas(full batch),
#the convergence rate of the stocastic method (SGD) approches the one of plain method (GD)

full_batch = len(z)

batch_sizes = [1, 5, 10, 50, 100, 200, full_batch]

for batch_size in batch_sizes:
    model = LinearRegression(beta_initial=beta_initial, gradient_mode="autograd", learning_rate=0.01, n_epochs=n_epochs, batch_size=batch_size, momentum=momentum)
    beta, mse_history = model.beta_SGD_history(X, z) #choose the method (SGD for example)
    if batch_size == full_batch:
        plt.plot(mse_history, label=f"full batch, batch size: {batch_size}")
    else:
        plt.plot(mse_history, label=f"batch size: {batch_size}")

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epochs for Various sizes of mini-batch')
plt.legend()
plt.grid(True)
plt.show()