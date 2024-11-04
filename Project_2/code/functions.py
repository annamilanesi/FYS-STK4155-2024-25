import autograd.numpy as np 
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def accuracy(predictions, targets):
    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_targets = targets.astype(int)  # Assicurati che i target siano nel formato binario (0 o 1)
    
    # Calculate accuracy
    return accuracy_score(binary_targets, binary_predictions)

#function to calculate the design matrix
def design_matrix(x, y, degree):
    N = len(x)
    l = int((degree + 1) * (degree + 2) / 2)  # Number of polynomial terms
    X = np.ones((N, l))

    idx = 0
    for i in range(degree + 1):
        for j in range(i + 1):
            X[:, idx] = (x ** (i - j)) * (y ** j)
            idx += 1
    return X

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def leaky_ReLU(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def ELU(x, alpha=1.0):
    """Exponential Linear Unit (ELU) activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

#derivatives of activation functions
def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1-sig)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def softmax_der(z):
    s = softmax(z)
    # Create the Jacobian matrix
    jacobian_m = np.diagflat(s) - np.outer(s, s)
    return jacobian_m

def leaky_ReLU_der(x, alpha=0.01):
    """Derivative of the Leaky ReLU activation function."""
    return np.where(x > 0, 1, alpha)

def ELU_der(x, alpha=1.0):
    """Derivative of the ELU activation function."""
    return np.where(x > 0, 1, elu(x, alpha) + alpha)
    
#cost functions
def mse(predict, target):
    return np.mean((predict - target) ** 2)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

#derivatives of cost functions
def mse_der(predict, target):
    predict = predict.reshape(-1, 1) 
    target = target.reshape(-1, 1)    
    return 2 * (predict - target) / predict.shape[0]

def cross_entropy_der(predictions, targets):
    return predictions - targets
