import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import *

np.random.seed(2024)

class LinearRegression:
    
    # Class initialization
    def __init__(self, gradient_mode='autograd', learning_rate=0.01, momentum=0.9, lmd=0, n_epochs=100, batch_size=32, beta_initial=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient_mode = gradient_mode
        self.lmd = lmd
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta_initial = beta_initial

        # Method to calculate the cost function
    
    # Gradient calculation methods
    def cost(self, X, y, beta):
        error = X @ beta - y.reshape(-1, 1)
        mse = (1 / len(y)) * np.sum(error ** 2)  # Media dell'errore quadratico
        ridge_penalty = self.lmd * np.sum(beta ** 2)
        return mse + ridge_penalty

    def gradient(self, X, y, beta):
        if self.gradient_mode == "analytical":
            return (2 / len(y)) * X.T @ (X @ beta - y.reshape(-1, 1)) + 2 * self.lmd * beta
        elif self.gradient_mode == "autograd":
            cost_fn = lambda b: self.cost(X, y, b)
            cost_grad = grad(cost_fn)
            return cost_grad(beta)

    # Methods to calculate beta with matrix inversion
    def beta_inversion(self, X, y, lmd=0):
        I = np.identity(X.shape[1])  # Identity matrix
        beta = np.linalg.inv(X.T @ X + lmd * I) @ X.T @ y
        return beta

    # Methods to calculate beta with gradient descent

    def initialize_beta(self, X):
        # Imposta `self.beta_initial` se non è già definito
        if self.beta_initial is None:
            self.beta_initial = np.random.randn(X.shape[1], 1)
        return np.copy(self.beta_initial)  # Usa una copia per evitare modifiche

    # Gradient Descent
    def beta_GD(self, X, y):
        beta = self.initialize_beta(X)
        v = np.zeros_like(beta)

        for _ in range(self.n_epochs):
            grad = self.gradient(X, y, beta)
            v = self.momentum * v + self.learning_rate * grad
            beta -= v

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta

    def beta_GD_history(self, X, y):
        beta = self.initialize_beta(X)
        v = np.zeros_like(beta)

        mse_history = [] 

        for _ in range(self.n_epochs):

            mse = self.cost(X, y, beta)
            mse_history.append(mse)

            grad = self.gradient(X, y, beta)
            v = self.momentum * v + self.learning_rate * grad 
            beta -= v 

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, mse_history

    # Stochastic Gradient Descent (SGD)
    def beta_SGD(self, X, y):
        n_samples = X.shape[0]
        beta = self.initialize_beta(X)
        v = np.zeros_like(beta)

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, beta)
                v = self.momentum * v + self.learning_rate * grad
                beta -= v

        return beta

    def beta_SGD_history(self, X, y):
        n_samples = X.shape[0]
        beta = self.initialize_beta(X)
        v = np.zeros_like(beta)
        mse_history = [] 

        for epoch in range(self.n_epochs):
             
            mse_ = mse(self.predict(X, beta), y)
            mse_history.append(mse_)

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, beta)
                v = self.momentum * v + self.learning_rate * grad 
                beta -= v 

        return beta, mse_history 

    # ADAM
    def beta_ADAM(self, X, y, rho1=0.9, rho2=0.99, delta=1e-8):
        beta = self.initialize_beta(X)
        s = np.zeros_like(beta)
        r = np.zeros_like(beta)

        for t in range(1, self.n_epochs + 1):
            grad = self.gradient(X, y, beta)
            s = rho1 * s + (1 - rho1) * grad
            r = rho2 * r + (1 - rho2) * (grad ** 2)
            s_hat = s / (1 - rho1 ** t)
            r_hat = r / (1 - rho2 ** t)
            beta -= self.learning_rate * s_hat / (np.sqrt(r_hat) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta

    def beta_ADAM_history(self, X, y, rho1=0.9, rho2=0.99, delta=1e-8):
        beta = self.initialize_beta(X)
        s = np.zeros_like(beta)
        r = np.zeros_like(beta)

        mse_history = []

        for t in range(1, self.n_epochs + 1):
             
            mse = self.cost(X, y, beta)
            mse_history.append(mse)

            grad = self.gradient(X, y, beta)
            s = rho1 * s + (1 - rho1) * grad
            r = rho2 * r + (1 - rho2) * (grad ** 2)
            s_hat = s / (1 - rho1 ** t)
            r_hat = r / (1 - rho2 ** t)
            beta -= self.learning_rate * s_hat / (np.sqrt(r_hat) + delta)
            
            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, mse_history 

    def beta_ADAM_SGD(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        s = np.zeros_like(theta)
        r = np.zeros_like(theta)
        t = 1

        for epoch in range(self.n_epochs):
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                grad = self.gradient(Xi, yi, theta)
                s = 0.9 * s + (1 - 0.9) * grad
                r = 0.99 * r + (1 - 0.99) * (grad ** 2)
                s_hat = s / (1 - 0.9 ** t)
                r_hat = r / (1 - 0.99 ** t)
                theta = theta - self.learning_rate * s_hat / (np.sqrt(r_hat) + 1e-8)
                t += 1
            
        return theta

    def beta_ADAM_SGD_history(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        s = np.zeros_like(theta)
        r = np.zeros_like(theta)
        t = 1

        mse_history = []

        for epoch in range(self.n_epochs):

            mse = self.cost(X, y, theta)
            mse_history.append(mse)

            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                grad = self.gradient(Xi, yi, theta)
                s = 0.9 * s + (1 - 0.9) * grad
                r = 0.99 * r + (1 - 0.99) * (grad ** 2)
                s_hat = s / (1 - 0.9 ** t)
                r_hat = r / (1 - 0.99 ** t)
                theta = theta - self.learning_rate * s_hat / (np.sqrt(r_hat) + 1e-8)
                t += 1

        return theta, mse_history

    # RMSprop
    def beta_RMS(self, X, y, rho=0.9, delta=1e-6):
        beta = self.initialize_beta(X)
        r = np.zeros_like(beta)

        for _ in range(self.n_epochs):
            grad = self.gradient(X, y, beta)
            r = rho * r + (1 - rho) * (grad ** 2)
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break
        return beta

    def beta_RMS_history(self, X, y, rho=0.9, delta=1e-6):
        beta = self.initialize_beta(X)
        r = np.zeros_like(beta)

        mse_history = []

        for _ in range(self.n_epochs):

            mse = self.cost(X, y, beta)
            mse_history.append(mse)
            
            grad = self.gradient(X, y, beta)
            r = rho * r + (1 - rho) * (grad ** 2)
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, mse_history

    def beta_RMS_SGD(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        for epoch in range(self.n_epochs):
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r = 0.9 * r + (1 - 0.9) * (grad ** 2)
                theta -= self.learning_rate * grad / (np.sqrt(r + 1e-6))

        return theta

    def beta_RMS_SGD_history(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        mse_history = []

        for epoch in range(self.n_epochs):

            mse = self.cost(X, y, theta)
            mse_history.append(mse)

            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r = 0.9 * r + (1 - 0.9) * (grad ** 2)
                theta -= self.learning_rate * grad / (np.sqrt(r + 1e-6))

        return theta, mse_history
    # AdaGrad
    def beta_AdaGrad(self, X, y, delta=1e-7):
        beta = self.initialize_beta(X)
        r = np.zeros_like(beta)

        for _ in range(self.n_epochs):
            grad = self.gradient(X, y, beta)
            r += grad ** 2
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta

    def beta_AdaGrad_history(self, X, y, delta=1e-7):
        beta = self.initialize_beta(X)
        r = np.zeros_like(beta)

        mse_history = []

        for _ in range(self.n_epochs):

            mse = self.cost(X, y, beta)
            mse_history.append(mse)

            grad = self.gradient(X, y, beta)
            r += grad ** 2
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, mse_history

    def beta_AdaGrad_SGD(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        for epoch in range(self.n_epochs):
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r += grad ** 2
                theta -= self.learning_rate * grad / (np.sqrt(r + 1e-7))

        return theta

    def beta_AdaGrad_SGD_history(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        mse_history = []

        for epoch in range(self.n_epochs):

            mse = self.cost(X, y, theta)
            mse_history.append(mse)

            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r += grad ** 2
                theta -= self.learning_rate * grad / (np.sqrt(r + 1e-7))

        return theta, mse_history

    # Method to make predictions
    def predict(self, X, beta):
        return X @ beta

    def predict_all(self, X_train, X_test, beta):
        y_tilde = X_train @ beta
        y_pred = X_test @ beta
        return y_tilde, y_pred

    # Functions that calculate mse and plot heatmap varying different parameters

    def select_optimization_method(self, method, X_train, y_train):
        if method == "SGD":
            return self.beta_SGD_history(X_train, y_train)
        elif method == "GD":
            return self.beta_GD_history(X_train, y_train)
        elif method == "ADAM":
            return self.beta_ADAM_history(X_train, y_train)
        elif method == "RMS":
            return self.beta_RMS_history(X_train, y_train)
        elif method == "AdaGrad":
            return self.beta_AdaGrad_history(X_train, y_train)
        elif method == "ADAM_SGD":
            return self.beta_ADAM_SGD_history(X_train, y_train)
        elif method == "RMS_SGD":
            return self.beta_RMS_SGD_history(X_train, y_train)
        elif method == "AdaGrad_SGD":
            return self.beta_AdaGrad_SGD_history(X_train, y_train)
        else:
            raise ValueError("Unsupported method")

    def plot_heatmaps(self, mse_train, mse_test, x_labels, y_labels, method, xlabel='X-axis', ylabel='Y-axis'):
        # Create heatmaps for visualization
        plt.figure(figsize=(14, 6))
        plt.suptitle(f'MSE Analysis using {method} Method', fontsize=20)

        # Heatmap for training MSE
        plt.subplot(1, 2, 1)
        sns.heatmap(mse_train, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
        plt.title('MSE Heatmap (Train Set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Heatmap for test MSE
        plt.subplot(1, 2, 2)
        sns.heatmap(mse_test, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
        plt.title('MSE Heatmap (Test Set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title and text
        plt.show()


    def mse_polydegree(self, x, y, z, maxdegree, method):
        # Inizialize vectors for MSE values
        mse_train = np.zeros(maxdegree + 1)
        mse_test = np.zeros(maxdegree + 1)

        # Dividi il dataset in train e test
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

        # Loop over polynomial degrees
        for degree in range(maxdegree + 1):
            # Create design matrices for train and test
            X_train = design_matrix(x_train, y_train, degree)
            X_test = design_matrix(x_test, y_test, degree)
        
            self.beta_initial=None
            beta, _ = self.select_optimization_method(method, X_train, z_train)

            # Calculate predictions
            z_tilde, z_pred = self.predict_all(X_train, X_test, beta)

            # Calculate MSE for training and testing sets
            mse_train[degree] = mse(z_train, z_tilde)
            mse_test[degree] = mse(z_test, z_pred)

        # Crea il grafico dell'andamento dell'MSE
        plt.figure(figsize=(10, 6))
        plt.plot(range(maxdegree + 1), mse_train, label='MSE Train', marker='o')
        plt.plot(range(maxdegree + 1), mse_test, label='MSE Test', marker='o')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error')
        plt.title(f'MSE for Train and Test sets by Polynomial Degree using {method}')
        plt.legend()
        plt.grid(True)
        plt.show()

        return mse_train, mse_test



    def mse_epochs_vs_batchsize(self, X_train, y_train, X_test, y_test, method, n_epochs_list, minibatch_sizes):
        # Initialize matrices for MSE values
        mse_train = np.zeros((len(n_epochs_list), len(minibatch_sizes)))
        mse_test = np.zeros((len(n_epochs_list), len(minibatch_sizes)))

        # Loop over epochs and minibatch sizes
        for i, n_epochs in enumerate(n_epochs_list):
            for j, minibatch_size in enumerate(minibatch_sizes):
                # Update epochs and minibatch size for each run
                self.n_epochs = n_epochs
                self.batch_size = minibatch_size

                # Select the optimization method dynamically
                beta, _ = self.select_optimization_method(method, X_train, y_train)

                """# Calculate predictions
                z_tilde, z_pred = predict_all(X_train, X_test, z_train, z_test, beta)"""

                # Calculate MSE for training and testing sets
                mse_train[i, j] = self.cost(X_train, y_train, beta)
                mse_test[i, j] = self.cost(X_test, y_test, beta)

        # Create heatmaps for visualization
        self.plot_heatmaps(mse_train, mse_test, minibatch_sizes, n_epochs_list, method)

        return mse_train, mse_test

    def mse_learningrate_vs_lmd(self, X_train, y_train, X_test, y_test, method, learning_rate_values, lmd_values):
        # Initialize matrices for MSE values
        mse_train = np.zeros((len(learning_rate_values), len(lmd_values)))
        mse_test = np.zeros((len(learning_rate_values), len(lmd_values)))

        # Loop over learning rates and lambda values
        for i, learning_rate in enumerate(learning_rate_values):
            for j, lmd in enumerate(lmd_values):
                # Update lambda and learning rate for each run
                self.lmd = lmd
                self.learning_rate = learning_rate

                # Select the optimization method dynamically
                beta, _ = self.select_optimization_method(method, X_train, y_train)

                # Calculate MSE for training and testing sets
                mse_train[i, j] = self.cost(X_train, y_train, beta)
                mse_test[i, j] = self.cost(X_test, y_test, beta)

        # Create heatmaps for visualization
        self.plot_heatmaps(mse_train, mse_test, learning_rate_values, lmd_values, method, xlabel='Lambda (lmd)', ylabel='Learning Rate')

        return mse_train, mse_test

    def mse_lambda_vs_learningrate(self, X_train, y_train, X_test, y_test, method, learning_rate_values, lmd_values):
        # Initialize matrices for MSE values
        mse_train = np.zeros((len(lmd_values), len(learning_rate_values)))
        mse_test = np.zeros((len(lmd_values), len(learning_rate_values)))

        # Loop over lambda values and learning rates
        for i, lmd in enumerate(lmd_values):
            for j, learning_rate in enumerate(learning_rate_values):
                # Update lambda and learning rate for each run
                self.lmd = lmd
                self.learning_rate = learning_rate

                # Select the optimization method dynamically
                beta, _ = self.select_optimization_method(method, X_train, y_train)

                # Calculate MSE for training and testing sets
                mse_train[i, j] = self.cost(X_train, y_train, beta)
                mse_test[i, j] = self.cost(X_test, y_test, beta)

        # Create heatmaps for visualization
        self.plot_heatmaps(mse_train, mse_test, learning_rate_values, lmd_values, method, xlabel='Lambda', ylabel='Learning Rate')

        return mse_train, mse_test

    def mse_batchsize_vs_learningrate(self, X_train, y_train, X_test, y_test, method, batch_sizes, learning_rate_values):
        # Initialize matrices for MSE values
        mse_train = np.zeros((len(batch_sizes), len(learning_rate_values)))
        mse_test = np.zeros((len(batch_sizes), len(learning_rate_values)))

        # Loop over batch sizes and learning rates
        for i, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rate_values):
                # Update batch size and learning rate for each run
                self.batch_size = batch_size
                self.learning_rate = learning_rate

                # Select the optimization method dynamically
                beta, _ = self.select_optimization_method(method, X_train, y_train)

                # Calculate MSE for training and testing sets
                mse_train[i, j] = self.cost(X_train, y_train, beta)
                mse_test[i, j] = self.cost(X_test, y_test, beta)

        # Create heatmaps for visualization
        self.plot_heatmaps(mse_train, mse_test, learning_rate_values, batch_sizes, method, xlabel='Learning Rate', ylabel='Batch Size')

        return mse_train, mse_test