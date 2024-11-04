import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from functions import *

np.random.seed(2024)

class LogisticRegression:
    
    def __init__(self, gradient_mode='autograd', learning_rate=0.01, momentum=0.9, lmd=0, n_epochs=100, batch_size=32, beta_initial=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient_mode = gradient_mode
        self.lmd = lmd
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta_initial = beta_initial

    # Cost function using Cross Entropy
    def cost(self, X, y, beta):
        # Calculate probability of predictions
        proba_predictions = self.predict_proba(X, beta)
        
        # Add a small constant to avoid log(0)
        epsilon = 1e-15
        proba_predictions = np.clip(proba_predictions, epsilon, 1 - epsilon)
        
        # Calculate cross-entropy loss
        cross_entropy_value = -np.mean(y * np.log(proba_predictions) + (1 - y) * np.log(1 - proba_predictions))
        
        # Calculate ridge regularization term (excluding the bias term if necessary)
        ridge_penalty = (self.lmd / 2) * np.sum(beta[1:] ** 2)
        
        # Total cost with regularization
        total_cost = cross_entropy_value + ridge_penalty
        return total_cost

    # Gradient of the cost function
    def gradient(self, X, y, beta):
        predictions = sigmoid(X @ beta)
        if self.gradient_mode == "analytical":
            return X.T @ (predictions - y.reshape(-1, 1)) / len(y) + 2 * self.lmd * beta
        elif self.gradient_mode == "autograd":
            cost_fn = lambda b: self.cost(X, y, b)
            cost_grad = grad(cost_fn)
            return cost_grad(beta)

    # Beta Initialization
    def initialize_beta(self, X):
        if self.beta_initial is None:
            self.beta_initial = np.random.randn(X.shape[1], 1)
        return np.copy(self.beta_initial)

    # Sigmoid prediction function
    def predict_proba(self, X, beta):
        return sigmoid(X @ beta)

    def predict(self, X, beta):
        # Returns binary predictions based on a 0.5 threshold
        proba_predictions = self.predict_proba(X, beta)
        return (proba_predictions >= 0.5).astype(int)

    def predict_all(self, X_train, X_test, beta):
        # Calculate probabilities for both training and test sets
        y_tilde_proba = self.predict_proba(X_train, beta)
        y_pred_proba = self.predict_proba(X_test, beta)
        
        # Convert probabilities to binary predictions using the 0.5 threshold
        y_tilde = (y_tilde_proba >= 0.5).astype(int)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return y_tilde, y_pred

    # Accuracy calculation
    def accuracy(self, predictions, targets):
        # Ensure predictions and targets are binary (0 or 1)
        binary_predictions = (predictions >= 0.5).astype(int)
        binary_targets = targets.astype(int)
        # Calculate accuracy using the external accuracy function
        return accuracy_score(binary_targets, binary_predictions)

    # Training using Gradient Descent
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

        accuracy_history = []

        for _ in range(self.n_epochs):
            predictions = self.predict(X, beta)
            acc = self.accuracy(predictions, y)
            accuracy_history.append(acc)

            grad = self.gradient(X, y, beta)
            v = self.momentum * v + self.learning_rate * grad
            beta -= v

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, accuracy_history

    # Training using SGD
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

        accuracy_history = []

        for epoch in range(self.n_epochs):
            predictions = self.predict(X, beta)
            acc = self.accuracy(predictions, y)
            accuracy_history.append(acc)

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, beta)
                v = self.momentum * v + self.learning_rate * grad
                beta -= v

        return beta, accuracy_history

    # Training with ADAM
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

        accuracy_history = []

        for t in range(1, self.n_epochs + 1):
            predictions = self.predict(X, beta)
            acc = self.accuracy(predictions, y)
            accuracy_history.append(acc)

            grad = self.gradient(X, y, beta)
            s = rho1 * s + (1 - rho1) * grad
            r = rho2 * r + (1 - rho2) * (grad ** 2)
            s_hat = s / (1 - rho1 ** t)
            r_hat = r / (1 - rho2 ** t)
            beta -= self.learning_rate * s_hat / (np.sqrt(r_hat) + delta)

            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, accuracy_history

    def beta_ADAM_SGD(self, X, y):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        s = np.zeros_like(theta)
        r = np.zeros_like(theta)
        t = 1

        for epoch in range(self.n_epochs):
            # Shuffle the samples at each epoch
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                # Gradient computation based on logistic loss
                grad = self.gradient(Xi, yi, theta)
                
                # ADAM update rules
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

        accuracy_history = []

        for epoch in range(self.n_epochs):
            # Calculate and store accuracy for the current model
            y_pred = self.predict(X, theta)
            accuracy = np.mean(y_pred == y)  # Calculate accuracy
            accuracy_history.append(accuracy)

            # Shuffle the samples at each epoch
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                # Gradient computation based on logistic loss
                grad = self.gradient(Xi, yi, theta)

                # ADAM update rules
                s = 0.9 * s + (1 - 0.9) * grad
                r = 0.99 * r + (1 - 0.99) * (grad ** 2)
                s_hat = s / (1 - 0.9 ** t)
                r_hat = r / (1 - 0.99 ** t)
                theta = theta - self.learning_rate * s_hat / (np.sqrt(r_hat) + 1e-8)
                t += 1

        return theta, accuracy_history

    # Training with RMSprop

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

        accuracy_history = []

        for _ in range(self.n_epochs):
            # Calculate and store accuracy
            y_pred = self.predict(X, beta)
            accuracy = np.mean(y_pred == y)
            accuracy_history.append(accuracy)
            
            # Calculate the gradient
            grad = self.gradient(X, y, beta)
            r = rho * r + (1 - rho) * (grad ** 2)
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            # Stop if gradient is sufficiently small
            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, accuracy_history

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

    def beta_RMS_SGD_history(self, X, y, rho=0.9, delta=1e-6):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        accuracy_history = []

        for epoch in range(self.n_epochs):
            # Calculate and store accuracy
            y_pred = self.predict(X, theta)
            accuracy = np.mean(y_pred == y)
            accuracy_history.append(accuracy)

            # Shuffle samples
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            # Mini-batch SGD with RMS
            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r = rho * r + (1 - rho) * (grad ** 2)
                theta -= self.learning_rate * grad / (np.sqrt(r) + delta)

        return theta, accuracy_history

    # Training with AdaGrad
    
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

        accuracy_history = []

        for _ in range(self.n_epochs):
            # Calculate and store accuracy
            y_pred = self.predict(X, beta)
            accuracy = np.mean(y_pred == y)
            accuracy_history.append(accuracy)

            # Calculate the gradient
            grad = self.gradient(X, y, beta)
            r += grad ** 2
            beta -= self.learning_rate * grad / (np.sqrt(r) + delta)

            # Stop if gradient is sufficiently small
            if np.linalg.norm(grad) <= 1e-8:
                break

        return beta, accuracy_history

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

    def beta_AdaGrad_SGD_history(self, X, y, delta=1e-7):
        n_samples, n_features = X.shape
        theta = self.initialize_beta(X)
        r = np.zeros_like(theta)

        accuracy_history = []

        for epoch in range(self.n_epochs):
            # Calculate and store accuracy
            y_pred = self.predict(X, theta)
            accuracy = np.mean(y_pred == y)
            accuracy_history.append(accuracy)

            # Shuffle samples
            random_index = np.random.permutation(n_samples)
            X_shuffled = X[random_index]
            y_shuffled = y[random_index]

            # Mini-batch SGD with AdaGrad
            for i in range(0, n_samples, self.batch_size):
                Xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]
                grad = self.gradient(Xi, yi, theta)
                r += grad ** 2
                theta -= self.learning_rate * grad / (np.sqrt(r) + delta)

        return theta, accuracy_history

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

    def plot_heatmaps(self, accuracy_train, accuracy_test, x_labels, y_labels, method, xlabel='X-axis', ylabel='Y-axis'):
        # Create heatmaps for visualization
        plt.figure(figsize=(14, 6))
        plt.suptitle(f'Accuracy Analysis using {method} Method', fontsize=20)

        # Heatmap for training accuracy
        plt.subplot(1, 2, 1)
        sns.heatmap(accuracy_train, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
        plt.title('Accuracy Heatmap (Train Set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Heatmap for test accuracy
        plt.subplot(1, 2, 2)
        sns.heatmap(accuracy_test, annot=True, fmt=".4f", xticklabels=x_labels, yticklabels=y_labels, cmap="viridis")
        plt.title('Accuracy Heatmap (Test Set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title and text
        plt.show()


    def accuracy_polydegree(self, x, y, maxdegree, method):
        accuracy_train = np.zeros(maxdegree)
        accuracy_test = np.zeros(maxdegree)

        # Split the dataset into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        for degree in range(1, maxdegree + 1):
            X_train = design_matrix(x_train, degree)
            X_test = design_matrix(x_test, degree)

            beta, _ = self.select_optimization_method(method, X_train, y_train)
            y_train_pred, y_test_pred = self.predict_all(X_train, X_test, beta)

            accuracy_train[degree - 1] = accuracy(y_train, y_train_pred)
            accuracy_test[degree - 1] = accuracy(y_test, y_test_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, maxdegree + 1), accuracy_train, label='Accuracy Train', marker='o')
        plt.plot(range(1, maxdegree + 1), accuracy_test, label='Accuracy Test', marker='o')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy for Train and Test Sets by Polynomial Degree using {method}')
        plt.legend()
        plt.grid(True)
        plt.show()

        return accuracy_train, accuracy_test

    def accuracy_epochs_vs_batchsize(self, X_train, y_train, X_test, y_test, method, n_epochs_list, minibatch_sizes):
        accuracy_train = np.zeros((len(n_epochs_list), len(minibatch_sizes)))
        accuracy_test = np.zeros((len(n_epochs_list), len(minibatch_sizes)))

        for i, n_epochs in enumerate(n_epochs_list):
            for j, minibatch_size in enumerate(minibatch_sizes):
                self.n_epochs = n_epochs
                self.batch_size = minibatch_size

                beta, _ = self.select_optimization_method(method, X_train, y_train)
                y_train_pred, y_test_pred = self.predict_all(X_train, X_test, beta)

                accuracy_train[i, j] = accuracy(y_train, y_train_pred)
                accuracy_test[i, j] = accuracy(y_test, y_test_pred)

        self.plot_heatmaps(accuracy_train, accuracy_test, minibatch_sizes, n_epochs_list, method, xlabel='Batch size', ylabel='Epochs')
        return accuracy_train, accuracy_test

    def accuracy_learningrate_vs_lmd(self, X_train, y_train, X_test, y_test, method, learning_rate_values, lmd_values):
        accuracy_train = np.zeros((len(learning_rate_values), len(lmd_values)))
        accuracy_test = np.zeros((len(learning_rate_values), len(lmd_values)))

        for i, learning_rate in enumerate(learning_rate_values):
            for j, lmd in enumerate(lmd_values):
                self.lmd = lmd
                self.learning_rate = learning_rate

                beta, _ = self.select_optimization_method(method, X_train, y_train)
                y_train_pred, y_test_pred = self.predict_all(X_train, X_test, beta)

                accuracy_train[i, j] = accuracy(y_train, y_train_pred)
                accuracy_test[i, j] = accuracy(y_test, y_test_pred)

        self.plot_heatmaps(accuracy_train, accuracy_test, lmd_values, learning_rate_values, method, xlabel='Lambda', ylabel='Learning Rate')
        return accuracy_train, accuracy_test

    def accuracy_batchsize_vs_learningrate(self, X_train, y_train, X_test, y_test, method, batch_sizes, learning_rate_values):
        accuracy_train = np.zeros((len(batch_sizes), len(learning_rate_values)))
        accuracy_test = np.zeros((len(batch_sizes), len(learning_rate_values)))

        for i, batch_size in enumerate(batch_sizes):
            for j, learning_rate in enumerate(learning_rate_values):
                self.batch_size = batch_size
                self.learning_rate = learning_rate

                beta, _ = self.select_optimization_method(method, X_train, y_train)
                y_train_pred, y_test_pred = self.predict_all(X_train, X_test, beta)

                accuracy_train[i, j] = accuracy(y_train, y_train_pred)
                accuracy_test[i, j] = accuracy(y_test, y_test_pred)

        self.plot_heatmaps(accuracy_train, accuracy_test, learning_rate_values, batch_sizes, method, xlabel='Learning rate', ylabel='Batch size')
        return accuracy_train, accuracy_test

    def accuracy_batchsize_vs_lambda(self, X_train, y_train, X_test, y_test, method, batch_sizes, lambdas):
        accuracy_train = np.zeros((len(batch_sizes), len(lambdas)))
        accuracy_test = np.zeros((len(batch_sizes), len(lambdas)))

        for i, batch_size in enumerate(batch_sizes):
            for j, lmbd in enumerate(lambdas):
                self.batch_size = batch_size
                self.lmd = lmbd

                beta, _ = self.select_optimization_method(method, X_train, y_train)
                y_train_pred, y_test_pred = self.predict_all(X_train, X_test, beta)

                accuracy_train[i, j] = accuracy(y_train, y_train_pred)
                accuracy_test[i, j] = accuracy(y_test, y_test_pred)

        self.plot_heatmaps(accuracy_train, accuracy_test, lambdas, batch_sizes, method, xlabel='Lambda', ylabel='Batch size')
        return accuracy_train, accuracy_test
