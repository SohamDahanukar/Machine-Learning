import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data (you can replace this with your own dataset)
np.random.seed(42)
X = np.random.rand(100, 2)  # Features (2D data)
y = np.where(X[:, 0] + X[:, 1] > 1, 1, -1)  # Binary labels (1 or -1)

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Training the SVM
for _ in range(num_iterations):
    for i in range(len(X)):
        if y[i] * (np.dot(w, X[i]) + b) < 1:
            w += learning_rate * (y[i] * X[i])
            b += learning_rate * y[i]

# Make predictions
def predict(X):
    return np.sign(np.dot(X, w) + b)

# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = np.dot(xy, w) + b
Z = Z.reshape(XX.shape)

plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary (Without sklearn)")
plt.show()
