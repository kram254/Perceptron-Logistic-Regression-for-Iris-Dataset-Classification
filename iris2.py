import numpy as np
import matplotlib.pyplot as plt

def perceptron(X, y, n_iter=50):
    # initialize weights to zero
    w = np.zeros(X.shape[1] + 1)
    errors = []

    for _ in range(n_iter):
        # shuffle data
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        # iterate over each data point
        for xi, yi in zip(X, y):
            # add bias term
            xi = np.append(xi, 1)
            # calculate predicted class
            y_hat = np.where(np.dot(xi, w) >= 0, 1, -1)
            # update weights if incorrect
            if y_hat != yi:
                delta = yi - y_hat
                w += delta * xi
        # calculate and store number of misclassifications
        errors.append((y != np.where(np.dot(X, w[:-1]) + w[-1] >= 0, 1, -1)).sum())

    return w, errors

if __name__ == '__main__':
    # load data
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0, 1, 2, 3, 4), dtype='object')

    # extract features and labels
    X = data[:, :-1].astype(float)
    y = np.where(data[:, -1] == 'Iris-versicolor', 1, -1)

    # train perceptron
    w, errors = perceptron(X, y)

    # plot number of misclassifications vs. iteration
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Number of misclassifications')
    plt.show()

    # plot data and decision boundary (Sepal features only)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = np.where(np.dot(np.c_[xx.ravel(), yy.ravel(), np.zeros(len(xx.ravel())), np.zeros(len(xx.ravel()))], w[:-1]) + w[-1] >= 0, 1, -1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.title('Perceptron - Iris data (Sepal features only)')
    plt.show()

    # plot errors during training
    plt.plot(range(1, len(errors)+1), errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.title('Perceptron - Learning rate 0.001')
    plt.show()
    
    
    
    # plot data and decision boundary (Petal features only)
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap='bwr')
    x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = np.where(np.dot(np.c_[np.zeros(len(xx.ravel())), np.zeros(len(xx.ravel())), xx.ravel(), yy.ravel()], w[:-1]) + w[-1] >= 0, 1, -1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.title('Perceptron - Iris data (Petal features only)')
    plt.show()
    
    # plot data and decision boundary
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bo', label='Iris-versicolor')
    plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'ro', label='Iris-setosa')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx = np.linspace(x_min, x_max)
    yy = -(w[0] * xx + w[-1]) / w[1]
    plt.plot(xx, yy, 'k-', label='decision boundary')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.title('Perceptron - Iris data (Petal features only)')
    plt.legend(loc='upper left')
    plt.show()