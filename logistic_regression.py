import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

#########################################################################
#   Read, Normalize and Split Data
#########################################################################

def load_and_process_data(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        X = []
        y = []
        for row in csv_reader:
            y.append(int(row[0]))
            temp = [float(i) / 255.0 for i in row[1:]]  # Normalize pixel values
            X.append(temp)

    # Normalize and preprocess the data
    X = np.asarray(X)
    y = np.asarray(y)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = np.append(X, np.ones((X.shape[0], 1), np.float64), axis=1)

    # Convert y to one-hot encoding
    num_classes = len(np.unique(y))
    y = np.eye(num_classes)[y]

    return X, y

# Usage:
# To load and process the training data
X_train, y_train = load_and_process_data('mnist_train.csv')

# To load and process the test data
X_test, y_test = load_and_process_data('mnist_test.csv')

#########################################################################
#   Logistic regression
#########################################################################

def compute_loss(y_true, y_pred):
    epsilon = 1e-9
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    return -np.mean(y1 + y2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(X, weights, bias):
    z = np.dot(X, weights) + bias
    A = sigmoid(z)
    return A

def fit(X, y, n_iters, lr):
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    losses = []

    for _ in range(n_iters):
        A = feed_forward(X, weights, bias)
        losses.append(compute_loss(y, A))
        dz = A - y
        dw = (1 / n_samples) * np.dot(X.T, dz)
        db = (1 / n_samples) * np.sum(dz, axis=0)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias, losses

def predict(X, weights, bias):
    y_hat = np.dot(X, weights) + bias
    y_predicted = sigmoid(y_hat)
    return y_predicted

learning_rate = 0.001
n_iters = 500

weights, bias, losses = fit(X_train, y_train, n_iters, learning_rate)

plt.figure(1)
plt.plot(range(n_iters), losses, '-g', label='Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

predictions = predict(X_test, weights, bias)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

print("Test accuracy: {0:.3f}".format(np.sum(np.diag(cm)) / np.sum(cm)))
print("Confusion Matrix:")
print(np.array(cm))
