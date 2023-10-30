import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



#########################################################################
#   Read, Normalize and Split Data
#########################################################################

def load_and_process_data(file_name):
    # Read data from CSV file and preprocess it
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        X = []  # Initialize a list to store input data (pixel values)
        y = []  # Initialize a list to store labels
        for row in csv_reader:
            y.append(int(row[0]))
            temp = [float(i) / 255.0 for i in row[1:]]  # Normalize pixel values
            X.append(temp)

    # Convert data to NumPy arrays for further processing
    X = np.asarray(X)
    y = np.asarray(y)

    # Normalize the pixel values using Min-Max scaling
    '''
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    '''


    # Normalize the pixel values using Z-score (standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # Add a bias term (1) to the input data
    X = np.append(X, np.ones((X.shape[0], 1), np.float64), axis=1)

    # Convert labels (y) to one-hot encoding
    num_classes = len(np.unique(y))
    y = np.eye(num_classes)[y]

    return X, y


# Read Training Data
X_train, y_train = load_and_process_data('mnist_train.csv')

# Read Test Data
X_test, y_test = load_and_process_data('mnist_test.csv')


#########################################################################
#   Logistic regression
#########################################################################

def compute_loss(y_true, y_pred, weights, lambda_reg):
    # Compute the binary cross-entropy loss with L2 regularization
    epsilon = 1e-9
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
    regularization_term = (lambda_reg / (2 * len(y_true))) * np.sum(weights ** 2)
    return -np.mean(y1 + y2) + regularization_term



def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))


def feed_forward(X, weights, bias):
    # Perform feedforward operation
    z = np.dot(X, weights) + bias
    A = sigmoid(z)
    return A


def fit(X, y, lr, lambda_reg, stopping_threshold):
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    losses = []  # To store loss at each iteration

    previous_loss = float('inf')  # Set to a large value initially
    iteration = 0

    while True:
        iteration += 1
        A = feed_forward(X, weights, bias)
        loss = compute_loss(y, A, weights, lambda_reg)
        losses.append(loss)

        if abs(loss - previous_loss) < stopping_threshold:
            print("Stopping training as loss change is smaller than the stopping threshold.")
            break

        dz = A - y
        dw = (1 / n_samples) * (np.dot(X.T, dz) + lambda_reg * weights)
        db = (1 / n_samples) * np.sum(dz, axis=0)
        weights -= lr * dw
        bias -= lr * db

        previous_loss = loss

        if iteration % 10 == 0:
            print(f"Iteration {iteration} - Loss: {loss}")

    return weights, bias, losses, iteration

# Call the fit function with the stopping_threshold
learning_rate = 0.001
lambda_reg = 0.1
stopping_threshold = 1e-6

weights, bias, losses,n_iters = fit(X_train, y_train, learning_rate, lambda_reg, stopping_threshold)



# Plot the loss over iterations
plt.figure(1)
plt.plot(range(n_iters), losses, '-g', label='Logistic Regression')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# Make predictions on the test data
predictions = feed_forward(X_test, weights, bias)
predicted_classes = np.argmax(predictions, axis=1)

# Compute and display the confusion matrix and test accuracy
cm = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)
print("Test accuracy: {0:.3f}".format(np.sum(np.diag(cm)) / np.sum(cm)))
print("Confusion Matrix:")
print(np.array(cm))

input('Close app?')