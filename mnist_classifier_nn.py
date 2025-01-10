import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import os

def extract_and_normalise_data(data):
    X_data = data[:, 1:].T
    X_data = X_data / 255.0
    Y_data = data[:,0]
    return X_data, Y_data

def preprocess_image_data(data, train_percentage):

    # Convert DataFrame object to numpy array.
    data = np.array(data)

    # m = #no. of samples, n = #no of pixels in each image.
    m, n = data.shape

    # Avoid ordered data by shuffling all rows.
    np.random.shuffle(data)

    # Take 80% of data for trianing the neural entwork.
    train_data = data[0:int(train_percentage*m), :]
    validation_data = data[int(train_percentage*m):m, :]

    # Remove index column, scale, and transpose image data to organise by columns.
    X_train, Y_train = extract_and_normalise_data(train_data)
    X_validation, Y_validation = extract_and_normalise_data(validation_data)
    return X_train, Y_train, X_validation, Y_validation, m

def initialise_random_parameters():
    # Random weights and biases range between -0.5 and +0.5.
    W1 = np.random.rand(10, 784) - 0.5  
    B1 = np.random.rand(10, 1) - 0.5 
    W2 = np.random.rand(10, 10) - 0.5 
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(X):
    return np.maximum(X,0)

# softmax(Y) = [e^(Y_n)]/[SIGMA(e^(Y_i))].
def soft_max(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

# Make classification prediction.
def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = soft_max(Z2)
    return Z1, A1, Z2, A2

def one_hot_converter(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    return one_hot_Y.T

# Optimise weights and biases.
def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y, m):
    one_hot_Y = one_hot_converter(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2)
    dZ1 = 1/m * W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1/m * dZ1.dot(X.T)
    dB1 = 1/m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, m):
    W1, B1, W2, B2 = initialise_random_parameters()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y, m)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if (i%20) == 0:
            print("Iteration number: " + str(i))
            print("Accuracy = " + str(get_accuracy(get_predictions(A2), Y)))
            
    return W1, B1, W2, B2

if __name__ == "__main__":
    # Read training image data from csv file.
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file_name = 'train.csv'
    training_data = pd.read_csv(os.path.join(__location__, file_name))
    X_train, Y_train, X_validation, Y_validation, m = preprocess_image_data(training_data, 0.8)
    W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.1, 10000, m)