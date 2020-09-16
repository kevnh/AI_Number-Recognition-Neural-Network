# http://neuralnetworksanddeeplearning.com/chap1.html used as reference

import numpy as np
import random


# Define Network class
class Network(object):
    def __init__(self, sizes, rand):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if rand:    # Randomly initialize weights/biases
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:       # Load weights/biases from file
            self.load_file()


    # Propagate input forward starting with inputs (784x1)
    def feedforward(self, inputs):
        # Iterates through each layer and uses outputs as inputs for the next layer
        for b, w in zip(self.biases, self.weights):
            inputs = sigmoid(np.dot(w, inputs)+b)

        return inputs


    # Stochastic gradient descent algorithm
    def SGD(self, training_data, epochs, mini_batch_size, alpha, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # Create mini batches of specified size from training data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)   # Iterate from 0 to n in increments
            ]                                           # of mini_batch_size
            # Loop to train the network with mini batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            # Separate case for running test data (no learning/adjusting weights)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))


    # Run training_data in specified batch blocks with alpha (learning rate)
    def update_mini_batch(self, mini_batch, alpha):
        # Initialize matrices for weight/bias deltas
        total_bias_delta = [np.zeros(b.shape) for b in self.biases]
        total_weight_delta = [np.zeros(w.shape) for w in self.weights]

        # Calculate weight/bias deltas for entire batch rather than individual runs
        for x, y in mini_batch:     # x is input, y is expected output
            bias_delta, weight_delta = self.backprop(x, y)
            total_bias_delta = [b+db for b, db in zip(total_bias_delta, bias_delta)]
            total_weight_delta = [w+dw for w, dw in zip(total_weight_delta, weight_delta)]

        # Adjust weights/biases
        self.weights = [w+(alpha/len(mini_batch))*nw for w, nw in 
            zip(self.weights, total_weight_delta)]

        self.biases = [b+(alpha/len(mini_batch))*nb for b, nb in
            zip(self.biases, total_bias_delta)]


    # Run actual training algorithm to calculate weight/bias deltas
    def backprop(self, x, y):
        bias_delta = [np.zeros(b.shape) for b in self.biases]
        weight_delta = [np.zeros(w.shape) for w in self.weights]

        activation = x      # Starting x is image input
        activations = [x]   # Stores inputs for each layer
        zs = []             # Stores outputs for each layer
        # Propagate forward, but save inputs/outputs
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Calulate output layer deltas
        delta = sigmoid_prime(zs[-1]) * self.cost_derivative(activations[-1], y)
        bias_delta[-1] = delta
        weight_delta[-1] = np.dot(delta, activations[-2].transpose())

        # Iterate backwards through layers skipping output layer
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            # Calculate delta for current layer using delta from next layer
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            bias_delta[-layer] = delta
            weight_delta[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (bias_delta, weight_delta)


    # Runs network without backpropagation only getting the results
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    # Runs network with user image
    def user_eval(self, data):
        test_results = np.argmax(self.feedforward(data))
        return test_results


    # Compare expected outputs and actual outputs
    def cost_derivative(self, output_activations, y):
        return (y - output_activations)


    # Save weights/biases
    def save_file(self):
        try:
            np.save("data/biases", self.biases)
            np.save("data/weights", self.weights)
        except:
            print("Unable to open file(s) data/biases.npy and/or data/weights.npy.")
            exit()


    # Load weights/biases from file
    def load_file(self):
        try:
            self.biases = np.load("data/biases.npy", allow_pickle=True)
            self.weights = np.load("data/weights.npy", allow_pickle=True)
        except IOError as e:
            print("File(s) data/biases.npy and/or data/weights.npy do not exist.")
            exit()


# Sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
