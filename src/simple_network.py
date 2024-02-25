import numpy as np
import random


class Network():
    def __init__(self, sizes, activation = "Sigmoid", cost = "MSE"):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])] # Second layer parameter x comes before first layer parameter y to enable matrix multiplication
        self.activation_type = activation
        self.cost = cost

    def forward(self, x):
        for i in range(self.num_layers - 1):
            b = self.biases[i]
            w = self.weights[i]
            x = self.activation(w @ x + b)
        return x
    
    # Use Leaky RELU by default
    def activation(self, x):
        if (self.activation_type == "Leaky RELU"):
            return np.where(x > 0, x, x * 0.03)
        elif (self.activation_type == "Sigmoid"):
            return self.sigmoid(x)
        
        raise Exception("Activation Function Type Does Not Exist")

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def activation_derivative(self, x):
        if (self.activation_type == "Sigmoid"):
            return self.sigmoid(x)*(1-self.sigmoid(x))


        raise Exception("Activation Function Type Does Not Exist")

    def SGD(self, train_data, epochs, batch_size, alpha, test_data = None):
        n = len(train_data)

        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k: k + batch_size] for k in range(0, n, batch_size)]
            
            for batch in batches:
                self.update(batch, alpha)
            
            if (test_data):
                print("Epoch", i + 1, "Accuracy: ", self.evaluate(test_data))
            else:
                print("Epoch ", i + 1, "Complete")
        
    def evaluate(self, data):
        preds = [1 if (np.argmax(self.forward(x)) == np.argmax(y)) else 0 for x,y in data]
        return np.mean(preds)

    def update(self, batch, alpha):

        # Create matrices that store the gradients
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        x,y = zip(*batch)

        X = np.hstack(x)
        Y = np.hstack(y)

        delta_b, delta_w = self.backprop(X, Y)

        # Tale the average of the gradients and then update each weight matrix and bias vector
        self.weights = [w - alpha*(delw)/len(batch) for w, delw in zip(self.weights, delta_w)]
        self.biases = [b - alpha*(delb)/len(batch) for b, delb in zip(self.biases, delta_b)]
        
    def backprop(self, x, y):


        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # Forward Pass
        a = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation(z)
            activations.append(a)

        # Backward Pass  
        for i in range(1, self.num_layers):
            z = zs[-i]
            sp = self.activation_derivative(z)
            if (i == 1):
                delta = self.cost_derivative(activations[-1], y) * self.activation_derivative(zs[-1])
            else:
                delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            delta_b[-i] = np.sum(delta, axis = -1, keepdims = True)
            delta_w[-i] = np.dot(delta, activations[-i-1].transpose())

        return (delta_b, delta_w)

    def cost_derivative(self, output_activations, y):
        if (self.cost == "MSE"):
            return (output_activations-y)
        
        raise Exception("Cost Function Type Does Not Exist")
