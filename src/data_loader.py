# -*- coding: utf-8 -*-
import pickle
import gzip
import os
import numpy as np

def load_data():
    """ Loads the MNIST data and returns it as a tuple of the form
    (training_data, validation_data, testing_data)
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train, val, test = pickle.load(f, encoding="latin1")
    f.close()
    return (train, val, test)


def load_data_wrapper():
    print("In this file")
    print(os.getcwd())
    train, val, test = load_data()
    train_input = [np.reshape(x, (len(x), 1)) for x in train[0]] # Reshape input (n, ) to (n, 1)
    train_result = [one_hot_encode(y) for y in train[1]]
    train_data = list(zip(train_input, train_result))
    val_input = [np.reshape(x, (len(x), 1)) for x in val[0]] # Reshape input (n, ) to (n, 1)
    val_result = [one_hot_encode(y) for y in val[1]]
    val_data = list(zip(val_input, val_result))
    test_input = [np.reshape(x, (len(x), 1)) for x in test[0]] # Reshape input (n, ) to (n, 1)
    test_result = [one_hot_encode(y) for y in test[1]]
    test_data = list(zip(test_input, test_result))

    return (train_data, val_data, test_data)


def one_hot_encode(n): 
    """Converts a decimal digit into a one-hot encoded vector and returns the vector.
    
    This is the output that we are trying to match for each digit during prediction.
    """
    v = np.zeros((10, 1))
    v[n] = 1.0
    return v

