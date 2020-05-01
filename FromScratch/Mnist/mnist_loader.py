# libraries
# S.L.
import gzip
import pickle

import numpy as np


def load_data():
    """unzip and deserialize the data
    training_data[0] numbers
    training_data[1] labels

    validation_data[0] numbers
    validation_data[1] labels

    test_data[0] numbers
    test_data[1] labels
    """
    # https://github.com/MichalDanielDobrzanski/DeepLearningPython35
    with gzip.open('../../mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data  # a new way to define tuple


def load_data_handle():
    """
    formats training, validate and test data.
    :returns : a tuple containing (training_data, validation_data, test_data)
    """
    tr_d, v_d, te_d = load_data()

    flat = (28 * 28, 1)
    training_inputs = [np.reshape(x_in, flat) for x_in in tr_d[0]]  # flatten data
    training_results = [vectorize_load(x_res) for x_res in tr_d[1]]  # determine data correspoding to the digit.
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(v_in, flat) for v_in in v_d[0]]
    validation_results = v_d[1]  # [vectorize_load(v_res) for v_res in v_d[1]]  # there is only one for that dataset
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(te_in, flat) for te_in in te_d[0]]
    test_results = te_d[1]  # [vectorize_load(te_res) for te_res in te_d[1]] # there is only one for that dataset
    test_data = zip(test_inputs, test_results)

    return training_data, validation_data, test_data

def vectorize_load(j):
    """
    0 <-> 9
    0 -> 0000000001
    1 -> 0000000010
    2 -> 0000000100
    3 -> 0000001000
    ...
    9 -> 1000000000
    :param j:
    :return:
    """
    # (0..9) => classify as J'st index is True or 1
    e = np.zeros((10, 1))  # 10 digits
    e[j] = 1.0
    return e
