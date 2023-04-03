import loglinear as ll
import random
import numpy as np
import utils as u
import os

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}
ITERATIONS = 30
L_RATE = 0.01


def feats_to_vec(features):
    vec = np.zeros(u.TOP_K)
    for f in features:
        if f in u.F2I:
            vec[u.F2I[f]] += 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if u.L2I[label] == ll.predict(feats_to_vec(features), params):
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    W, b = params
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = u.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            #print(grads)
            W = W - learning_rate * grads[0]
            b = b - learning_rate * grads[1]
            params = [W, b]
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    data = u.read_data(u.TRAIN_PATH)
    # print(len(u.L2I))
    params = ll.create_classifier(u.TOP_K, len(u.L2I))
    trained_params = train_classifier(u.TRAIN, u.DEV, ITERATIONS, L_RATE, params)
