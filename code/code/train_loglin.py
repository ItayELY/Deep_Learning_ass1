import loglinear as ll
import random
import numpy as np
import utils as u
import sys

STUDENT={'name': 'Itay',
         'ID': '212356364_208390559'}
ITERATIONS = 50
L_RATE = 0.01
L2I = {}
F2I = {}


def feats_to_vec(features):
    vec = np.zeros(u.TOP_K)
    for f in features:
        if f in F2I:
            vec[F2I[f]] += 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if L2I[label] == ll.predict(feats_to_vec(features), params):
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
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            W = W - learning_rate * grads[0]
            b = b - learning_rate * grads[1]
            params = [W, b]
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # learning_rate = 0.99 * learning_rate
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    unigram = "u" in sys.argv
    L2I, F2I = u.L2I, u.F2I
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    print("*** Training bigram loglin. ***")
    # data = u.read_data(u.TRAIN_PATH)
    # print(len(F2I))
    # print(F2I)
    params = ll.create_classifier(u.TOP_K, len(u.L2I))
    trained_params = train_classifier(u.TRAIN, u.DEV, ITERATIONS, L_RATE, params)
    print("*** Testing ***")
    test_data = u.TEST
    I2F = {v: k for k, v in u.L2I.items()}
    test_output = []
    with open("./test.pred", "w") as test_file:
        for s in test_data:
            l = I2F[
                ll.predict(feats_to_vec(s), trained_params)
            ]
            test_output.append(l)
        test_file.writelines("\n".join(test_output))
    if unigram:
        L2I, F2I = u.L2I_UNI, u.F2I_UNI
        L_RATE = 0.1
        print("*** Training unigram loglin. ***")
        # print(len(F2I))
        # print(F2I)
        params = ll.create_classifier(u.TOP_K, len(u.L2I_UNI))
        trained_params = train_classifier(u.TRAIN_UNI, u.DEV_UNI, ITERATIONS, L_RATE, params)
    
