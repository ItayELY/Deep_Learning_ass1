import random

import numpy as np

import xor_data
import mlp1 as mlp
ITERATIONS = 100
L_RATE = 0.5
LAMBDA = 0.2
PRINT_ACC = 4
def feats_to_vec(features):
    return np.array(features)
def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if label == mlp.predict(feats_to_vec(features), params):
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            # print("hi")
            # print(params)
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            for i, _ in enumerate(params):
                params[i] -= learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I,
              np.round(train_loss, PRINT_ACC),
              np.round(train_accuracy, PRINT_ACC),
              np.round(dev_accuracy, PRINT_ACC),
              np.round(train_accuracy - dev_accuracy, PRINT_ACC),
              )
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    data = xor_data.data
    # print(len(u.L2I))
    params = mlp.create_classifier(2, 3, 2, reg=5)
    print("i", "t_lss ", "t_acc ", "d_acc ", "delta", "\t", "Wmx   ", "Wmn   ", "Umx   ", "Umn   ")
    trained_params = train_classifier(data, data, ITERATIONS, L_RATE, params)
    print("***** TESTING ******")
'''
    #test_data = u.TEST
   # I2F = {v: k for k, v in u.L2I.items()}
    test_output = []
    with open("./test.pred", "w") as test_file:
        for s in test_data:
            l = I2F[
                mlp.predict(feats_to_vec(s), trained_params)
            ]
            test_output.append(l)
        test_file.writelines("\n".join(test_output))
'''