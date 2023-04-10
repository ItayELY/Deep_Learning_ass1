import sys
print(sys.argv)
mlpn = len(sys.argv) == 2
if mlpn:
    import mlpn as mlp
    print("Using mlpn.")
else:
    import mlp1 as mlp
import random
import numpy as np
import utils as u
import os


STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}
HIDDEN_LAYER = 36
ITERATIONS = 50
L_RATE = 0.01
LAMBDA = 0.2
PRINT_ACC = 4

def feats_to_vec(features):
    vec = np.zeros(u.TOP_K)
    for f in features:
        if f in u.F2I:
            vec[u.F2I[f]] += 1
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if u.L2I[label] == mlp.predict(feats_to_vec(features), params):
            good += 1
        else:
            bad += 1
    return good / (good + bad)


# @mlp.jit(target_backend='cuda', forceobj=True)
def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = u.L2I[label]                  # convert the label to number if needed.
            #print("hi")
            #print(params)
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
              np.round(train_accuracy-dev_accuracy, PRINT_ACC),
              )
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    data = u.read_data(u.TRAIN_PATH)
    # print(len(u.L2I))
    if mlpn:
        params = mlp.create_classifier([int(s) for s in sys.argv[1].split(";")])
    else:
        params = mlp.create_classifier(u.TOP_K, HIDDEN_LAYER, len(u.L2I), reg=5)
    print("i", "t_lss ", "t_acc ", "d_acc ", "delta", "\t", "Wmx   ", "Wmn   ", "Umx   ", "Umn   ")
    trained_params = train_classifier(u.TRAIN, u.DEV, ITERATIONS, L_RATE, params)
    print("***** TESTING ******")
    test_data = u.TEST
    I2F = {v: k for k, v in u.L2I.items()}
    test_output = []
    with open("./test.pred", "w") as test_file:
        for s in test_data:
            l = I2F[
                mlp.predict(feats_to_vec(s), trained_params)
            ]
            test_output.append(l)
        test_file.writelines("\n".join(test_output))
