import numpy as np
from loglinear import softmax
STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    W, b, U, b_tag = params
    l1 = np.dot(x, W) + b
    l1 = np.tanh(l1)
    probs = np.dot(l1, U) + b_tag
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    l1 = np.dot(np.array(x), W) + b
    l1 = np.tanh(l1)
    probs = np.dot(l1, U) + b_tag
    
    softmax_x = softmax(probs)
    y_arr = np.zeros(softmax_x.shape)
    y_arr[y] = 1
    
    diff = softmax_x - y_arr
    loss = -np.log(softmax_x[y])
    
    # print("l1", l1)
    gU = np.outer(l1, diff)
    gbtag = diff
    # print(diff)
    Wx_plus_b = np.dot(x, W) + b
    d_alpha = np.multiply(
        1 - np.power(np.tanh(Wx_plus_b), 2),    # 100,1
        np.dot(U, diff)                         # 100,1
    )                                           # 100,1
    
    gW = np.outer(x, d_alpha)                     # 600,1
    gb = d_alpha
    # print(gW)
    return loss,[gW, gb, gU, gbtag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(in_dim, hid_dim)
    b = np.random.randn(hid_dim)
    U = np.zeros((hid_dim, out_dim))
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

