import numpy as np
#from cupy import cuda, jit
from loglinear import softmax

STUDENT={'name': 'Itay',
         'ID': '212356364_208390559'}
# HIDDEN_SIZE = 0
HIDDEN_MASK = np.array([])
IN_DIM = 0
LAMBDA = 0


def classifier_output(x, params):
    W, b, U, b_tag = params
    l1 = np.dot(x, W) + b
    l1 = np.tanh(l1)
    probs = np.dot(l1, U) + b_tag
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))


#@jit(target_backend='cuda', forceobj=True)
def loss_and_gradients(x, y, params):
    # FORWARD
    W, b, U, b_tag = params
    l1 = np.dot(np.array(x), W) + b
    l2 = np.tanh(l1)
    l1 = np.tanh(l1)
    
    probs = np.dot(l1, U) + b_tag
    
    softmax_x = softmax(probs)
    loss = -np.log(softmax_x[y])
    
    y_arr = np.zeros(softmax_x.shape)
    y_arr[y] = 1
    diff = softmax_x - y_arr
    
    # BACKWARD
    gU = np.outer(l1, diff)
    gbtag = diff
    
    d_l1 = 1 - np.power(
            np.tanh(np.dot(x, W) + b),
            2)
    d_alpha = np.multiply(
        d_l1,
        np.dot(U, diff)                         # 100,1
    )
    gW = np.outer(x, d_alpha)
    gb = d_alpha
    
    return loss,[gW, gb, gU, gbtag]

def create_classifier(in_dim, hid_dim, out_dim, reg=0, divide=1):
    global LAMBDA
    LAMBDA = reg
    global IN_DIM
    IN_DIM = in_dim
    
    W = np.random.randn(in_dim, hid_dim) / divide
    b = np.random.randn(hid_dim) - 2
    U = np.random.randn(hid_dim, out_dim) / divide
    b_tag = np.zeros(out_dim)
    return [W, b, U, b_tag]

