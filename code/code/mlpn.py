from typing import List
import numpy as np
from numba import cuda, jit
from loglinear import softmax

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


@jit(target_backend='cuda', forceobj=True)
def classifier_output(x, params: List[np.ndarray]):
    Ws = params[::2]
    bs = params[1:][::2]
    probs = x
    for W, b in zip(Ws, bs):
        probs = probs.dot(W) + b
        probs = np.tanh(probs)
    return probs

@jit(target_backend='cuda', forceobj=True)
def predict(x, params):
    return np.argmax(classifier_output(x, params))

@jit(target_backend='cuda', forceobj=True)
def loss_and_gradients(x, y, params: List[np.ndarray]):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # FORWARD
    W_ = params[::2]
    b_ = params[1:][::2]
    z_ = W_.copy()
    a_ = W_.copy()
    v = x
    last_l = len(W_) - 1
    
    for l, _ in enumerate(W_):
      
        z_[l] = v = v.dot(W_[l]) + b_[l]

        if l == last_l:
            a_[l] = v = softmax(v)
        else:
            a_[l] = v = np.tanh(v)
    
    loss = -np.log(v[y])
    
    # BACKWARD
    y_true = np.zeros(v.shape)
    y_true[y] = 1
    diff = v - y_true
    
    grads = []  # reversed after appending all db & dW
    d_zl = diff
    grads.append(d_zl)  # start with db
    dW = np.outer(a_[-2], d_zl)
    grads.append(dW)  # then dW
    for l in reversed(range(1, last_l + 1)):   # we are now at l=3 calculate for l=2
        d_al = np.dot(W_[l], d_zl)
        '''
             print("l={} calculating {}.".format(l, l-1))
        print("W_[l]", W_[l].shape)
        print("d_zl", d_zl.shape)
        print("d_al", d_al.shape)
        '''
        d_zl = 1-np.tanh(z_[l-1])**2
        d_zl = np.multiply(d_al, d_zl)
        grads.append(d_zl)
        if l > 1:
            grads.append(np.outer(a_[l-2], d_zl))
        else:
            grads.append(np.outer(x, d_zl))
    grads.reverse()
    return loss, grads

# @jit(target_backend='cuda', forceobj=True)
def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    print(dims)
    params = []
    for prv, nxt in zip(dims, dims[1:]):
        params.append(np.random.rand(prv, nxt) -0.5)
        params.append(np.random.rand(nxt)) # changed prv to nxt
    print("params:")
    [print(" ", u.shape, u.max(), u.min()) for u in params]
    return params

