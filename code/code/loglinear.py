import numpy as np

STUDENT={'name': 'Itay',
         'ID': '212356364_208390559'}

def softmax(x):
    x = np.exp(x - np.max(x))/np.exp(x - np.max(x)).sum()
    return x
    

def classifier_output(x, params):
    W,b = params
    probs = np.dot(x, W) + b
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    W,b = params
    x_np = np.array(x)
    softmax_x = softmax(np.dot(x_np, W) + b)
    y_arr = np.zeros(softmax_x.shape)
    y_arr[y] = 1
    diff = softmax_x - y_arr
    loss = -np.log(softmax_x[y])
    gW = np.outer(x_np, diff)
    gb = diff
    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
