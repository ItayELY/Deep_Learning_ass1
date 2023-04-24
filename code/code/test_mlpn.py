from grad_check import gradient_check
import numpy as np
import mlpn as mlp
if __name__ == '__main__':


    W, b, U, b_tag = mlp.create_classifier([5,32, 6])

    def _loss_and_W_grad(W):
        global b
        global U
        global b_tag
        loss,grads = mlp.loss_and_gradients(np.array([1,2,3,4,5]),0,[ W, b, U, b_tag ])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag

        loss,grads = mlp.loss_and_gradients(np.array([1,2,3,4,5]),0,[ W, b, U, b_tag ])
        return loss,grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)