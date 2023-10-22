import numpy as np

from comman.layers import (MatMul, SoftmaxWithLoss)


class CBOW:
    def __init__(self, vocal_size, hidden_size):
        W_in = 0.01 * np.random.randn(vocal_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocal_size).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        self.layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> np.ndarray:
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5

        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)

        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)


class SkipGram:
    def __init__(self, vocal_size, hidden_size):
        W_in = 0.01 * np.random.randn(vocal_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocal_size).astype('f')

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        self.layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> np.ndarray:
        h = self.in_layer.forward(target)
        score = self.out_layer.forward(h)

        loss1 = self.loss_layer1.forward(score, target)
        loss2 = self.loss_layer2.forward(score, target)

        return loss1 + loss2

    def backward(self, dout=1):
        dloss1 = self.loss_layer1.backward(dout)
        dloss2 = self.loss_layer2.backward(dout)
        dscore = dloss1 + dloss2

        dh = self.out_layer.backward(dscore)

        self.in_layer.backward(dh)


