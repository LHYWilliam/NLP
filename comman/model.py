import numpy as np

from comman.layers import (Affine, SoftmaxWithLoss)


class CBOW:
    def __init__(self, vocal_size, hidden_size):
        W_in = 0.01 * np.random.randn(vocal_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocal_size).astype('f')

        self.in_layer0 = Affine(W_in)
        self.in_layer1 = Affine(W_in)
        self.out_layer = Affine(W_out)
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

    def backward(self, dout: int = 1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
