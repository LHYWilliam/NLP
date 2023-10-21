import numpy as np

from comman.functions import (softmax, cross_entropy_error)


class Affine:
    def __init__(self, W: np.ndarray):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, = self.params
        out = np.dot(x, W)
        self.x = x

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        W, = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.y, self.t = None, None
        self.params, self.grads = [], []

    def forward(self, x: np.ndarray, t: np.ndarray):
        self.y, self.t = softmax(x), t

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)

        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx * dout / batch_size

        return dx
