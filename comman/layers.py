import numpy as np

from comman.functions import (softmax, cross_entropy_error)


class MatMul:
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


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None

    def forward(self, index):
        self.index = index
        W, = self.params

        out = W[index]

        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        np.add.at(dW, self.index, dout)

        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, index):
        target_W = self.embed.forward(index)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)

        return out

    def backward(self, dout: np.ndarray):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)

        dh = dout * target_W

        return dh
