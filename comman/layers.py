import numpy as np

from functions import (softmax, cross_entropy_error)


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray = None):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        W, b = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW

        if b is not None:
            db = np.sum(dout, axis=0)
            self.grads[1][...] = db

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
