import numpy as np


class Adam:
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.iter, self.m, self.v = 0, None, None

    def update(self, params: list, grads: list):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i] + 1e-7))
