import time

import numpy
import numpy as np
import matplotlib.pyplot as plt

from comman.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def train(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        total_loss, loss_count = 0, 0

        start_time = time.time()
        for epoch in range(max_epoch):
            idx = np.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                x_batch = x[iters * batch_size:(iters + 1) * batch_size]
                t_batch = t[iters * batch_size:(iters + 1) * batch_size]

                loss = self.model.forward(x_batch, t_batch)
                self.model.backward()

                params, grads = remove_duplicate(self.model.params, self.model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                self.optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsd_time = time.time() - start_time
                    print(
                        f'| epoch {self.current_epoch + 1} | iter {iters + 1}/{max_iters} | time {elapsd_time}s | loss {float(avg_loss):.2f}')
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)

        plt.plot(x, self.loss_list, label='train')

        plt.xlabel(f'iterations (x{self.eval_interval})')
        plt.ylabel('loss')

        plt.show()


def remove_duplicate(params: list, grads: list):
    params, grads = params[:], grads[:]

    while True:
        find_flag = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)

                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)

                if find_flag: break
            if find_flag: break
        if find_flag: break

    return params, grads
