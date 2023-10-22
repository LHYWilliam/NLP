import time

import numpy
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.loss_list = []
        self.val_per_iter = None
        self.current_epoch = 0

    def train(self, x, t, goal_epochs=10, batch_size=32, val_per_iter=20):
        self.val_per_iter = val_per_iter

        data_size = len(x)
        goal_iters = data_size // batch_size

        total_loss, loss_count = 0, 0

        start_time = time.time()
        for epoch in range(goal_epochs):
            index = np.random.permutation(numpy.arange(data_size))
            x, t = x[index], t[index]

            for iters in range(goal_iters):
                x_batch = x[iters * batch_size:(iters + 1) * batch_size]
                t_batch = t[iters * batch_size:(iters + 1) * batch_size]

                loss = self.model.forward(x_batch, t_batch)
                self.model.backward()

                params, grads = remove_duplicate(self.model.params, self.model.grads)
                self.optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                val = (val_per_iter is not None) and (iters % val_per_iter) == 0
                if val:
                    average_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(f'| epoch {self.current_epoch + 1} | iter {iters + 1}/{goal_iters} '
                          f'| time {elapsed_time:.2f}s | loss {float(average_loss):.6f}')
                    self.loss_list.append(float(average_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self):
        x = np.arange(len(self.loss_list))

        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x{self.val_per_iter})')
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

                if find_flag:
                    break
            if find_flag:
                break
        if find_flag:
            break

    return params, grads
