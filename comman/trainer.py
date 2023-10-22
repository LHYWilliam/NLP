import time

import numpy
import cupy as np
import matplotlib.pyplot as plt

from comman.util import (progress_bar)
from comman.util import (remove_duplicate)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.loss_list = []

        self.current_epoch = 0
        self.val_per_iter = None

    def train(self, x, t, goal_epochs=10, batch_size=32):
        data_size = len(x)
        goal_iters = data_size // batch_size
        self.val_per_iter = goal_iters // 10

        total_loss, loss_count = 0, 0

        start_time = time.time()
        for epoch in range(goal_epochs):
            index = numpy.random.permutation(numpy.arange(data_size))
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

                if (self.val_per_iter is not None) and (iters % self.val_per_iter) == 0 or iters == goal_iters - 1:
                    average_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    message = f'| epoch {self.current_epoch + 1:{len(str(goal_epochs))}} ' \
                              f'| iter {iters + 1:{len(str(goal_iters))}}/{goal_iters} ' \
                              f'| loss {float(average_loss):.4f} ' \
                              f'| time {elapsed_time:.2f}s'
                    progress_bar(iters, goal_iters, message)
                    self.loss_list.append(float(average_loss))
                    total_loss, loss_count = 0, 0

            print()
            self.current_epoch += 1

    def plot(self):
        x = numpy.arange(len(self.loss_list))

        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x{self.val_per_iter})')
        plt.ylabel('loss')

        plt.show()
