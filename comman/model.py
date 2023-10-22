import cupy as np

from comman.layers import (MatMul, SoftmaxWithLoss, Embedding, NegativeSamplingLoss)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class CBOW:
    def __init__(self, vocal_size, hidden_size, windows_size, corpus):
        W_in = 0.01 * np.random.randn(vocal_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(vocal_size, hidden_size).astype('f')

        self.in_layers = []
        for i in range(2 * windows_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)

        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)

        loss = self.ns_loss.forward(h, target)

        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)

        for layer in self.in_layers:
            layer.backward(dout)

        return None


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

    def forward(self, contexts, target):
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

        return None
