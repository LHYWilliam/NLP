from comman.model import (CBOW)
from comman.optimizer import Adam
from comman.trainer import Trainer
from comman.util import (context_target, to_gpu, save, load)

from dataset import ptb

if __name__ == '__main__':
    file = 'data/train.model'
    window_size = 1
    hidden_size = 100
    batch_size = 100
    goal_epoch = 1

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = context_target(corpus, window_size)
    contexts, target = to_gpu(contexts), to_gpu(target)

    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.train(contexts, target, goal_epoch, batch_size)
    trainer.plot()

    save(model, optimizer, corpus, word_to_id, id_to_word, file)
    # file = 'data/train.model'
    # window_size = 1
    # hidden_size = 100
    # batch_size = 100
    # goal_epoch = 100
    #
    # model, optimizer, corpus, word_to_id, id_to_word = load(file)
    # vocab_size = len(word_to_id)
    #
    # contexts, target = context_target(corpus, window_size)
    # contexts, target = to_gpu(contexts), to_gpu(target)
    #
    # trainer = Trainer(model, optimizer)
    #
    # trainer.train(contexts, target, goal_epoch, batch_size)
    # trainer.plot()
    #
    # save(model, optimizer, corpus, word_to_id, id_to_word, file)
