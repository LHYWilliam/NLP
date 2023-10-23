import argparse

from comman.model import (CBOW)
from comman.optimizer import Adam
from comman.trainer import Trainer
from comman.util import (context_target, to_gpu, save, load)

from dataset import ptb


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--window-size', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--save-file', type=str, default='data/train.model')
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


def print_args(args):
    for key, value in args.items():
        print(f'{key}:{value}', end='  ')
    print()


if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))

    lr = opt.lr
    goal_epoch = opt.epochs
    batch_size = opt.batch_size
    window_size = opt.window_size
    hidden_size = opt.hidden_size
    save_file = opt.save_file
    resume = opt.resume

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    contexts, target = context_target(corpus, window_size)
    contexts, target = to_gpu(contexts), to_gpu(target)

    model = load(save_file) if resume else CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam(lr=lr)
    trainer = Trainer(model, optimizer)

    trainer.train(contexts, target, goal_epochs=goal_epoch, batch_size=batch_size, save_file=save_file)
