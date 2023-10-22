import collections

import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def preprocess(text):
    text = text.lower().replace('.', ' .')
    words = text.split(' ')

    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word


def comatrix(corpus, vocab_count, window_size=1):
    co_matrix = np.zeros((vocab_count, vocab_count), dtype=np.int32)

    for index, word_id in enumerate(corpus):
        for window in range(1, window_size + 1):
            for window_index in (index - window, index + window):
                if 0 <= window_index < len(corpus):
                    window_id = corpus[window_index]
                    co_matrix[word_id, window_id] += 1

    return co_matrix


def similarity(x, y):
    nx = x / np.sqrt(np.sum(x ** 2))
    ny = y / np.sqrt(np.sum(y ** 2))
    return np.dot(nx, ny)


def similarities(query, word_to_id, id_to_word, co_matrix,
                 top=None, show=False) -> dict:
    query_id = word_to_id[query]
    query_vector = co_matrix[query_id]
    vocab_count = len(word_to_id)
    if top is None:
        top = vocab_count - 1

    similar_vector = np.zeros(vocab_count)
    for i in range(vocab_count):
        similar_vector[i] = similarity(co_matrix[i], query_vector)

    result = {}
    for word_id in (-similar_vector).argsort()[1:top + 1]:
        result[id_to_word[word_id]] = similar_vector[word_id]

    if show:
        print(f"[query] {query}")
        for key, value in result.items():
            print(f'{key:10}{value:4.2}')
        print()

    return result


def ppmi(co_matrix, eps=1e-8, show=False):
    ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix)
    S = np.sum(co_matrix, axis=0)

    now, total = 0, co_matrix.shape[0] * co_matrix.shape[1]

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(co_matrix[i, j] * N / (S[i] * S[j]) + eps)
            ppmi_matrix[i, j] = max(0, pmi)

            if show:
                now = now + 1
                progress_bar(now, total)

    return ppmi_matrix


def context_target(corpus, window_size=1):
    contexts = []
    target = corpus[window_size:-window_size]

    for index in range(window_size, len(corpus) - window_size):
        context = []
        for window in range(-window_size, window_size + 1):
            if window != 0:
                context.append(corpus[index + window])
        contexts.append(context)

    return np.array(contexts), np.array(target)


def convert_one_hot(source, vocal_size):
    target_shape = (*source.shape, vocal_size)
    target = np.zeros(target_shape, dtype=np.int32)

    target[*np.indices(source.shape), source] = 1

    return target


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def remove_duplicate(params, grads):
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


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        self.vocal_size = len(counts)

        self.word_p = np.zeros(self.vocal_size)
        for i in range(self.vocal_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        negative_sample = np.random.choice(self.vocal_size, size=(batch_size, self.sample_size),
                                           replace=True, p=self.word_p)

        return negative_sample


def progress_bar(now, total, message='', basis=0.01):
    count = int((now / total + basis) * 10)
    print(f'\r{message} [' + '-' * count + ' ' * (10 - count) + ']' +
          f' {count}/10', end='')


def to_gpu(x):
    import cupy
    if type(x) is cupy.ndarray:
        return x
    return cupy.asarray(x)
