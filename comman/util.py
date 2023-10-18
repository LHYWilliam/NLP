import numpy as np


def preprocess(text: str) -> (np.ndarray, dict, dict):
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


def comatrix(corpus: np.ndarray, vocab_count: int, window_size: int = 1) -> (np.ndarray):
    co_matrix = np.zeros((vocab_count, vocab_count), dtype=np.int32)

    for index, word_id in enumerate(corpus):
        for window in range(1, window_size + 1):
            for window_index in (index - window, index + window):
                if 0 <= window_index < len(corpus):
                    window_id = corpus[window_index]
                    co_matrix[word_id, window_id] += 1

    return co_matrix


def similarity(x: np.ndarray, y: np.ndarray) -> (np.float64):
    nx = x / np.sqrt(np.sum(x**2))
    ny = y / np.sqrt(np.sum(y**2))
    return np.dot(nx, ny)


def similarities(query: str, word_to_id: dict, id_to_word: dict, co_matrix: np.ndarray,
                 top: int = None, show: bool = False) -> (dict):
    query_id = word_to_id[query]
    query_vector = co_matrix[query_id]
    vocab_count = len(word_to_id)
    if top is None:
        top = vocab_count - 1

    similar_vector = np.zeros(vocab_count)
    for i in range(vocab_count):
        similar_vector[i] = similarity(co_matrix[i], query_vector)

    result = {}
    for id in (-similar_vector).argsort()[1:top+1]:
        result[id_to_word[id]] = similar_vector[id]

    if show:
        print(f"[query] {query}")
        for key, value in result.items():
            print(f'{key:10}{value:4.2}')
        print()

    return result


def ppmi(co_matrix: np.ndarray, eps: float = 1e-8, show: bool = False) -> (np.ndarray):
    ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix)
    S = np.sum(co_matrix, axis=0)

    cnt, now, total = 0, 0, co_matrix.shape[0] * co_matrix.shape[1]

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(co_matrix[i, j] * N / (S[i] * S[j]) + eps)
            ppmi_matrix[i, j] = max(0, pmi)

            if show:
                cnt += 1
                if (cnt / total) > (now / 10): now += 1
                print('\r[' + '-' * now + ' ' * (10 - now) + ']' + f' {now}/10', end='')

    return ppmi_matrix
