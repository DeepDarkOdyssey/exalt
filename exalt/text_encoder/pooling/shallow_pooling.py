from typing import List
from collections import Counter
import numpy as np
from text_encoder.vocab import BaseVocab


def avg_pooling(token_embeddings: np.ndarray) -> np.ndarray:
    return np.mean(token_embeddings, axis=0)


def max_pooling(token_embeddings: np.ndarray) -> np.ndarray:
    return np.max(token_embeddings, axis=0)


def last_pooling(token_embeddings: np.ndarray) -> np.ndarray:
    return token_embeddings[-1]


def tf_idf_pooling(tokens: List[str], vocab: BaseVocab) -> np.ndarray:
    token_counter = Counter(tokens)
    tf_idf = np.array(
        [
            count / len(tokens) * vocab.idf[token]
            for token, count in token_counter.most_common()
        ]
    )
    token_embeddings = np.array(
        [vocab.token2embed[token] for token, _ in token_counter.most_common()]
    )
    return tf_idf @ token_embeddings
