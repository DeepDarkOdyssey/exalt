from copy import copy
from typing import List, Tuple, Callable, Set
import io
import os
import numpy as np
import spacy


def gram_schmidt_process(A: np.array) -> Tuple[np.ndarray]:
    d, n = A.shape
    Q = np.zeros((d, n))
    R = np.zeros((n, n))
    for i in range(n):
        v_i = A[:, i]
        qs = Q[:, :i]
        rs = v_i @ qs
        R[:i, i] = rs
        q_i = v_i - np.sum((v_i @ qs) / np.sum((qs ** 2), axis=0) * qs, axis=1)
        norm = np.linalg.norm(q_i, ord=2)
        Q[:, i] = q_i / norm
        R[i, i] = norm
    return Q, R


def ngrams(
    text: str, max_n: int, tokenizer: Callable[[str], List[str]] = lambda x: list(x)
) -> Set:
    tokens = tokenizer(text)
    grams = set()
    for n in range(1, max_n + 1):
        grams = grams.union(
            set(["".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])
        )
    return grams


def get_related_cache_path(file_path: str) -> str:
    dir_path = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(dir_path, file_name + '.cache')