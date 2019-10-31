# REF: https://www.aclweb.org/anthology/P18-2084

from text_encoder.vocab import BaseVocab, Tokenizer
import numpy as np


def docov(text: str, vocab: BaseVocab, tokenizer: Tokenizer, add_mean: bool = True):
    tokens = tokenizer(text)
    document_matrix = vocab.embedding_matrix[vocab.tokens2indexes(tokens)]
    C = np.cov(document_matrix, rowvar=False)
    text_embed = np.zeros(vocab.embed_size * (vocab.embed_size + 1) // 2)
    for p in range(C.shape[0]):
        for q in range(C.shape[1]):
            if p < q:
                text_embed[p + q] = 2 ** 0.5 * C[p, q]
            elif p == q:
                text_embed[p + q] = C[p, q]
    if add_mean:
        mean_embed = np.mean(document_matrix, axis=0)
        text_embed = np.hstack((text_embed, mean_embed))
    return text_embed
