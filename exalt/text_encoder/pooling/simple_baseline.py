# REF: http://arxiv.org/abs/1602.03483
# Related Ropo: https://github.com/PrincetonML/SIF (Official)

import pickle
import numpy as np
from typing import List, Optional, Iterable
from text_encoder.vocab import BaseVocab, Tokenizer


class SIF(object):
    def __init__(self, a: float = 10e-3, num_principles: int = 1):
        self.a = a
        self.num_principles = num_principles
        self.corpus_principles = None

    def build_sentence_embed(self, tokens: Iterable[str], vocab: BaseVocab):
        sentence_embed = np.mean(
            [
                self.a
                / (self.a + vocab.token_frequency[token])
                * vocab.token2embed[token]
                for token in tokens
            ],
            axis=0,
        )
        return sentence_embed

    def build_corpus_principles(
        self,
        corpus: List[str],
        vocab: BaseVocab,
        tokenizer: Tokenizer,
        save_to: Optional[str] = None,
    ) -> np.ndarray:
        sentence_embeddings = np.zeros((vocab.embed_size, len(corpus)))
        for i, sentence in enumerate(corpus):
            tokens = tokenizer(sentence)
            sentence_embed = self.build_sentence_embed(tokens, vocab)
            sentence_embeddings[:, i] = sentence_embed
        U, _, __ = np.linalg.svd(sentence_embeddings, full_matrices=False)
        self.corpus_principles = U[:, : self.num_principles]
        with open(save_to, "wb") as f:
            pickle.dump(self.corpus_principles, f)
        return self.corpus_principles

    def load_corpus_principles(self, load_from: str):
        with open(load_from, "rb") as f:
            self.corpus_principles = pickle.load(f)

    def encode_text(
        self, text: str, vocab: BaseVocab, tokenizer: Tokenizer
    ) -> np.ndarray:
        tokens = tokenizer(text)
        sentence_embed = self.build_sentence_embed(tokens, vocab)
        sentence_embed = sentence_embed.reshape(1, -1)
        sentence_embed -= (
            sentence_embed @ self.corpus_principles @ self.corpus_principles.T
        )

        return np.ravel(sentence_embed)
