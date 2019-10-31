from typing import Iterable
from collections import Counter
import numpy as np


class Corpora(object):
    def __init__(self, documents: Iterable[Iterable[str]]):
        self.documents = documents
        self.token_counter = Counter()
        self.doc_counter = Counter()
        self.num_tokens, self.num_docs = 0, 0
        self._get_stats()

    def idf(self, token: str):
        if token not in self:
            raise KeyError(f"Token `{token}` is not in corpora!")
        return np.log10(
            self.num_docs / (self.doc_counter[token] + 1)
        )  # Avoid zero divide

    def _get_stats(self):
        for doc in self.documents:
            self.add(doc)

    def add(self, document: Iterable[str]):
        # TODO: append document to self
        tokens = list(document)
        self.num_tokens += len(tokens)
        self.num_docs += 1
        self.token_counter.update(tokens)
        self.doc_counter.update(set(tokens))

    def bow(self, document: Iterable[str], tfidf_weight: False) -> dict:
        token_counter = Counter(document)
        num_tokens = sum(token_counter.values())
        vec = []
        for token in self.doc_counter:
            weight = token_counter.get(token, 0)
            if tfidf_weight:
                tf = weight / num_tokens
                idf = self.idf(token)
                weight = tf * idf
            vec.append(weight)
        return np.array(vec)
    
    def __iter__(self) -> Iterable[Iterable[str]]:
        return iter(self.documents)

    def __contains__(self, token: str) -> bool:
        return token in self.doc_counter
    
    def __len__(self) -> int:
        return self.num_docs

