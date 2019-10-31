from typing import List, Optional, Union, Iterable, Callable, Generator, Dict, overload
from collections import Counter, defaultdict
from text_encoder.embed_loader import EmbedLoader, SpacyEmbedLoader
from text_encoder.nlp_tools import Tokenizer
import numpy as np

np.random.seed(20190923)


class BaseVocab(object):
    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        documents: Optional[Iterable[List[str]]] = None,
        embed_loader: Optional[EmbedLoader] = None,
        unk: Optional[str] = None,
        pad: Optional[str] = None,
    ):
        if tokens is None and documents is None:
            raise ValueError("`tokens` and `documents` can't both be None")

        self.unk, self.pad = unk, pad

        if tokens:
            self.id2token = tokens
            if self.unk:
                self.id2token.insert(0, unk)
            if self.pad:
                self.id2token.insert(0, pad)

        if documents:
            num_tokens, num_docs = 0, 0
            token_counter = Counter()
            document_counter = Counter()
            for document in documents:
                num_tokens += len(document)
                num_docs += 1
                token_counter.update(document)
                document_counter.update(set(document))
            if tokens is None:
                self.id2token = [token for token, count in token_counter.most_common()]
                if self.unk:
                    self.id2token.insert(0, unk)
                if self.pad:
                    self.id2token.insert(0, pad)

            self.token_freq = lambda token: token_counter.get(token, 0) / num_tokens
            # Avoid zero divide
            self.idf = lambda token: np.log10(num_docs / document_counter.get(token, 1))

        if self.unk:
            self.token2id = defaultdict(lambda: self.id2token.index(unk))
            for i, token in enumerate(self.id2token):
                self.token2id[token] = i
        else:
            self.token2id = dict(((token, i) for i, token in enumerate(self.id2token)))

        if embed_loader:
            self._embed_size = embed_loader.embed_size
            # if unk:
            #     try:
            #         unk_embed = embed_loader.avg_embedding
            #     except AttributeError:
            #         # unk_embed = np.random.randn(embed_loader.embed_size)
            #         unk_embed = np.zeros(embed_loader.embed_size)

            # def _token2embed(token: str) -> np.ndarray:
            #     if pad and (token == pad):
            #         return np.zeros(embed_loader.embed_size)
            #     else:
            #         embedding = embed_loader[token]
            #         if embedding is not None:
            #             return embedding
            #         else:
            #             return unk_embed

            # self.token2embed = lambda token: _token2embed(token)
            self.token2embed = lambda token: embed_loader[token]
        else:
            self._embed_size = None

    @classmethod
    def create_from(cls, *args, **kwargs):
        raise NotImplementedError()

    def tokens2indexes(self, tokens: List[str]) -> List[int]:
        return [self.token2id[token] for token in tokens]

    def token2embed(self, token: str) -> np.ndarray:
        raise NotImplementedError("Token embedding hasn't been initialized!")

    def tokens2embeds(self, tokens: List[str]) -> np.ndarray:
        return np.vstack([self.token2embed(token) for token in tokens])

    def text2indexes(self, text: str, tokenizer: Tokenizer) -> List[int]:
        tokens = tokenizer(text)
        return self.tokens2indexes(tokens)

    def text2embed(self, text: str, tokenizer: Tokenizer) -> np.ndarray:
        tokens = tokenizer(text)
        return self.tokens2embeds(tokens)

    def token_freq(self, token: str) -> float:
        raise NotImplementedError("`documents` is not provided!")

    def idf(self, token: str) -> float:
        raise NotImplementedError("`documents` is not provided!")

    def __len__(self):
        return len(self.id2token)

    @property
    def embed_size(self):
        if self._embed_size:
            return self._embed_size
        else:
            raise AttributeError("Token embedding hasn't been initialized!")

    def __contains__(self, id_or_string: Union[str, int]):
        if isinstance(id_or_string, str):
            return id_or_string in self.token2id
        elif isinstance(id_or_string, int):
            return id_or_string < len(self)
        else:
            raise TypeError("Only support `str` and `int`")

    def __getitem__(self, id_or_string: Union[int, str]):
        if isinstance(id_or_string, int):
            return self.id2token[id_or_string]
        elif isinstance(id_or_string, str):
            return self.token2id[id_or_string]
        else:
            raise TypeError("Only support `str` and `int`")


class VocabFromEmbed(BaseVocab):
    @classmethod
    def create_from(cls, embed_loader: EmbedLoader) -> BaseVocab:
        return cls(
            tokens=embed_loader.tokens,
            embed_loader=embed_loader,
            unk="<UNK>",
            pad="<PAD>",
        )


class VocabFromCorpusWithEmbed(BaseVocab):
    @classmethod
    def create_from(
        cls, corpus: Iterable[List[str]], embed_loader: EmbedLoader
    ) -> BaseVocab:
        return cls(
            tokens=embed_loader.tokens,
            documents=corpus,
            embed_loader=embed_loader,
            unk="<UNK>",
            pad="<PAD>",
        )
