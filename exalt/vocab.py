from typing import Iterable, Optional, List, Union
from itertools import chain
from collections.abc import Sequence


class Vocab(object):
    def __init__(
        self,
        tokens: Iterable[str],
        unk: Optional[str] = "<UNK>",
        pad: Optional[str] = None,
    ):
        self.pad = pad
        self.unk = unk
        self._tokens, self._token2id = self.build(tokens, pad, unk)

    def build(
        self,
        tokens: Iterable[str],
        pad: Optional[str] = None,
        unk: Optional[str] = None,
    ):
        id2token = list(dict.fromkeys(tokens).keys())
        if pad:
            id2token.insert(0, pad)
        if unk:
            id2token.insert(0, unk)
        token2id = {token: i for i, token in enumerate(id2token)}
        return id2token, token2id

    def add(self, tokens: Iterable[str]):
        self._tokens, self._token2id = self.build(chain(self._tokens, tokens))

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._tokens

    def __iter__(self) -> str:
        return self._tokens

    def token2id(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError(
                "Argument `token` should be type `str`, "
                f"while input is {token} with type {type(token)}"
            )
        try:
            token_id = self._token2id[token]
        except KeyError:
            if self.unk:
                token_id = self.unk
            else:
                raise KeyError(
                    f"Vocab doesn't contain token {token}, "
                    "`UNK` token hasn't been specified"
                )
        return token_id

    def tokens2ids(self, tokens: Iterable[str]) -> List[str]:
        return [self.token2id[token] for token in tokens]

    def id2token(self, token_id: int) -> str:
        if not isinstance(token_id, int):
            raise TypeError(
                "Argument `token_id` should be type `int`, "
                f"while input is {token_id} with type {type(token_id)}"
            )
        if token_id < 0 or token_id > len(self):
            raise KeyError(
                f"Token id {token_id} does't in the correct range (0, {len(self)})"
            )
        return self._tokens[token_id]

    def ids2tokens(self, token_ids: Iterable[int]) -> List[str]:
        return [self.id2token(i) for i in token_ids]

    def __getitem__(self, key: Union[str, int, Iterable[str], Iterable[int]]):
        if isinstance(key, int):
            return self.id2token(key)
        elif isinstance(key, str):
            return self.token2id(key)
        elif isinstance(key, Sequence):
            try:
                result = self.ids2tokens(key)
            except TypeError:
                result = self.tokens2ids(key)
            finally:
                return result
        else:
            raise TypeError(f"Type {type(key)} of input key is not suitable")

