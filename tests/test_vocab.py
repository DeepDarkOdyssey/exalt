import pytest
from exalt.vocab import Vocab


def test_vocab_error_raise():
    vocab = Vocab([], unk=None)
    with pytest.raises(TypeError):
        vocab.token2id(10)
    with pytest.raises(KeyError):
        vocab.token2id('test')
    with pytest.raises(TypeError):
        vocab.id2token('test')
    with pytest.raises(KeyError):
        vocab.id2token(10)
    with pytest.raises(TypeError):
        vocab[{'test': 10}]
    
def test_vocab_functionallity():
    tokens = list(map(str, range(10)))
    vocab = Vocab(tokens)
    assert vocab.id2token(0) == '<UNK>'
    for token in tokens:
        assert vocab[token] == int(token) + 1
    assert vocab[range(1, 11)] == list(tokens)
    additional_tokens = list(map(str, range(10, 20)))
    vocab.add(additional_tokens)
    for token in map(str, range(20)):
        assert vocab[token] == int(token) + 1
