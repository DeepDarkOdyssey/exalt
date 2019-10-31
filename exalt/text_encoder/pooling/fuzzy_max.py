# REF: http://arxiv.org/abs/1904.13264

from typing import Optional, List
from collections import Counter
import numpy as np
from text_encoder.vocab import BaseVocab, Tokenizer


def fuzzy_bow(
    token_embeddings: np.ndarray,
    universe_matrix: np.ndarray,
    token_count: Optional[List[int]] = None,
) -> np.ndarray:
    """构建Fuzzy Bag of Words

    使用 universe_matrix 对 token_embeddings 进行线性变换，即计算每个 token_embedding 和
    universe_matrix 每一行的內积，得到向量即为paper中的 membership 向量。对所有的 memebership
    向量进行 max_pooling 得到最终的 sentence embedding
    
    Args:
        token_embeddings (np.ndarray): token 词向量构成的[N*d]的矩阵，每一行为一个token
        universe_matrix (np.ndarray): [k*d]的universe矩阵，每一行代表一个向量，通常为全部词向量矩阵
    
    Returns:
        np.ndarray: 长度为d的文本向量
    """
    if token_count:
        token_embeddings = np.array(token_count).reshape((-1, 1)) * token_embeddings
    sentence_embedding = token_embeddings @ universe_matrix.T
    sentence_embedding = np.vstack(
        (sentence_embedding, np.zeros(sentence_embedding.shape[1]))
    )
    sentence_embedding = np.max(sentence_embedding, axis=0)
    return sentence_embedding


def fuzzy_jaccard(sent_embed1: np.ndarray, sent_embed2: np.ndarray) -> float:
    """Fuzzy Set的jaccard相似度计算
    
    Args:
        sent_embed1 (np.ndarray): 第一个句子对应的embedding向量
        sent_embed2 (np.ndarray): 第二个句子对应的embedding向量
    
    Returns:
        float: 两个句子的 jaccard 相似度
    """
    return np.sum(np.fmin(sent_embed1, sent_embed2)) / np.sum(
        np.fmax(sent_embed1, sent_embed2)
    )


def dynamax(
    sentence1: str, sentence2: str, vocab: BaseVocab, tokenizer: Tokenizer
) -> float:
    """计算两个句子的 Dynamic Max 相似度
    
    相当于拼接两个句子的 token_embeddings 构成 universe matrix，用其计算两个对应的 Fuzzy SoW，
    再计算两个 fuzzy SoW 之间的 fuzzy jaccard 相似度

    Args:
        sentence1 (str): 第一个句子（文本）
        sentence2 (str): 第二个句子（文本）
        vocab (BaseVocab): 获取词向量用的词典类
        tokenizer (Tokenizer): 分词器
    
    Returns:
        float: 两个句子的相似度, 0 ~ 1
    """
    tokens1 = tokenizer(sentence1)
    tokens2 = tokenizer(sentence2)
    bow1 = Counter(tokens1)
    # print(bow1)
    bow2 = Counter(tokens2)
    # print(bow2)
    # token_embeddings1 = vocab.text2embed(sentence1, tokenizer)
    # token_embeddings2 = vocab.text2embed(sentence2, tokenizer)
    token_embeddings1 = np.vstack([vocab.token2embed(token) for token in bow1])
    token_count1 = [bow1[token] for token in bow1]
    token_embeddings2 = np.vstack([vocab.token2embed(token) for token in bow2])
    token_count2 = [bow2[token] for token in bow2]
    universe_matrix = np.vstack((token_embeddings1, token_embeddings2))

    sentence_embed1 = fuzzy_bow(token_embeddings1, universe_matrix, token_count1)
    sentence_embed2 = fuzzy_bow(token_embeddings2, universe_matrix, token_count2)
    return fuzzy_jaccard(sentence_embed1, sentence_embed2)
