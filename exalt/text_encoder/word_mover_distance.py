# from itertools import product
# from collections import defaultdict
# from typing import Iterable
# import time
# import numpy as np
# import pulp
# from scipy.spatial.distance import euclidean
# from text_encoder.embed_loader import EmbedLoader

# class StopWords(object):
#     def __init__(self, file_path: str):
#         self._stopwords = set()
#         with open(file_path) as f:
#             for line in f:
#                 self._stopwords.add(line.rstrip())
    
#     def strip(self, tokens: Iterable[str]) -> Iterable[str]:
#         for token in tokens:
#             if token not in self._stopwords:
#                 yield token


# def tokens_to_fracdict(tokens):
#     cntdict = defaultdict(lambda : 0)
#     for token in tokens:
#         cntdict[token] += 1
#     totalcnt = sum(cntdict.values())
#     return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}


# def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wordvecs, lpFile=None):
#     all_tokens = list(set(first_sent_tokens+second_sent_tokens))

#     first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
#     second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

#     T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

#     prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
#     prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
#                         for token1, token2 in product(all_tokens, all_tokens)])
#     for token2 in second_sent_buckets:
#         prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
#     for token1 in first_sent_buckets:
#         prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets])==first_sent_buckets[token1]

#     if lpFile!=None:
#         prob.writeLP(lpFile)

#     prob.solve()

#     return prob

# if __name__ == "__main__":
#     from text_encoder.nlp_tools import LtpParser
#     ltp_parser = LtpParser('./assets/ltp_data_v3.4.0')
#     tokenizer = lambda text: ltp_parser.parse(text)['tokens']
#     stopwords = StopWords('./assets/chinese_stopwords.txt')
#     refined_tencent_embed = EmbedLoader(
#         "/home/data_normal/focus/SharedFolder/Models/word-embeddings/Refined_Tencent_Embedding.txt"
#     )
#     test_strings = [
#         '横幅的尺寸是多少',
#         # '横幅的尺寸是多少',
#         '我想问一下横幅的尺寸是多少'
#     ]
#     for text in test_strings:
#         tokens = tokenizer(text)
#         print(tokens)
#         print(list(stopwords.strip(tokens)))
#         for token in tokens:
#             if token not in refined_tencent_embed:
#                 print(token)

#     tick = time.time()
#     prob = word_mover_distance_probspec(tokenizer(test_strings[0]), tokenizer(test_strings[1]), refined_tencent_embed)
#     pulp.value(prob.objective)
#     tock = time.time()
#     print(tock - tick)


# %%
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/home/data_normal/focus/SharedFolder/Models/word-embeddings/Refined_Tencent_Embedding.txt')

# %%
import time
import re
import jieba
from pyltp import Segmentor
from typing import Iterable
import numpy

# %%
class StopWords(object):
    def __init__(self, file_path: str):
        self._stopwords = set()
        with open(file_path) as f:
            for line in f:
                self._stopwords.add(line.rstrip())

    def strip(self, tokens: Iterable[str]) -> Iterable[str]:
        for token in tokens:
            if token not in self._stopwords and not re.match('\s', token):
                yield token

stopwords = StopWords('./assets/chinese_stopwords.txt')
# %%
corpus = {}
sentences, labels = [], []
with open('/home/data_normal/focus/SharedFolder/DataFiles/mic_showroom_knowledge_base.txt') as f:
    for line in f:
        sentence, label = line.rstrip().split('\t')
        sentences.append(sentence)
        labels.append(label)

# %%
segmentor = Segmentor()
segmentor.load('./assets/ltp_data_v3.4.0/cws.model')
tokenizer = lambda text: segmentor.segment(text)
# preprocess = lambda text: stopwords.strip(tokenizer(text))
preprocess = lambda text: tokenizer(text)
argsort = lambda seq: [i for v, i in sorted((v, i) for i, v in enumerate(seq))]

# %%
file_path = './tmp/WMD_with_refined_tencent_vocab_jieba_tokenizer_on_mic_showroom_errors.txt'
f = open(file_path, "w")
top1_count, top5_count = 0, 0
time_intervals = []
for i in range(len(sentences)):
    tick = time.time()
    query = list(preprocess(sentences[i]))
    label = labels[i]
    distances = []
    for j in range(len(sentences)):
        dist = model.wmdistance(query, list(preprocess(sentences[j])))
        if i == j:
            try:
                assert dist == 0
            except AssertionError:
                dist = 0
        distances.append(dist)

    sorted_distances = argsort(distances)
    tock = time.time()
    time_intervals.append(tock - tick)
    try:
        assert i in sorted_distances[:7]
    except AssertionError:
        print(sentences[i], query)
        badcase = sentences[sorted_distances[0]]
        print(badcase, list(preprocess(badcase)))
        raise

    pred_ids = []
    iter_sort = iter(sorted_distances)
    while len(pred_ids) < 6:
        pred_id = next(iter_sort)
        if pred_id != i:
            pred_ids.append(pred_id)
    preds = [labels[n] for n in pred_ids]
    if preds[0] == label:
        top1_count += 1
        top5_count += 1
    elif label in preds[1:]:
        top5_count += 1
        error_message = (
            f"In top 5 not top1:\nQuest: {sentences[i]}\tLabel: {labels[i]}\t{list(preprocess(sentences[i]))}"
        )
        f.write(error_message + "\n")
        for sid in sorted_distances[2:6]:
            error_message = f"Res: {sentences[sid]}\tLabel: {labels[sid]}\t{list(preprocess(sentences[sid]))}"
            f.write(error_message + "\n")
    else:
        error_message = (
            f"Not in top 5:\nQuest: {sentences[i]}\tLabel: {labels[i]}\t{list(preprocess(sentences[i]))}"
        )
        f.write(error_message + "\n")
        for sid in sorted_distances[2:6]:
            error_message = f"Res: {sentences[sid]}\tLabel: {labels[sid]}\t{list(preprocess(sentences[sid]))}"
            f.write(error_message + "\n")

print(f"TOP1: {top1_count / len(labels)}")
print(f"TOP5: {top5_count / len(labels)}")
print(f'Average time cost by each query: {sum(time_intervals)/len(time_intervals)}s')
