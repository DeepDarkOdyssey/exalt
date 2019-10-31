from typing import List
import numpy as np


class DCT(object):
    def __init__(self, K: int = 2):
        self.K = K

    def _transform(self, vector: np.ndarray) -> np.ndarray:
        N = vector.size
        coefficients = []
        for k in range(N):
            if k == 0:
                c = np.math.sqrt(1 / N) * np.sum(vector)
            else:
                cosine_weights = np.cos((np.arange(N) + 0.5) * k * np.math.pi / N)
                c = np.math.sqrt(2 / N) * np.sum(vector * cosine_weights)
            coefficients.append(c)
        return np.array(coefficients)

    def pool_embeds(self, token_embeddings: np.ndarray):
        transformed_embeds = np.apply_along_axis(
            self._transform, axis=0, arr=token_embeddings
        )
        N = transformed_embeds.shape[0]
        if N < self.K:
            np.concatenate(
                (
                    transformed_embeds,
                    np.zeros((self.K - N, transformed_embeds.shape[1])),
                )
            )
        pooled_vector = np.ravel(transformed_embeds[: self.K])
        return pooled_vector


if __name__ == "__main__":
    from pyltp import Segmentor
    from text_encoder.helpers import StopWords
    from text_encoder.embed_loader import EmbedLoaderV2
    
    sentences, labels = [], []
    with open('/home/data_normal/focus/SharedFolder/DataFiles/mic_showroom_knowledge_base.txt') as f:
        for line in f:
            sentence, label = line.rstrip().split('\t')
            sentences.append(sentence)
            labels.append(label)

    stopwords = StopWords('/home/data_normal/focus/yuanminglei/general-text-encoder/assets/chinese_stopwords.txt')
    # embed_loader = EmbedLoader('/home/data_normal/focus/SharedFolder/Models/word-embeddings/Refined_Tencent_Embedding.txt',)
    embed_loader = EmbedLoaderV2('/home/data_normal/focus/SharedFolder/Models/word-embeddings/Refined_Tencent_Embedding.txt', oov_handler='rand')

    segmentor = Segmentor()
    segmentor.load('./assets/ltp_data_v3.4.0/cws.model')
    tokenizer = lambda text: segmentor.segment(text)
    # preprocess = lambda text: stopwords.strip(tokenizer(text))
    preprocess = lambda text: tokenizer(text)
    argsort = lambda seq: [i for v, i in sorted((v, i) for i, v in enumerate(seq))]

    dct = DCT()
    sent_embeds = []
    for sentence in sentences:
        token_embeds = []
        for token in preprocess(sentence):
            token_embed = embed_loader[token]
            token_embeds.append(token_embed)
        sent_embed = dct.pool_embeds(np.vstack(token_embeds))
        if len(sent_embed) != 400:
            print(sentence, len(sent_embed))
        sent_embeds.append(sent_embed)
    target_array = np.vstack(sent_embeds)
    target_array /= np.linalg.norm(target_array, axis=0, keepdims=True)
    print(embed_loader.oov_rand_embeds.keys())
    print('LEGO' in embed_loader)


    while True:
        query = input('Enter the query: ')
        if query == 'exit':
            break
        processed_query = list(preprocess(query))
        print('processed query: ', processed_query)
        for token in processed_query:
            if token not in embed_loader:
                print(token)
        query_array = np.vstack([embed_loader[token] for token in processed_query])
        query_vec = dct.pool_embeds(query_array)
        query_vec /= np.linalg.norm(query_vec)
        for i in np.argsort(target_array @ query_vec)[::-1][:5]:
            print(sentences[i])
