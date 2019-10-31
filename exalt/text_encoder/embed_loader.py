from functools import lru_cache
from pathlib import Path
from hashlib import md5
from typing import Dict, Union, Optional
from text_encoder.utils import get_related_cache_path
from os.path import getsize, exists, basename
import pickle
import time
import numpy as np
import spacy


class EmbedLoader(object):
    def __init__(
        self,
        file_path: str,
        file_type: str = "w2v",
        cache_size: int = 300000,
        refresh_cache: bool = False,
        ignore_errors: bool = True,
        verbose:int=100000  #TODO: add verbose control
    ):
        self.file_path = file_path
        self.file_type = file_type
        self.cache_size = cache_size
        self.refresh_cache = refresh_cache
        self.ignore_errors = ignore_errors

        self.cache_decorator = lru_cache(cache_size)
        cached_info = self.warm_start()
        self.num_tokens = cached_info["num_tokens"]
        self.embed_size = cached_info["embed_size"]
        self.avg_embedding = cached_info["avg_embedding"]
        self._token_positions = cached_info["token_positions"]
        self.tokens = list(self._token_positions.keys())

    def _load_embed_from_disk(self, token: str) -> np.ndarray:
        if token not in self:
            return None
        with open(self.file_path) as f:
            f.seek(self._token_positions[token])
            embedding = np.array(f.readline().rstrip().split()[-self.embed_size: ], np.float)
            return embedding

    def warm_start(self) -> Dict[str, Union[int, np.ndarray, Dict[str, int]]]:
        tick = time.time()
        cache_path = get_related_cache_path(self.file_path)
        if exists(cache_path) & (not self.refresh_cache):
            print(f"Pre-loading {basename(cache_path)} embedding infos...")
            with open(cache_path, "rb") as f:
                cached_info = pickle.load(f)

        else:
            print("Loading embedding files...")
            with open(self.file_path) as f:
                num_tokens, embed_size = tuple(map(int, f.readline().rstrip().split()))
                avg_embedding = np.zeros(embed_size)
                token_positions = dict()
                count = 0
                while True:
                    count += 1
                    if count % 100000 == 0:
                        print(f"Reading line {count}/{num_tokens}")
                    current_position = f.tell()
                    line = f.readline().rstrip()
                    if line == "":
                        break
                    items = line.split(' ')
                    token = " ".join(items[: -int(embed_size)])
                    embedding = np.array(items[-embed_size:], dtype=np.float)
                    try:
                        assert token not in token_positions, f'{token} aleady cached, line {count}'
                    except AssertionError as e:
                        if self.ignore_errors:
                            count -= 1
                            num_tokens -= 1
                            continue
                        else:
                            raise e
                    avg_embedding += embedding

                    token_positions[token] = current_position
                avg_embedding /= count - 1
            assert (
                len(token_positions) == num_tokens  # Sanity Check
            ), f"Specified {num_tokens} tokens, read {len(token_positions)} tokens instead."

            cached_info = {
                "num_tokens": num_tokens,
                "embed_size": embed_size,
                "avg_embedding": avg_embedding,
                "token_positions": token_positions,
            }

            print("Saving embedding infos to cache...")
            with open(cache_path, "wb") as f:
                pickle.dump(cached_info, f)
        tock = time.time()
        print(f"Warm start finished. Time cost: {tock-tick:.4f}s.")
        return cached_info

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key: str) -> np.ndarray:
        return self.cache_decorator(self._load_embed_from_disk)(key)

    def __contains__(self, key: str) -> bool:
        return key in self._token_positions

    def __iter__(self):
        return iter(self._token_positions)


class SpacyEmbedLoader(object):
    def __init__(self, cache_size: int = 300000):
        self.cache_decorator = lru_cache(cache_size)
        self.nlp = spacy.load("en_core_web_md")
        self.vocab = self.nlp.vocab
        self.embed_size = self.vocab.vectors_length
        self.tokens = list(self.vocab.strings)

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, key: str) -> np.ndarray:
        # if self.vocab.has_vector(key):
        #     return self.vocab.get_vector(key)
        # else:
        #     return None
        doc = self.nlp(key)
        return doc.vector

    def __contains__(self, key: str) -> bool:
        return key in self.vocab

    def __iter__(self):
        return iter(self.vocab)


class EmbedLoaderV2(object):
    def __init__(
        self,
        file_path: str,
        refresh_cache: bool = False,
        lru_size: int = 300000,
        oov_handler: Optional[str] = None,
        ignore_overlap: bool = True,
        verbose:int=100000  #TODO: add verbose control
    ):
        self.file_path = Path(file_path).absolute()
        assert self.file_path.exists()  # Sanity Check
        path_hash_val = md5(str(file_path).encode()).hexdigest()

        cache_dir = Path.home().joinpath('.cache', 'text_encoder')
        cache_dir.mkdir(exist_ok=True)
        cache_name = f'{Path(self.file_path).name}_{path_hash_val[:8]}_cache.pkl'
        self.cache_path = cache_dir.joinpath(cache_name)

        self.lru_size = lru_size
        self.refresh_cache = refresh_cache
        self.ignore_overlap = ignore_overlap
        self.verbose = verbose

        self.oov_handler = oov_handler
        self.oov_rand_embeds = {}

        cached_info = self.warm_start()
        self.num_tokens = cached_info["num_tokens"]
        self.embed_size = cached_info["embed_size"]
        self.avg_embedding = cached_info["avg_embedding"]
        self._token_positions = cached_info["token_positions"]
        self.tokens = list(self._token_positions.keys())

    def _load_embed_from_disk(self, token: str) -> np.ndarray:
        if token not in self:
            return None
        with open(self.file_path) as f:
            f.seek(self._token_positions[token])
            embedding = np.array(f.readline().rstrip().split()[-self.embed_size: ], np.float)
            return embedding
    
    def warm_start(self) -> Dict[str, Union[int, np.ndarray, Dict[str, int]]]:
        tick = time.time()
        if self.cache_path.exists() and not self.refresh_cache:
            print(f"Pre-loading {self.cache_path.name} embedding infos...")
            with open(self.cache_path, "rb") as f:
                cached_info = pickle.load(f)

        else:
            print("Loading embedding files...")
            with open(self.file_path) as f:
                num_tokens, embed_size = tuple(map(int, f.readline().rstrip().split()))
                avg_embedding = np.zeros(embed_size)
                token_positions = dict()
                count = 0
                while True:
                    count += 1
                    if self.verbose > 0 and count % self.verbose == 0:
                        print(f"Reading line {count}/{num_tokens}")
                    current_position = f.tell()
                    line = f.readline().rstrip()
                    if line == "":
                        break
                    items = line.split(' ')
                    token = " ".join(items[: -int(embed_size)])
                    embedding = np.array(items[-embed_size:], dtype=np.float)
                    try:
                        assert token not in token_positions, f'{token} aleady cached, line {count}'
                    except AssertionError as e:
                        if self.ignore_overlap:
                            count -= 1
                            num_tokens -= 1
                            continue
                        else:
                            raise e
                    avg_embedding += embedding

                    token_positions[token] = current_position
                avg_embedding /= count - 1
            assert (
                len(token_positions) == num_tokens  # Sanity Check
            ), f"Specified {num_tokens} tokens, read {len(token_positions)} tokens instead."

            cached_info = {
                "num_tokens": num_tokens,
                "embed_size": embed_size,
                "avg_embedding": avg_embedding,
                "token_positions": token_positions,
            }

            print(f"Saving embedding infos to {self.cache_path}...")
            with open(self.cache_path, "wb") as f:
                pickle.dump(cached_info, f)
        tock = time.time()
        print(f"Warm start finished. Time cost: {tock-tick:.4f}s.")
        return cached_info
    
    def _get_embed(self, key: str) -> Optional[np.ndarray]:
        if key in self:
            return self._load_embed_from_disk(key)
        elif self.oov_handler is None:
            return None
        elif self.oov_handler == 'avg':
            return self.avg_embedding
        elif self.oov_handler == 'zero':
            return np.zeros((self.embed_size,))
        elif self.oov_handler == 'rand':
            rand_vec = np.random.rand(self.embed_size)
            self.oov_rand_embeds[key] = rand_vec
            return rand_vec
        else:
            raise ValueError(f"`oov_handler` {self.oov_handler} isn't in ('avg', 'rand', 'zero')")
    
    def __getitem__(self, key: str) -> Optional[np.ndarray]:
        return lru_cache(self.lru_size)(self._get_embed)(key)

    def __len__(self) -> int:
        return self.num_tokens

    def __contains__(self, key: str) -> bool:
        return key in self._token_positions

    def __iter__(self):
        return iter(self._token_positions)