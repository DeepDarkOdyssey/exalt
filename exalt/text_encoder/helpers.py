from typing import Iterable

class StopWords(object):
    def __init__(self, file_path: str):
        self._stopwords = set()
        with open(file_path) as f:
            for line in f:
                self._stopwords.add(line.rstrip())

    def strip(self, tokens: Iterable[str]) -> Iterable[str]:
        for token in tokens:
            if token not in self._stopwords:
                yield token