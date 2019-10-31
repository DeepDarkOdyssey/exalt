import os
import io
from typing import List, Iterable, Callable, Optional
import jieba
import spacy
from pyltp import (
    Segmentor,
    Postagger,
    NamedEntityRecognizer,
    Parser,
    SementicRoleLabeller,
)

Tokenizer = Callable[[str], List[str]]


class JiebaTokenizer(object):
    def __init__(self, token_set: Optional[Iterable[str]]):
        if token_set:
            f = io.StringIO("\n".join(token_set))
            jieba.load_userdict(f)

    def __call__(self, text: str) -> List[str]:
        return jieba.lcut(text)


class LtpParser(object):
    def __init__(self, data_dir: str):
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(data_dir, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(data_dir, "pos.model"))
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(data_dir, "ner.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(data_dir, "parser.model"))
        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(data_dir, "pisrl.model"))

    def parse(self, text: str) -> List[str]:
        tokens = self.segmentor.segment(text)
        postags = self.postagger.postag(tokens)
        netags = self.recognizer.recognize(tokens, postags)
        arcs = self.parser.parse(tokens, postags)
        roles = self.labeller.label(tokens, postags, arcs)
        srlabels = {}
        for role in roles:
            srlabels[role.index] = {
                arg.name: {"start": arg.range.start, "end": arg.range.end}
                for arg in role.arguments
            }
        return {
            "tokens": list(tokens),
            "postags": list(postags),
            "netags": list(netags),
            "srlabels": srlabels,
        }

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()
        self.parser.release()
        self.labeller.release()


class SpacyParser(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def parse(self, text: str):
        doc = self.nlp(text)
        tokens, postags, netags, lemmas = [], [], [], []
        for token in doc:
            tokens.append(token.lower_)
            postags.append(token.tag_)
            if token.ent_type_:
                netag = f"{token.ent_iob_}-{token.ent_type_}"
            else:
                netag = token.ent_iob_
            netags.append(netag)
            lemmas.append(token.lemma_)
        return {
            "tokens": tokens,
            "postags": postags,
            "netags": netags,
            "lemmas": lemmas,
        }

    def release(self):
        pass