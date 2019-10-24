from typing import List
from unicodedata import *
import re
import requests

UNICODE_SCRIPTS_URL = 'http://www.unicode.org/Public/UNIDATA/Scripts.txt'

class SepTokenizer(object):
    def __init__(self, seps: List[str] =['\s'], keep_sep:bool=False):
        if keep_sep:
            self.pattern = '|'.join(self.seps)
        else:
            self.pattern = f"({'|'.join(self.seps)})"
    
    def tokenize(self, text: str) -> List[str]:
        return re.split(pattern, text)


def parse_unicode_scripts():
    r = requests.get(UNICODE_SCRIPTS_URL)
    doc = r.text
    for line in doc.split('\n'):
        if not line.startswith((' ', '#')):
            print(line)



if __name__ == "__main__":
    parse_unicode_scripts()