from typing import List
from unicodedata import *
from os.path import dirname, join
from pathlib import Path
from hashlib import md5
import re
import tempfile
import requests


class SepTokenizer(object):
    def __init__(self, seps: List[str] = ["\s"], keep_sep: bool = False):
        if keep_sep:
            self.pattern = "|".join(self.seps)
        else:
            self.pattern = f"({'|'.join(self.seps)})"

    def tokenize(self, text: str) -> List[str]:
        return re.split(pattern, text)


class ScriptTokenizer(object):
    def __init__(self):
        # Get local cache directory
        cache_dir: Path = Path.home().joinpath(".cache", "exalt")
        cache_dir.mkdir(exist_ok=True)

        # Get unicode scripts original file
        scripts_file = cache_dir.joinpath("unicode_scripts.txt")
        if not scripts_file.exists():
            print("No cached unicode scipts found, downloading from the Internet...")
            r = requests.get("http://www.unicode.org/Public/UNIDATA/Scripts.txt")
            with open(scripts_file, "wb") as f:
                f.write(r.content)
                print(f"Unicode scripts file has been cached to {scripts_file}")
        self.scripts = self.parse_unicode_scripts(scripts_file)

    @staticmethod
    def parse_unicode_scripts(file_path):
        result = {}
        with open(file_path) as f:
            for line in f:
                if not line.startswith(("#", "\n")):
                    line = line.rstrip()
                    code_points, properties = line.split(";")
                    script, general_category = properties.strip().split("#")
                    script = script.strip()
                    general_category = general_category[1:3]

                    if ".." in code_points:
                        start_point, end_point = code_points.rstrip().split("..")
                        for i in range(int(start_point, 16), int(end_point, 16) + 1):
                            if i in result:
                                raise ValueError()
                            result[i] = {"script": script, "gc": general_category}
                    else:
                        code_point = code_points.rstrip()
                        result[int(code_point, 16)] = {
                            "script": script,
                            "gc": general_category,
                        }
        return result
    
    def tokenize(self, text: str) -> List[str]:
        script_properties = [tuple(self.scripts[ord(char)].values()) for char in text]
        seps = map(lambda x: x[0]!=x[1], zip(script_properties[:-1], script_properties[1:]))
        tokens, token = [], []
        for i, sep in enumerate(seps):
            token.append(text[i])
            if sep:
                tokens.append(''.join(token))
                token = []
        token.append(text[-1])
        tokens.append(''.join(token))
        return tokens



if __name__ == "__main__":
    tokenizer = ScriptTokenizer()
    print(tokenizer.tokenize('我I 爱love 你you'))
