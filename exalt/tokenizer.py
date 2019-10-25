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
        cache_dir = Path('~/.cache').mkdir( exist_ok=True)

        # Get unicode scripts original file
        scripts_file = Path(join(cache_dir, 'unicode_scripts.txt'))
        if not scripts_file.exists():
            print('No cached unicode scipts found, downloading from the Internet...')
            r = requests.get("http://www.unicode.org/Public/UNIDATA/Scripts.txt")
            with open(scripts_file, 'wb') as f:
                f.write(r.content)
                print(f'Unicode scripts file has been cached to {scripts_file}')

    def parse_unicode_scripts():
        UNICODE_SCRIPTS_URI = join(dirname(__file__), "assets/Scripts.txt")
        result = {}
        with tempfile.NamedTemporaryFile(
            suffix=".cache", prefix="unicode_scripts", dir="~/.cache", delete=False
        ) as f:
            r = requests.get("http://www.unicode.org/Public/UNIDATA/Scripts.txt")

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


if __name__ == "__main__":
    tokenizer = ScriptTokenizer()

