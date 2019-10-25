from typing import List, Union
import re


class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.is_leaf = False
        self.fail = None

    def insert(self, words: Union[List[str], str]):
        current_node = self
        for word in words:
            if word not in current_node.children:
                current_node.children[word] = TrieNode()
            current_node = current_node.children[word]
        current_node.is_leaf = True

    def print(self, prefix: str = '', root: str = ''):
        if root:
            print(root)
        for i, (word, node) in enumerate(self.children.items()):
            print(prefix + '+-- ' + word)
            if i == len(self.children) - 1:
                node.print(prefix=prefix + '    ')
            else:
                node.print(prefix=prefix + '|   ')


class ACNode(TrieNode):
    def __init__(self, depth: int = 0):
        super().__init__()
        self.depth = depth
    
    def insert(self, words: Union[List[str], str]):
        current_node = self
        for word in words:
            if word not in current_node.children:
                current_node.children[word] = ACNode(current_node.depth+1)
            current_node = current_node.children[word]
        current_node.is_leaf = True


class ACAutomation(object):
    def __init__(self, words_list: Union[List[str], List[List[str]]]):
        self.root = ACNode()
        for words in words_list:
            self.root.insert(words)
        for node in self.root.children.values():
            node.fail = self.root
        self.add_fails()

    def add_fails(self):
        queue = list(self.root.children.values())
        while len(queue) > 0:
            node = queue.pop(0)
            for word, child in node.children.items():
                fail_to = node.fail
                while True:
                    if word in fail_to.children:
                        child.fail = fail_to.children[word]
                        break
                    elif fail_to is self.root:
                        child.fail = self.root
                        break
                    else:
                        fail_to = fail_to.fail
                queue.append(child)

    def search(self, source: Union[str, List[str]]):
        result = []
        t_pointer = self.root
        for s_pointer in range(len(source)):
            word = source[s_pointer]
            while (t_pointer.fail is not None) and (word not in t_pointer.children):
                t_pointer = t_pointer.fail
            if word in t_pointer.children:
                t_pointer = t_pointer.children[word]
            if t_pointer.is_leaf:
                result.append((s_pointer - t_pointer.depth + 1, s_pointer))
        return result


def filter_by_keywords(sample, automation):
    name_result = automation.search(sample['name'])
    desc_result = automation.search(sample['desc'])
    if len(name_result) > 0 & len(desc_result) > 0:
        return True
    else:
        return False


def filter_by_dots(sample, *args, **kwargs):
    r = re.fullmatch(r'\.+', sample['desc'])
    if r:
        return True
    else:
        return False

def filter_by_desc_length(sample, *args, **kwargs):
    if len(sample['desc']) < 50 or len(sample['desc']) > 2000:
        return True
    else:
        return False

if __name__ == "__main__":
    import json
    import jieba
    import pandas as pd

    with open('server/saved_model/keywords.json', encoding='utf8') as f:
        keywords = json.load(f)
    
    ac_automation = ACAutomation([jieba.lcut(keyword) for keyword in keywords['detection']])
    ac_automation.root.print()

    df = pd.read_csv('data/chemical_industry/test.tsv', sep='\t', encoding='utf8')
    # for i, desc in enumerate(df['描述']):
    #     result = ac_automation.search(desc)
    #     if result:
    #         print(result)
    #     for keyword in keywords['detection']:
    #         if keyword in desc:
    #             print(i)
    desc = df.iloc[3365]['描述']

    print(jieba.lcut(desc))
    # for keyword in keywords['detection']:
    #     if keyword in desc:
    #         print(keyword)
    
    print(ac_automation.search(jieba.lcut(desc)))