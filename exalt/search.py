from typing import Iterable, Union, List, TypeVar
import re

Node = TypeVar('Node')


class TrieNode(object):
    def __init__(self, key: str = ""):
        self.key = key
        self.next = {}
        self.is_word = False

    @property
    def is_leaf(self):
        return len(self.next) == 0

    def insert(self, word: Iterable[str]):
        node = self
        for char in word:
            if char not in node.next:
                node.next[char] = TrieNode(char)
            node = node.next[char]
        node.is_word = True

    def insert_many(self, words: Iterable[str]):
        for word in words:
            self.insert(word)

    def find(self, target: str):
        node = self
        for char in target:
            if char not in node.next:
                return False
            node = node.next[char]
        if node.is_word:
            return "FullMatch"
        else:
            return "Partial"

    def delete(self, target: str):
        nodes = []
        node = self
        for char in target:
            nodes.append(node)
            if char not in node.next:
                return False
            node = node.next[char]
        node = nodes.pop(-1)
        if not node.is_word:
            return False
        else:
            node.is_word = False

        removed_word = []
        while True:
            try:
                node = nodes.pop(-1)
            except IndexError:
                break
            if node.is_leaf:
                removed_word.insert(nodes[-1].next.pop(node.key).key)
        return removed_word

    @staticmethod
    def show(node: Node, word: str = ''):
        if node.is_word:
            print(word)

        for key, node in node.next.items():
            TrieNode.show(node, word + key)

    def show_tree(self, prefix: str = ''):
        # for i, (word, node) in enumerate(self.next.items()):
        #     print(prefix + '+-- ' + word)
        #     if i == len(self.next) - 1:
        #         node.show_tree(prefix=prefix + '    ')
        #     else:
        #         node.show_tree(prefix=prefix + '|   ')
        node = self
        print(node.key)
        for node in node.next:
            print(node)
    
    def __repr__(self):
        repr_string = f'{self.key}'
        if not self.is_leaf:
            for i, node in enumerate(self.next):
                if i == 0:
                    repr_string += f' ┳━━━━ {node}\n'
                elif i == len(self.next) - 1:
                    repr_string += f'  ┗━━━━ {node}\n'
                else:
                    repr_string += f'  ┣━━━━ {node}\n'
        return repr_string

class ACNode(TrieNode):
    def __init__(self, depth: int = 0):
        super().__init__()
        self.depth = depth

    def insert(self, words: Union[List[str], str]):
        current_node = self
        for word in words:
            if word not in current_node.children:
                current_node.children[word] = ACNode(current_node.depth + 1)
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


def brute_force(target: str, pattern: str) -> int:
    for i in range(len(target)):
        j = 0
        while j < len(pattern):
            if pattern[j] == target[i]:
                i += 1
                j += 1
            else:
                break
        if j == len(pattern):
            return i - j


class KMP(object):
    def __init__(self, pattern: str):
        next_array = [-1] * len(pattern)
        i, j = 0, -1
        while i < len(pattern) - 1:
            if j == -1 or pattern[i] == pattern[j]:
                i += 1
                j += 1
                next_array[i] = j
            else:
                j = next_array[j]
        self.pattern = pattern
        self.next = next_array
        print(self.next)

    def match(self, target: str) -> int:
        i, j = 0, 0
        while i < len(target) and j < len(self.pattern):
            if j == -1 or target[i] == self.pattern[j]:
                i += 1
                j += 1
            else:
                j = self.next[j]
        if j == len(self.pattern):
            return i - j


if __name__ == "__main__":
    words = "banana bananas bandana band apple all beast".split()
    root = TrieNode()
    root.insert_many(words)
    print(repr(root.next['b']))
    # root.show_tree()
    # TrieNode.show(root)
    # print('*' * 50)
    # root.delete('bandana')
    # TrieNode.show(root)
