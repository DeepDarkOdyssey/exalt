from typing import Iterable, Union, List, TypeVar
from copy import copy
import re

Node = TypeVar("TrieNode")


class TrieNode(object):
    def __init__(self, key: str = ""):
        self.key = key
        self.next = {}
        self.is_word = False

    @property
    def is_leaf(self):
        return len(self.next) == 0

    def insert(self, word: str):
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
            if char not in node.next:
                return False
            node = node.next[char]
            nodes.append(node)
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
                removed_word.insert(0, nodes[-1].next.pop(node.key).key)
        return removed_word

    def show(self, word: str = ""):
        if self.is_word:
            print(word)

        for node in self.next.values():
            node.show(word + node.key)


class ACNode(TrieNode):
    def __init__(self, key: str = "", depth: int = 0):
        super().__init__(key)
        self.depth = depth
        self.fail = None

    def insert(self, word: str):
        curr = self
        for char in word:
            if char not in curr.next:
                curr.next[char] = ACNode(char, curr.depth + 1)
            curr = curr.next[char]
        curr.is_word = True


class ACAutomaton(object):
    def __init__(self, words: Iterable[str]):
        self.root = ACNode()
        self.root.insert_many(words)
        self.add_fails()

    def add_fails(self):
        queue = []
        for node in self.root.next.values():
            node.fail = self.root
            queue.append(node)

        while len(queue) > 0:
            curr: ACNode = queue.pop(0)
            fail_to = curr.fail
            for key, node in curr.next.items():
                while True:
                    if fail_to is not None and key in fail_to.next:
                        node.fail = fail_to.next[key]
                        break
                    elif fail_to is None:
                        node.fail = self.root
                        break
                    else:
                        fail_to = fail_to.fail
                queue.append(node)

    def search(self, target: str):
        result = []
        curr = self.root
        i = 0
        while i < len(target):
            char = target[i]
            if char in curr.next:
                node = curr.next[char]
                if node.is_word:
                    result.append((i - node.depth + 1, i))
                elif node.fail.is_word:
                    result.append((i - node.fail.depth + 1, i))
                curr = node
                i += 1
            else:
                if curr.fail is None:
                    curr = self.root
                    i += 1
                else:
                    curr = curr.fail
        return result

    def skip_search(self, target: str, skip_pattern:str='\s', max_skip=3):
        result = []
        curr = self.root
        i = 0
        num_skips = 0
        while i < len(target):
            char = target[i]
            if num_skips > max_skip:
                num_skips = 0
                curr = self.root
            if re.match(skip_pattern, char):
                num_skips += 1
                i += 1
            elif char in curr.next:
                node = curr.next[char]
                if node.is_word:
                    result.append((i - num_skips - node.depth + 1, i))
                elif node.fail.is_word:
                    result.append((i - num_skips - node.fail.depth + 1, i))
                curr = node
                i += 1
            else:
                if curr.fail is None:
                    curr = self.root
                    num_skips = 0
                    i += 1
                else:
                    curr = curr.fail
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
    # words = "banana bananas bandana band apple all beast".split()
    # root = TrieNode()
    # root.insert_many(words)
    # root.show()
    # root.show_tree()
    # print('*' * 50)
    # root.delete('bandana')
    # root.show()
    import time
    words = ["abd", "abdk", "abchijn", "chnit", "ijabdf", "ijaij"]
    automaton = ACAutomaton(words)
    # automaton.root.show()
    target = "abchnijab dfk"
    for start, end in automaton.skip_search(target):
        print(target[start : end + 1])

