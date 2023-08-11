#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict, OrderedDict
import os
from typing import Any, Callable, Dict, Iterable, List, Set


def namespace_match(pattern: str, namespace: str):
    """
    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self,
                 non_padded_namespaces: Set[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {padding_token: 0, oov_token: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {0: padding_token, 1: oov_token},
                                                       lambda: {})


DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = '[PAD]'
DEFAULT_OOV_TOKEN = '[UNK]'
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


class Vocabulary(object):
    def __init__(self,
                 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 ):
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = 'tokens') -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            return self._token_to_index[namespace][self._oov_token]

    def get_token_from_index(self, index: int, namespace: str = 'tokens'):
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])

    def save_to_files(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', encoding='utf-8') as f:
            for namespace_str in self._non_padded_namespaces:
                f.write('{}\n'.format(namespace_str))

        for namespace, token_to_index in self._token_to_index.items():
            filename = os.path.join(directory, '{}.txt'.format(namespace))
            with open(filename, 'w', encoding='utf-8') as f:
                for token, _ in token_to_index.items():
                    f.write('{}\n'.format(token))

    @classmethod
    def from_files(cls, directory: str) -> 'Vocabulary':
        with open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', encoding='utf-8') as f:
            non_padded_namespaces = [namespace_str.strip() for namespace_str in f]

        vocab = cls(non_padded_namespaces=non_padded_namespaces)

        for namespace_filename in os.listdir(directory):
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            if namespace_filename.startswith("."):
                continue
            namespace = namespace_filename.replace('.txt', '')
            if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                is_padded = False
            else:
                is_padded = True
            filename = os.path.join(directory, namespace_filename)
            vocab.set_from_file(filename, is_padded, namespace=namespace)

        return vocab

    def set_from_file(self,
                      filename: str,
                      is_padded: bool = True,
                      oov_token: str = DEFAULT_OOV_TOKEN,
                      namespace: str = "tokens"
                      ):
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}

        with open(filename, 'r', encoding='utf-8') as f:
            index = 1 if is_padded else 0
            for row in f:
                token = str(row).strip()
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
                index += 1

    def convert_tokens_to_ids(self, tokens: List[str], namespace: str = "tokens"):
        result = list()
        for token in tokens:
            idx = self._token_to_index[namespace].get(token)
            if idx is None:
                idx = self._token_to_index[namespace][self._oov_token]
            result.append(idx)
        return result

    def convert_ids_to_tokens(self, ids: List[int], namespace: str = "tokens"):
        result = list()
        for idx in ids:
            idx = self._index_to_token[namespace][idx]
            result.append(idx)
        return result

    def pad_or_truncate_ids_by_max_length(self, ids: List[int], max_length: int, namespace: str = "tokens"):
        pad_idx = self._token_to_index[namespace][self._padding_token]

        length = len(ids)
        if length > max_length:
            result = ids[:max_length]
        else:
            result = ids + [pad_idx] * (max_length - length)
        return result


def demo1():
    import jieba

    vocabulary = Vocabulary()
    vocabulary.add_token_to_namespace('白天', 'tokens')
    vocabulary.add_token_to_namespace('晚上', 'tokens')

    text = '不是在白天, 就是在晚上'
    tokens = jieba.lcut(text)

    print(tokens)

    ids = vocabulary.convert_tokens_to_ids(tokens)
    print(ids)

    padded_idx = vocabulary.pad_or_truncate_ids_by_max_length(ids, 10)
    print(padded_idx)

    tokens = vocabulary.convert_ids_to_tokens(padded_idx)
    print(tokens)
    return


if __name__ == '__main__':
    demo1()
