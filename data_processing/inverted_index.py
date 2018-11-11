from constants import *

from typing import Tuple, List, Dict, Set
import collections as cols
import itertools as it
from tqdm import tqdm
from glob import glob
import pickle
import json
import os

import nltk
from nltk.stem.snowball import EnglishStemmer


class Index:
    """ Inverted index data structure """
    stemmer: EnglishStemmer
    stopwords: Set[str]

    token2id: Dict[str, int]
    id2token: Dict[int, str]
    title2id: Dict[str, int]
    id2title: Dict[int, str]

    index: Dict[Tuple[int], List[int]]

    def __init__(self, stemmer, stopwords):
        """
        :param stemmer: NLTK compatible stemmer
        :param stopwords: set of ignored words
        """
        self.stemmer = stemmer
        self.stopwords = set(stopwords)

        self.unigram_index = cols.defaultdict(set)
        self.bigram_index = cols.defaultdict(lset)

        self.title2id = cols.defaultdict(lambda: len(self.title2id))
        self.id2title = dict()
        self.token2id = cols.defaultdict(lambda: len(self.token2id))
        self.id2token = dict()

    def lookup(self, query: str):
        """
        Lookup a query in the index. Query is 1 or 2-word tuple for unigram or bigram lookup.
        """
        query = tuple(self.token2id.get(self.stemmer.stem(token.lower()), -1) for token in nltk.word_tokenize(query))
        if len(query) == 2:
            return self.bigram_index.get(query, None)
        elif len(query) == 1:
            return self.unigram_index.get(query, None)
        else:
            raise ValueError(f'Query has more than 2 non-stopwords {query}')

    def add(self, title: str, document: List[List[str]]):
        """
        Add a document to the index. Prepares documents by stemming, removing stopwords, and converting tokens
        to ids. Creates a unigram and a bigram index.
        """
        tokens = self.__prepare_tokens(title, document)
        self._add_unigram(tokens, title)
        self._add_bigram(tokens, title)

    def _add_unigram(self, tokens: List[int], title: str):
        for token_id in tokens:
            unigram: Tuple[int] = (token_id,)  # comma needed for tuple creation
            document_id: int = self.title2id[title]
            # if document_id not in self.unigram_index[unigram]:
            self.unigram_index[unigram].add(document_id)

    def _add_bigram(self, tokens: List[int], title: str):
        for bigram in nltk.bigrams(tokens):
            document_id: int = self.title2id[title]
            # if document_id not in self.bigram_index[bigram]:
            self.bigram_index[bigram].add(document_id)

    def __prepare_tokens(self, title: str, document: List[List[str]]) -> List[int]:
        # flatten document to list of words and join in string.
        document = " ".join(it.chain(*document))
        # tokenize, stem, and remove stopwords
        tokens = [self.stemmer.stem(token.lower()) for token in nltk.word_tokenize(document) if
                  token.lower() not in self.stopwords]

        # update ids of titles and tokens if necessary
        if title not in self.title2id:
            self.id2title[self.title2id[title]] = title
        for token in tokens:
            if token not in self.token2id:
                self.id2token[self.token2id[token]] = token

        # create the list of token ids.
        return [self.token2id[token] for token in tokens]


def build():
    index = Index(EnglishStemmer(), nltk.corpus.stopwords.words('english'))

    file_paths = glob(os.path.join(PREPROCESSED_DATA_DIR, '*.dict.tar'))
    for file_path in tqdm(file_paths):
        with open(file_path, 'r') as file:
            articles = json.load(file)

        for (title, article) in articles.items():
            index.add(title, article)
            index.add(title, article)

    with open(INDEX_FILE, 'wb') as pkl:
        index.unigram_index = dict(index.unigram_index)
        index.bigram_index = dict(index.bigram_index)

        index.title2id = dict(index.title2id)
        index.token2id = dict(index.token2id)

        pickle.dump(index, pkl)


def load():
    with open(INDEX_FILE, 'rb') as file:
        index = pickle.load(file)

    return index


if __name__ == '__main__':
    build()
