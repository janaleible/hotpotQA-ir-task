import os
import json
from collections import defaultdict
from glob import glob
import pickle

import nltk
from nltk.stem.snowball import EnglishStemmer


class Index:
    """ Inverted index data structure """

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopwords = set(stopwords)

        self.index = defaultdict(list)

        self._title2id = defaultdict(lambda: len(self._title2id))
        self._id2title = dict()
        self._token2id = defaultdict(lambda: len(self._token2id))
        self._id2token = dict()

    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        word = self.stemmer.stem(word)

        return self.index.get(self._token2id.get(word, -1), None)

    def add(self, document, title):
        """
        Add a document string to the index
        """
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue

            token = self.stemmer.stem(token)

            if token not in self._token2id:
                self._id2token[self._token2id[token]] = token
            if title not in self._title2id:
                self._id2title[self._title2id[title]] = title

            self.index[self._token2id[token]].append(self._title2id[title])


def build():
    index = Index(nltk.word_tokenize,
                  EnglishStemmer(),
                  nltk.corpus.stopwords.words('english'))
    files = glob(os.path.join('..', 'data', 'preprocessed', '*.json'))

    for file in files:
        with open(file, 'r') as f:
            articles = json.load(f)

        for (title, article) in articles.items():
            index.add(" ".join([item for sublist in article for item in sublist]), title)

    with open(os.path.join('..', 'data', 'preprocessed', 'index.tar'), 'wb') as pkl:
        index.index = dict(index.index)
        index._title2id = dict(index._title2id)
        index._token2id = dict(index._token2id)

        pickle.dump(index, pkl)


if __name__ == '__main__':
    # build()
    with open('../data/preprocessed/index.tar', 'rb') as f:
        idx = pickle.load(f)

    pass
