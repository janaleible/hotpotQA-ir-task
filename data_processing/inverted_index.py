from constants import *
from data_processing.sql import *

from typing import Tuple, List, Dict, Set
import collections as cols
from tqdm import tqdm
import sqlite3
import pickle
import os

import nltk
from nltk.stem.snowball import EnglishStemmer


class Index(object):
    stemmer: EnglishStemmer
    stopwords: Set[str]

    token2id: Dict[str, int]
    id2token: Dict[int, str]
    title2id: Dict[str, int]
    id2title: Dict[int, str]

    unigram_index: Dict[Tuple[int], Set[int]]
    bigram_index: Dict[Tuple[int], Set[int]]

    def __init__(self):
        self.stemmer = stemmer
        self.stopwords = set(stopwords)

        with open(UNIGRAM_INDEX, 'rb') as file:
            self.unigram_index = pickle.load(file)
        with open(BIGRAM_INDEX, 'rb') as file:
            self.bigram_index = pickle.load(file)

        with open(TOKEN2ID, 'rb') as file:
            self.token2id = pickle.load(file)
        with open(ID2TOKEN, 'rb') as file:
            self.id2token = pickle.load(file)
        with open(TITLE2ID, 'rb') as file:
            self.title2id = pickle.load(file)
        with open(ID2TITLE, 'rb') as file:
            self.id2title = pickle.load(file)


stemmer = EnglishStemmer()
stopwords = nltk.corpus.stopwords.words('english')

token2id: Dict[str, int] = cols.defaultdict(lambda: len(token2id))
id2token: Dict[int, str] = dict()
title2id: Dict[str, int] = cols.defaultdict(lambda: len(title2id))
id2title: Dict[int, str] = dict()

unigram_index: Dict[Tuple[int], Set[int]] = cols.defaultdict(set)
bigram_index: Dict[Tuple[int], Set[int]] = cols.defaultdict(set)


def load() -> Index:
    return Index()


def build_mappings():
    with sqlite3.connect(PREPROCESSED_DB) as conn:
        cursor = conn.cursor()
        cursor.arraysize = 1000

        cursor.execute(COUNT_ALL)
        count = cursor.fetchone()

        with tqdm(total=count[0]) as progress:
            cursor.execute(RETRIEVE_ALL)
            paged_results = cursor.fetchmany()
            progress.update(len(paged_results))
            while paged_results:
                for (title, tokens) in paged_results:
                    tokens = pickle.loads(tokens)

                    if title not in title2id:
                        id2title[title2id[title]] = title

                    for token in tokens:
                        if token not in token2id:
                            id2token[token2id[token]] = token

                paged_results = cursor.fetchmany()
                progress.update(len(paged_results))

    os.makedirs(MAPS_DIR, exist_ok=True)
    with open(TOKEN2ID, 'wb') as file:
        pickle.dump(dict(token2id), file)
    with open(ID2TOKEN, 'wb') as file:
        pickle.dump(dict(id2token), file)
    with open(TITLE2ID, 'wb') as file:
        pickle.dump(dict(title2id), file)
    with open(ID2TITLE, 'wb') as file:
        pickle.dump(dict(id2title), file)


def build_index():
    with sqlite3.connect(PREPROCESSED_DB) as conn:
        cursor = conn.cursor()
        cursor.arraysize = 1000

        cursor.execute(COUNT_ALL)
        count = cursor.fetchone()

        with tqdm(total=count[0]) as progress:
            cursor.execute(RETRIEVE_ALL)
            paged_results = cursor.fetchmany()
            progress.update(len(paged_results))
            while paged_results:
                for (title, tokens) in paged_results:
                    tokens = pickle.loads(tokens)
                    token_ids = __prepare_tokens(tokens)

                    _add_unigram(token_ids, title)
                    _add_bigram(token_ids, title)

                paged_results = cursor.fetchmany()
                progress.update(len(paged_results))

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(UNIGRAM_INDEX, 'wb') as file:
        pickle.dump(dict(unigram_index), file)
    with open(BIGRAM_INDEX, 'wb') as file:
        pickle.dump(dict(bigram_index), file)


def _add_unigram(token_ids: List[int], title: str):
    for token_id in token_ids:
        unigram: Tuple[int] = (token_id,)  # comma needed for tuple creation
        document_id: int = title2id[title]

        unigram_index[unigram].add(document_id)


def _add_bigram(token_ids: List[int], title: str):
    for bigram in nltk.bigrams(token_ids):
        document_id: int = title2id[title]

        bigram_index[bigram].add(document_id)


def __prepare_tokens(tokens: List[str]) -> List[int]:
    # create the list of token ids.
    return [token2id[token] for token in tokens]
