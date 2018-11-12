from constants import *
from data_processing.sql import *
from services import parallel

from typing import Tuple, List, Dict, Set
import collections as cols
from tqdm import tqdm
from glob import glob
import sqlite3
import pickle
import os

import nltk
from nltk.stem.snowball import EnglishStemmer



# class Index:
#     """ Inverted index data structure """
#     stemmer: EnglishStemmer
#     stopwords: Set[str]
#
#     token2id: Dict[str, int]
#     id2token: Dict[int, str]
#     title2id: Dict[str, int]
#     id2title: Dict[int, str]
#
#     unigram_index: Dict[Tuple[int], Set[int]]
#     bigram_index: Dict[Tuple[int], Set[int]]
#
#     def __init__(self, stemmer, stopwords):
#         """
#         :param stemmer: NLTK compatible stemmer
#         :param stopwords: set of ignored words
#         """
#         self.stemmer = stemmer
#         self.stopwords = set(stopwords)
#
#         self.unigram_index = cols.defaultdict(set)
#         self.bigram_index = cols.defaultdict(set)
#
#         self.title2id = cols.defaultdict(lambda: len(self.title2id))
#         self.id2title = dict()
#         self.token2id = cols.defaultdict(lambda: len(self.token2id))
#         self.id2token = dict()
#
#     def lookup(self, query: str):
#         """
#         Lookup a query in the index. Query is 1 or 2-word tuple for unigram or bigram lookup.
#         """
#         query = tuple(self.token2id.get(self.stemmer.stem(token.lower()), -1) for token in nltk.word_tokenize(query))
#         if len(query) == 2:
#             return self.bigram_index.get(query, None)
#         elif len(query) == 1:
#             return self.unigram_index.get(query, None)
#         else:
#             raise ValueError(f'Query has more than 2 non-stopwords {query}')
#
#     def add(self, title: str, document: List[List[str]], group: str):
#         """
#         Add a document to the index. Prepares documents by stemming, removing stopwords, and converting tokens
#         to ids. Creates a unigram and a bigram index.
#         """
#         tokens = self.__prepare_tokens(title, document)
#         self._add_unigram(tokens, title)
#         self._add_bigram(tokens, title)
#
#     def _add_unigram(self, tokens: List[int], title: str):
#         for token_id in tokens:
#             unigram: Tuple[int] = (token_id,)  # comma needed for tuple creation
#             document_id: int = self.title2id[title]
#             # if document_id not in self.unigram_index[unigram]:
#             self.unigram_index[unigram].add(document_id)
#
#     def _add_bigram(self, tokens: List[int], title: str):
#         for bigram in nltk.bigrams(tokens):
#             document_id: int = self.title2id[title]
#             # if document_id not in self.bigram_index[bigram]:
#             self.bigram_index[bigram].add(document_id)
#
#     def __prepare_tokens(self, title: str, document: List[List[str]]) -> List[int]:
#         # flatten document to list of words and join in string.
#         document = " ".join(it.chain(*document))
#         # tokenize, stem, and remove stopwords
#         tokens = [self.stemmer.stem(token.lower()) for token in nltk.word_tokenize(document) if
#                   token.lower() not in self.stopwords]
#
#         # update ids of titles and tokens if necessary
#         if title not in self.title2id:
#             self.id2title[self.title2id[title]] = title
#         for token in tokens:
#             if token not in self.token2id:
#                 self.id2token[self.token2id[token]] = token
#
#         # create the list of token ids.
#         return [self.token2id[token] for token in tokens]
#
#
# def build(group: str = None):
#     index = Index(EnglishStemmer(), nltk.corpus.stopwords.words('english'))
#     if group is not None:
#         file_paths = glob(os.path.join(PREPROCESSED_DATA_DIR, f'{group}.dict.tar'))
#     else:
#         file_paths = glob(os.path.join(PREPROCESSED_DATA_DIR, f'*.dict.tar'))
#
#     for file_path in tqdm(file_paths):
#         with open(file_path, 'rb') as file:
#             articles = pickle.load(file)
#
#         for (title, article) in articles.items():
#             index.add(title, article, file_path.split('/')[-1].split('.')[0])
#
#     with open(INDEX_FILE, 'wb') as pkl:
#         index.unigram_index = dict(index.unigram_index)
#         index.bigram_index = dict(index.bigram_index)
#
#         index.title2id = dict(index.title2id)
#         index.token2id = dict(index.token2id)
#
#         pickle.dump(index, pkl)
#
#
# def load():
#     with open(INDEX_FILE, 'rb') as file:
#         index = pickle.load(file)
#
#     return index


stemmer = EnglishStemmer()
stopwords = nltk.corpus.stopwords.words('english')

token2id: Dict[str, int] = cols.defaultdict(lambda: len(token2id))
id2token: Dict[int, str] = dict()
title2id: Dict[str, int] = cols.defaultdict(lambda: len(title2id))
id2title: Dict[int, str] = dict()

unigram_index: Dict[Tuple[int], Set[int]] = cols.defaultdict(set)
bigram_index: Dict[Tuple[int], Set[int]] = cols.defaultdict(set)


def build_mappings():
    with sqlite3.connect(PREPROCESSED_DB) as conn:
        cursor = conn.cursor()
        cursor.arraysize = 10000

        cursor.execute(COUNT_ALL)
        count = cursor.fetchone()

        with tqdm(total=count[0]) as progress:
            cursor.execute(RETRIEVE_ALL)
            paged_results = cursor.fetchmany()
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
        cursor.arraysize = 10000

        cursor.execute(COUNT_ALL)
        count = cursor.fetchone()

        with tqdm(total=count[0]) as progress:
            cursor.execute(RETRIEVE_ALL)
            paged_results = cursor.fetchmany()

            while paged_results:
                for (title, tokens) in paged_results:
                    tokens = pickle.loads(tokens)
                    token_ids = __prepare_tokens(title, tokens)

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


def __prepare_tokens(self, tokens: List[str]) -> List[int]:
    # create the list of token ids.
    return [self.token2id[token] for token in tokens]
