import logging
import string
from datetime import datetime

import nltk
from nltk import StemmerI

from constants import TITLE2WID, WID2TITLE, INDRI_INDEX_DIR, EOP, EOS
from typing import Dict, List, Set, Tuple
from xml.etree import ElementTree
import pickle
import pyndri

logging.basicConfig(level='INFO')


class Index(object):
    """Wrapper around an Indri index.
    title2wid: mapping from title to Wikipedia id.
    wid2title: mapping from Wikipedia id.

    stemmer: An NLTK stemmer according to the one used by Indri. Implementation differences possible. Use with care.
    stopwords: The stopwords specified in the indri index build parameters as a frozenset.
    """

    index: pyndri.Index
    token2id: Dict[str, int]
    id2token: Dict[int, str]
    id2df: Dict[int, int]
    id2tf: Dict[int, int]

    title2wid: Dict[str, List[int]]
    wid2title: Dict[int, str]

    stemmer: StemmerI
    stopwords: Set[int]

    def __init__(self):
        self.punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        start = datetime.now()

        self.index = pyndri.Index(INDRI_INDEX_DIR)

        self.token2id, self.id2token, self.id2df = self.index.get_dictionary()
        self.id2tf = self.index.get_term_frequencies()

        with open(TITLE2WID, 'rb') as file:
            self.title2wid = pickle.load(file)
        with open(WID2TITLE, 'rb') as file:
            self.wid2title = pickle.load(file)

        tree = ElementTree.parse('build_indri_index.xml')
        stemmer: str = tree.find('stemmer').find('name').text
        if stemmer == 'porter':
            self.stemmer = nltk.stem.porter.PorterStemmer()
        stopwords = set()
        for elem in tree.find('stopper').iter('word'):
            stopwords.add(elem.text)
        self.stopwords = frozenset(stopwords)

        stop = datetime.now()
        logging.info(f'Loaded index from {INDRI_INDEX_DIR} with {stemmer.capitalize()} stemmer in {stop - start}.')

    def bigram_lookup(self, first: str, second: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to bigram full text search."""
        return self.index.query(f'#1({first} {second})')

    def unigram_lookup(self, first: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram text search."""
        return self.index.query(f'{first}')

    def title_lookup(self, title) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram title only search."""
        return self.index.query(f'{title}.title')

    def documents(self) -> Tuple[str, Tuple[int]]:
        """Generator over the documents in the index."""
        for idx in range(self.index.document_base(), self.index.maximum_document()):
            yield self.index.document_base(idx)

    def inspect_document(self, doc: Tuple[str, Tuple[int]], include_stop: bool, format: bool) -> str:
        """Reproduce the stemmed document stored by indri as a string.

        :param doc: Whe document as retrieved from index.document(id)
        :param include_stop: Whether to include stop words.
        :param format: Whether to format according to original paragraph delimitation.
        """
        doc_id, doc_tokens = doc

        if include_stop:
            doc_str = " ".join([self.id2token.get(tid, '<STOP>') for tid in doc_tokens])
        else:
            doc_str = " ".join([self.id2token.get(tid, -1) for tid in doc_tokens if self.id2token.get(tid, -1) != -1])
        if format:
            doc_str = doc_str.replace(EOP, '\n\n').replace(EOS, '')

        return doc_str

    def _remove_punctuation(self, s: str):
        return s.translate(self.punctuation)


if __name__ == '__main__':
    index = Index()