import string

from unidecode import unidecode

from main_constants import TITLE2WID, WID2TITLE, INDRI_INDEX_DIR, EOP, EOS, INDRI_PARAMETERS
from typing import Dict, List, Tuple
from xml.etree import ElementTree
from datetime import datetime
from nltk import StemmerI
import logging
import pyndri
import pickle
import nltk
import os

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

    def __enter__(self, **kwargs):
        idx = self.__init__(**kwargs)

        return idx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.index.close()

    def __init__(self, load_indri_maps: bool = False, load_stemmer: bool = False):
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Loading index {INDRI_INDEX_DIR}]')
        start = datetime.now()

        self.index = pyndri.Index(f'{INDRI_INDEX_DIR}')
        if load_indri_maps:
            self.token2id, self.id2token, self.id2df = self.index.get_dictionary()
            self.id2tf = self.index.get_term_frequencies()
        with open(TITLE2WID, 'rb') as file:
            self.title2wid = pickle.load(file)
        with open(WID2TITLE, 'rb') as file:
            self.wid2title = pickle.load(file)
        if load_stemmer:
            tree = ElementTree.parse('build_indri_index.xml')
            stemmer: str = tree.find('stemmer').find('name').text
            if stemmer == 'porter':
                self.stemmer = nltk.stem.porter.PorterStemmer()
            else:
                raise ValueError('Unknown stemmer selected')

        tree = ElementTree.parse(INDRI_PARAMETERS)
        if INDRI_PARAMETERS.split('/')[-1] == 'indri_stop_stem.xml':
            logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Loading stopwords.]')
            stopwords = set()
            for elem in tree.find('stopper').iter('word'):
                stopwords.add(elem.text)
            self.stopwords = frozenset(stopwords)
        elif INDRI_PARAMETERS.split('/')[-1] == 'index.xml':
            logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Not loading stopwords.]')
        else:
            raise NotImplementedError(f'Unknown index setting: {INDRI_PARAMETERS.split("/")[-1]}')
        self.punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        stop = datetime.now()
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Loaded index in {stop - start}.]')

    def bigram_lookup(self, first: str, second: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to bigram full text search."""
        return self.index.query(f'#1({self.normalize(first)} {self.normalize(second)})',
                                results_requested=65500)

    def unigram_lookup(self, first: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram text search."""
        return self.index.query(f'{self.normalize(first)}', results_requested=65500)

    def title_lookup(self, title) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram title only search."""
        return self.index.query(f'{title}.title')

    def documents(self) -> Tuple[str, Tuple[int]]:
        """Generator over the documents in the index."""
        for idx in range(self.index.document_base(), self.index.maximum_document()):
            yield self.index.document_base(idx)

    def internal2external(self, internal: int) -> int:
        return int(self.index.document(internal)[0])

    def external2internal(self, external: int) -> int:
        return self.index.document_ids([str(external)])[0][1]

    def tokenize(self, s: str) -> List[str]:
        normalized = self.normalize(s)
        if INDRI_PARAMETERS.split('/')[-1] == 'indri_stop_stem.xml':
            tokenized = [token.lower() for token in nltk.word_tokenize(normalized) if
                         token.lower() not in self.stopwords]
        else:
            tokenized = [token.lower() for token in nltk.word_tokenize(normalized)]

        return tokenized

    def normalize(self, s: str) -> str:
        s = unidecode(s)
        s = s.translate(self.punctuation)

        return s

    def inspect_document(self, doc: Tuple[str, Tuple[int]], include_stop: bool, format_paragraph: bool) -> str:
        """Reproduce the stemmed document stored by indri as a string.

        :param doc: Whe document as retrieved from index.document(id)
        :param include_stop: Whether to include stop words.
        :param format_paragraph: Whether to format according to original paragraph delimitation.
        """
        doc_id, doc_tokens = doc

        if include_stop:
            doc_str = " ".join([self.id2token.get(tid, '<STOP>') for tid in doc_tokens])
        else:
            doc_str = " ".join([self.id2token.get(tid, -1) for tid in doc_tokens if self.id2token.get(tid, -1) != -1])
        if format_paragraph:
            doc_str = doc_str.replace(EOP, '\n\n').replace(EOS, '')

        return doc_str
