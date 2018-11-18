from main_constants import TITLE2WID, WID2TITLE, INDRI_INDEX_DIR, EOP, EOS
from retrieval.tokenizer import Tokenizer
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
    tokenizer: Tokenizer

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
        self.tokenizer = Tokenizer()

        stop = datetime.now()
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Loaded index in {stop - start}.]')

    def bigram_lookup(self, first: str, second: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to bigram full text search."""
        return self.index.query(f'#1({self.tokenizer.normalize(first)} {self.tokenizer.normalize(second)})',
                                results_requested=65500)

    def unigram_lookup(self, first: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram text search."""
        return self.index.query(f'{self.tokenizer.normalize(first)}', results_requested=65500)

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
