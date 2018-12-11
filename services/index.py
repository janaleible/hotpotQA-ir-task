import os
import string
import pyndri
import pickle
import nltk
from unidecode import unidecode

from main_constants import TITLE2WID, WID2TITLE, INDRI_INDEX_DIR, EOP, EOS, INDRI_PARAMETERS, WID2INT, INT2WID
from typing import Dict, List, Tuple, Union, Set
from xml.etree import ElementTree
from datetime import datetime
from services import helpers


class Tokenizer:
    stopwords: Union[Set, None]

    def __init__(self) -> None:

        tree = ElementTree.parse(INDRI_PARAMETERS)
        if INDRI_PARAMETERS.split('/')[-1] == 'indri_stop_stem.xml':
            helpers.log('Loading stopwords.')
            stopwords = set()
            for elem in tree.find('stopper').iter('word'):
                stopwords.add(elem.text)
            self.stopwords = frozenset(stopwords)
        elif INDRI_PARAMETERS.split('/')[-1] == 'index.xml':
            self.stopwords = None
        else:
            raise NotImplementedError(f'Unknown index setting: {INDRI_PARAMETERS.split("/")[-1]}')
        self._punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def normalize(self, s: str) -> str:
        """Translate non-ascii characters to ascii and remove any punctuation."""
        s = unidecode(s)
        s = s.translate(self._punctuation)
        s = s.lower()

        return s

    def tokenize(self, s: str, keep_stopwords=True) -> List[str]:
        normalized = self.normalize(s)
        if not keep_stopwords and self.stopwords is not None:
            tokenized = [token for token in nltk.word_tokenize(normalized) if token not in self.stopwords]
        else:
            tokenized = [token for token in nltk.word_tokenize(normalized)]

        return tokenized


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

    title2wid: Dict[str, int]
    wid2title: Dict[int, str]
    int2wid: Dict[int, int]
    wid2int: Dict[int, int]

    tokenizer: Tokenizer

    def __enter__(self, **kwargs):
        idx = self.__init__()

        return idx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.index.close()
        del self

    def __init__(self, env: str = 'default', verbose: bool = False):
        if verbose:
            helpers.log(f'Loading index {INDRI_INDEX_DIR} with {env} query environment.')
        start = datetime.now()

        self.index = pyndri.Index(f'{INDRI_INDEX_DIR}')
        self.token2id, self.id2token, self.id2df = self.index.get_dictionary()
        self.id2tf = self.index.get_term_frequencies()

        self.tokenizer = Tokenizer()

        if os.path.isfile(TITLE2WID):
            with open(TITLE2WID, 'rb') as file:
                self.title2wid = pickle.load(file)

        if os.path.isfile(WID2TITLE):
            with open(WID2TITLE, 'rb') as file:
                self.wid2title = pickle.load(file)
        try:
            if os.path.isfile(WID2INT):
                with open(WID2INT, 'rb') as file:
                    self.wid2int = pickle.load(file)

            if os.path.isfile(INT2WID):
                with open(INT2WID, 'rb') as file:
                    self.int2wid = pickle.load(file)
        except FileNotFoundError:
            helpers.log('ID mappings do not exist yet. Not loaded.')

        if env == 'default':
            self.env = pyndri.QueryEnvironment(self.index)
        elif env == 'tfidf':
            self.env = pyndri.TFIDFQueryEnvironment(self.index, k1=1.2, b=0.75)
        elif env == 'prf':
            env = pyndri.QueryEnvironment(self.index)
            self.env = pyndri.PRFQueryEnvironment(env, fb_docs=10, fb_terms=10)
        else:
            raise ValueError(f'Unknown environment configuration {env}')

        stop = datetime.now()
        if verbose:
            helpers.log(f'Loaded index in {stop - start}.')

    def unigram_query(self, text: str, request: int = 5000) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram text search."""
        return list(self.env.query(f'{self.normalize(text)}', results_requested=request))

    def bigram_query(self, first: str, second: str, request: int = 5000) -> List[Tuple[int, float]]:
        """Retrieve documents according to bigram full text search."""
        return list(self.env.query(f'#1({self.normalize(first)} {self.normalize(second)})', results_requested=request))

    def title_lookup(self, title: str) -> List[Tuple[int, float]]:
        """Retrieve documents according to unigram title only search."""
        return list(self.env.query(f'{title}.title'))

    def document_int_ids(self) -> Tuple[str, Tuple[int]]:
        """Generator over the documents in the index."""
        for idx in range(self.index.document_base(), self.index.maximum_document()):
            yield idx
        return

    def get_wid(self, int_id) -> int:
        return int(self.index.ext_document_id(int_id))

    def count(self) -> int:
        return self.index.document_count()

    def internal2external(self, internal: int) -> int:
        """Find an external id given and internal one."""
        return self.int2wid[internal]

    def external2internal(self, external: int) -> int:
        """Find an external id given and internal one."""
        return self.wid2int[external]

    def tokenize(self, s: str, keep_stopwords=True) -> List[str]:
        """Tokenize the string in a list of normalized, lower-cased words."""
        return self.tokenizer.tokenize(s, keep_stopwords)

    def normalize(self, s: str) -> str:
        """Translate non-ascii characters to ascii and remove any punctuation."""
        return self.tokenizer.normalize(s)

    def doc_str(self, doc_tokens: Tuple[int], include_stop: bool, format_paragraph: bool) -> str:
        """Reproduce the stemmed document stored by indri as a string.

        :param doc_tokens: The document tokens
        :param include_stop: Whether to include stop words.
        :param format_paragraph: Whether to format according to original paragraph delimitation.
        """

        if include_stop:
            doc_str = " ".join([self.id2token.get(tid, '<STOP>') for tid in doc_tokens])
        else:
            doc_str = " ".join([self.id2token.get(tid, -1) for tid in doc_tokens if self.id2token.get(tid, -1) != -1])
        if format_paragraph:
            doc_str = doc_str.replace(EOP.strip(), '\n\n').replace(EOS.strip(), '. ')

        return doc_str

    def get_document_by_title(self, title: str) -> Tuple[int, ...]:
        external_id = self.title2wid[title]
        internal_id = self.external2internal(external_id)
        document = self.index.document(internal_id)[1]

        return document

    def get_pretty_document_by_title(self, title: str) -> str:
        external_id = self.title2wid[title]
        internal_id = self.external2internal(external_id)
        document = self.index.document(internal_id)

        return self.doc_str(document[1], include_stop=True, format_paragraph=False)

    def get_document_by_int_id(self, doc_int_id: int) -> Tuple[int, ...]:
        return self.index.document(doc_int_id)[1]
