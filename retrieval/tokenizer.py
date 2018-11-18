from main_constants import INDRI_PARAMETERS
from typing import Set, List, Dict
from xml.etree import ElementTree
from unidecode import unidecode
import string
import nltk


class Tokenizer:
    stopwords: Set[str]
    punctuation: Dict[int, int]

    def __init__(self) -> None:
        tree = ElementTree.parse(INDRI_PARAMETERS)

        stopwords = set()
        for elem in tree.find('stopper').iter('word'):
            stopwords.add(elem.text)
        self.stopwords = frozenset(stopwords)

        self.punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def tokenize(self, s: str) -> List[str]:
        normalized = self.normalize(s)
        tokenized = [token for token in nltk.word_tokenize(normalized) if token not in self.stopwords]
        return tokenized

    def normalize(self, s: str) -> str:
        s = unidecode(s)
        s = s.translate(self.punctuation)

        return s
