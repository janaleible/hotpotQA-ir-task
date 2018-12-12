import re

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services.index import Index
from typing import Any, Dict, List
import main_constants as ct
import collections as cl
import numpy as np
import spacy


class EntityExtractor(FeatureExtractor):
    feature_name = ['entity_match_PER', 'entity_match_LOC', 'entity_match_ORG', 'entity_match_MISC']

    entity_model: Any
    e2i: Dict[str, int] = ct.E2I
    i2e: Dict[int, str] = ct.I2E

    def __init__(self, index: Index):
        super().__init__(index)

        # load spacy entity recognizer trained on wiki data.
        self.entity_model = spacy.load('xx_ent_wiki_sm')

    def extract(self, question: str, doc: str):
        qu_ents = self._get_entities(question)
        doc_ents = self._get_entities(doc)

        entity_scores = self._score_entities(qu_ents, doc_ents)

        return entity_scores.tolist()

    def _get_entities(self, text: str) -> Dict[str, List[str]]:
        parse = self.entity_model(text)

        ents = cl.defaultdict(list)
        for ent in parse.ents:
            ents[ent.label_].append(ent.text)

        return dict(ents)

    def _score_entities(self, qu_ents: Dict[str, List[str]], doc_ents: Dict[str, List[str]]):
        regex = re.compile("\ +")
        ent_type_word_matches = np.zeros(len(self.e2i), dtype=np.float)
        doc_ent_type_word_count = np.zeros(len(self.e2i), dtype=np.float)
        for _type, _qu_ents in qu_ents.items():
            _id = self.e2i[_type]
            # a bit unnatural but loop first over the documents so I can get the count of words without double-counting.
            for _doc_ents in doc_ents.get(_type, []):
                for doc_ent_word in regex.split(_doc_ents.strip()):
                    doc_ent_word = doc_ent_word.lower()
                    # count the words in the document entities
                    doc_ent_type_word_count[_id] += 1
                    for _q_ent in _qu_ents:
                        for question_entity_word in regex.split(_q_ent.strip()):
                            question_entity_word = question_entity_word.lower()
                            ent_type_word_matches[_id] += int(question_entity_word == doc_ent_word)
                            doc_ent_word = doc_ent_word.lower()
        # do a safe division where 0s remain 0s.
        # Because of the way it is built, if the doc_count is 0 also the matches are also 0.
        return np.divide(ent_type_word_matches, doc_ent_type_word_count, where=doc_ent_type_word_count != 0)
