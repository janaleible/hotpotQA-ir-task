import os
from typing import List

import nltk
from nltk import AlignedSent
from tqdm import tqdm
import dill as pickle # use dill rather than pickle for defaultdicts

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from retrieval.term.dataset import Dataset
from services import parallel

import main_constants as constants


def _build_bitext(datachunk) -> List[AlignedSent]:
    bitext = []

    chunk_id, data = datachunk

    for question in tqdm(data):
        (gold1, gold2) = question.gold_articles

        gold_doc1 = ''.join(question.context[gold1])
        gold_doc2 = ''.join(question.context[gold2])

        tokenized_question = nltk.word_tokenize(question.question)
        tokenized_doc_1 = nltk.word_tokenize(gold_doc1)
        tokenized_doc_2 = nltk.word_tokenize(gold_doc2)

        bitext.append(AlignedSent(tokenized_question, tokenized_doc_1))
        bitext.append(AlignedSent(tokenized_question, tokenized_doc_2))

    return bitext


class IBM1FeatureExtractor(FeatureExtractor):

    ibm1: nltk.IBMModel1
    normalized: bool

    @property
    def feature_name(self) -> List[str]:
        return [f'{ "n" if self.normalized else "" }ibm1']

    def __init__(self, normalized: bool):
        super().__init__(None)

        self.normalized = normalized

        if os.path.isfile(constants.IBM_MODEL):
            with open(constants.IBM_MODEL, 'rb') as file:
                self.ibm1 = pickle.load(file)
        else:
            dataset = Dataset.from_file(constants.TRAIN_HOTPOT_SET)

            bitext = []
            batches = parallel.chunk(constants.CHUNK_SIZE, dataset.questions)
            # for partial_bitext in map(_build_bitext, batches):
            for partial_bitext in parallel.execute(_build_bitext, batches):
                bitext.extend(partial_bitext)

            self.ibm1 = nltk.IBMModel1(bitext, 5)

            os.makedirs(constants.FEATURE_EXTRACTION_DIR, exist_ok=True)
            with open(constants.IBM_MODEL, 'wb') as file:
                pickle.dump(self.ibm1, file)

    def extract(self, question: str, doc: str):

        tokenized_question = nltk.word_tokenize(question)
        tokenized_doc = nltk.word_tokenize(doc)

        aligned_sentence = AlignedSent(tokenized_question, tokenized_doc)
        self.ibm1.align(aligned_sentence)

        probability = 1
        for alignment in aligned_sentence.alignment:
            word = aligned_sentence.words[alignment[0]]
            mot = None if alignment[1] is None else aligned_sentence.mots[alignment[1]]
            probability *= self.ibm1.translation_table[word][mot]

        if self.normalized:
            probability /= len(tokenized_question)

        return probability

if __name__ == '__main__':

    FE = IBM1FeatureExtractor(False)

    feature = FE.extract(
        'Were Scott Derrickson and Ed Wood of the same nationality?',
        'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood\'s life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
    )

    print(feature)