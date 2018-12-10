import nltk
from multiset import Multiset

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services.index import Index, Tokenizer


class BigramOverlapFeatureExtractor(FeatureExtractor):

    def __init__(self, index: Index, feature_name: str):
        super().__init__(index, feature_name)

        self.tokenizer = Tokenizer()

    def extract(self, question: str, doc: str) -> int:

        tokenized_question = self.tokenizer.tokenize(question)
        tokenized_doc = self.tokenizer.tokenize(doc)

        question_bigrams = Multiset(nltk.bigrams(tokenized_question))
        doc_bigrams = Multiset(nltk.bigrams(tokenized_doc))

        overlap = sum(question_bigrams.intersection(doc_bigrams).values())

        return overlap

if __name__ == '__main__':

    # index = Index()

    FE = BigramOverlapFeatureExtractor(None, '')

    feature = FE.extract(
        'Were Scott Derrickson and Ed Wood of the same nationality?',
        'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood\'s life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
    )

    print()