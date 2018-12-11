import nltk
from multiset import Multiset

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services.index import Index, Tokenizer


class BigramOverlapFeatureExtractor(FeatureExtractor):

    normalized: bool

    @property
    def feature_name(self) -> str:
        return f'BigramOverlap{ "Normalized" if self.normalized else "Unnormalized" }'

    def __init__(self, normalized: bool):
        super().__init__(None)

        self.normalized = normalized
        self.tokenizer = Tokenizer()

    def extract(self, question: str, doc: str) -> float:

        tokenized_question = self.tokenizer.tokenize(question)
        tokenized_doc = self.tokenizer.tokenize(doc)

        question_bigrams = Multiset(nltk.bigrams(tokenized_question))
        doc_bigrams = Multiset(nltk.bigrams(tokenized_doc))

        overlap = sum(question_bigrams.intersection(doc_bigrams).values())

        if self.normalized:
            overlap /= len(tokenized_question)

        return overlap


if __name__ == '__main__':

    FE = BigramOverlapFeatureExtractor(False)

    feature = FE.extract(
        'Were Scott Derrickson and Ed Wood of the same nationality?',
        'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood\'s life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
    )

    print()