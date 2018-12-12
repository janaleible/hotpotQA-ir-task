from typing import List

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services.index import Index


class QuestionWordFeatureExtractor(FeatureExtractor):

    @property
    def feature_name(self) -> List[str]:
        return [f'q{qword}' for qword in self.question_words]

    def __init__(self):
        super().__init__(None)

        self.question_words = ['what', 'which', 'who', 'in', 'are', 'is', 'was', 'when', 'where', 'how']

    def extract(self, question: str, doc: str):

        question_word = question.split(' ')[0].lower()

        return [int(word == question_word) for word in self.question_words]


if __name__ == '__main__':

    FE = QuestionWordFeatureExtractor()

    feature = FE.extract(
        'What Scott Derrickson and Ed Wood of the same nationality?',
        'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood\'s life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
    )

    print(feature)