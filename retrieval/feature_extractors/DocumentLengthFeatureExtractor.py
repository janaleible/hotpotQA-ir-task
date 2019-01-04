from typing import List

from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services.index import Index


class DocumentLengthFeatureExtractor(FeatureExtractor):

    @property
    def feature_name(self) -> List[str]:
        return ['doclen']

    def __init__(self):
        super().__init__(None)

    def extract(self, question: str, doc: str):
        return len(doc.split(' '))


if __name__ == '__main__':
    FE = DocumentLengthFeatureExtractor()

    feature = FE.extract(
        'Were Scott Derrickson and Ed Wood of the same nationality?',
        'Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood\'s life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.'
    )

    print()
