from abc import abstractmethod

from services.index import Index


class FeatureExtractor(object):
    index: Index
    feature_name: str

    def __init__(self, index: Index, feature_name: str):
        super().__init__()
        self.index = index
        self.feature_name = feature_name

    @abstractmethod
    def extract(self, question: str, doc: str):
        raise NotImplementedError
