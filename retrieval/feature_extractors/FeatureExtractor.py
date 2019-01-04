from abc import abstractmethod
from typing import List

from services.index import Index


class FeatureExtractor(object):
    index: Index

    @property
    @abstractmethod
    def feature_name(self) -> List[str]:
        raise NotImplementedError

    def __init__(self, index: Index):
        super().__init__()

        self.index = index

    @abstractmethod
    def extract(self, question: str, doc: str):
        raise NotImplementedError
