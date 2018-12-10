from retrieval.neural.modules.encoders import Encoder
from retrieval.neural.modules.scorers import Scorer
import main_constants as const
from abc import abstractmethod
from torch import nn
import torch


class Ranker(nn.Module):
    epochs_trained: int
    weight: float

    def __init__(self, query_encoder: Encoder, document_encoder: Encoder, scorer: Scorer):
        super().__init__()

        self.epochs_trained = 0

        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.scorer = scorer

    @abstractmethod
    def forward(self, query: torch.tensor, document: torch.tensor):
        raise NotImplementedError


class Pointwise(Ranker):

    def __init__(self, query_encoder: Encoder, document_encoder: Encoder, scorer: Scorer):
        super().__init__(query_encoder, document_encoder, scorer)
        self.weight = const.TRAIN_NO_CANDIDATES / 2  # ratio of relevant to irrelevant documents.
        self.weight = 1.0
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.weight])).to(device=const.DEVICE)

    def forward(self, query: torch.tensor, document: torch.tensor) -> torch.tensor:
        query_hns = self.query_encoder(query)
        document_hns = self.document_encoder(document)

        score = self.scorer(document_hns, query_hns)

        return score
