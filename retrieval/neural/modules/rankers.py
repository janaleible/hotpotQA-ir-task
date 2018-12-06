import main_constants as const
from retrieval.neural.modules.encoders import Encoder
from retrieval.neural.modules.scorers import Scorer
from torch import nn


class Pointwise(nn.Module):
    epochs_trained: int

    def __init__(self, query_encoder: Encoder, document_encoder: Encoder, scorer: Scorer):
        super().__init__()

        self.epochs_trained = 0

        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.scorer = scorer

        self.criterion = nn.BCEWithLogitsLoss().to(device=const.DEVICE)

    def forward(self, query: [str], document: [str]):
        query_hns = self.query_encoder(query)
        document_hns = self.document_encoder(document)

        score = self.scorer(document_hns, query_hns)

        return score
