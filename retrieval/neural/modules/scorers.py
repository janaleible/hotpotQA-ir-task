import torch
from torch import nn
from abc import abstractmethod


class Scorer(nn.Module):

    @abstractmethod
    def forward(self, document_encodings: torch.FloatTensor, query_encodings: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class CosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, document_encodings: torch.FloatTensor, query_encodings: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = document_encodings.shape[0]
        return self.similarity_measure(document_encodings, query_encodings).view(batch_size, 1)


class AbsoluteCosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, document_encodings: torch.FloatTensor, query_encodings: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = document_encodings.shape[0]
        similarity = self.similarity_measure(document_encodings, query_encodings)

        return torch.abs(similarity).view(batch_size, 1)


class LinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, True)

    def forward(self, document_encodings: torch.FloatTensor, query_encodings: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = document_encodings.shape[0]
        return self.linear(torch.cat([document_encodings, query_encodings], dim=2)).view(batch_size, 1)


class BilinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.bilinear = nn.Bilinear(in_features, in_features, 1, True)

    def forward(self, document_encodings: torch.FloatTensor, query_encodings: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = document_encodings.shape[0]
        return self.bilinear(document_encodings, query_encodings).view(batch_size, 1)
