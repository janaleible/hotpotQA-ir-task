import torch
from torch import nn
from abc import abstractmethod


class Scorer(nn.Module):

    @abstractmethod
    def forward(self, document_encodings: torch.float, query_encodings: torch.float) -> torch.float:
        raise NotImplementedError


class CosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, document_encodings: torch.float, query_encodings: torch.float) -> torch.float:
        batch_size = document_encodings.shape[0]
        return self.similarity_measure(document_encodings, query_encodings).view(batch_size, 1)


class AbsoluteCosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, document_encodings: torch.float, query_encodings: torch.float) -> torch.float:
        batch_size = document_encodings.shape[0]
        similarity = self.similarity_measure(document_encodings, query_encodings)

        return torch.abs(similarity).view(batch_size, 1)


class LinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, document_encodings: torch.float, query_encodings: torch.float) -> torch.float:
        batch_size = document_encodings.shape[0]
        energies = self.linear(torch.cat([document_encodings, query_encodings], dim=1)).view(batch_size, 1)
        if self.training:
            return energies
        else:
            return self.sigmoid(energies)


class BilinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.bilinear = nn.Bilinear(in_features, in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, document_encodings: torch.float, query_encodings: torch.float) -> torch.float:
        batch_size = document_encodings.shape[0]
        energies = self.bilinear(document_encodings, query_encodings).view(batch_size, 1)

        if self.training:
            return energies
        else:
            return self.sigmoid(energies)
