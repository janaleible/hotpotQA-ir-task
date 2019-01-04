from typing import List

import torch
from torch import nn
from abc import abstractmethod


class Scorer(nn.Module):

    @abstractmethod
    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        raise NotImplementedError


class CosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        return self.similarity_measure(*inputs).view(-1, 1)


class AbsoluteCosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(dim=2)

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        similarity = self.similarity_measure(*inputs)

        return torch.abs(similarity).view(-1, 1)


class FullLinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:

        energies = self.linear(torch.cat(inputs, dim=1)).view(-1, 1)
        if self.training:
            return energies
        else:
            return self.sigmoid(energies)


class SemanticLinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        doc_enc, query_enc, feats = inputs
        energies = self.linear(torch.cat([doc_enc, query_enc], dim=1)).view(-1, 1)
        if self.training:
            return energies
        else:
            return self.sigmoid(energies)


class FeatureLinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.linear = nn.Linear(in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        doc_enc, query_enc, feats = inputs
        energies = self.linear(feats).view(-1, 1)
        if self.training:
            return energies
        else:
            return self.sigmoid(energies)


class FullBilinearLogisticRegression(Scorer):
    def __init__(self, in_features: int):
        super().__init__()

        self.bilinear = nn.Bilinear(in_features, in_features, 1, True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        energies = self.bilinear(*inputs).view(-1, 1)

        if self.training:
            return energies
        else:
            return self.sigmoid(energies)
