from abc import abstractmethod

import numpy as np

from main_constants import *
from torch import nn


class Encoder(nn.Module):
    embed: nn.Module

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        # initialize pre-trained GloVe embeddings.
        self.embed = nn.Embedding \
            .from_pretrained(torch.from_numpy(self._get_embeddings()), freeze=False) \
            .to(device=DEVICE, dtype=torch.float)

    def _get_embeddings(self):
        if self.embedding_dim == 50:
            return np.load(EMBEDDINGS_50)['array']
        elif self.embedding_dim == 100:
            return np.load(EMBEDDINGS_100)['array']
        elif self.embedding_dim == 200:
            return np.load(EMBEDDINGS_200)['array']
        elif self.embedding_dim == 300:
            return np.load(EMBEDDINGS_300)['array']
        else:
            raise ValueError('Unknown embedding dimension.')

    def forward(self, doc_batch: torch.LongTensor):
        doc_embedding = self.embed(doc_batch)

        return self.encode(doc_embedding)

    @abstractmethod
    def encode(self, doc_embedding: torch.LongTensor):
        raise NotImplementedError


class MaxPoolEncoder(Encoder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)

    def encode(self, document_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        return torch.unsqueeze(torch.max(document_embeddings, dim=1)[0], dim=1)


class MeanPoolEncoder(Encoder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)

    def encode(self, document_embeddings: torch.FloatTensor) -> torch.FloatTensor:
        return torch.unsqueeze(torch.mean(document_embeddings, dim=1), dim=1)


class GRUEncoder(Encoder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)

        self.document_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )

    def encode(self, document_embeddings):
        (document_encoding, document_hn) = self.document_encoder(document_embeddings)
        return document_hn.permute(0, 1)
