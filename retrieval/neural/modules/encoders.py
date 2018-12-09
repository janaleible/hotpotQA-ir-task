from abc import abstractmethod

import main_constants as const
import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    embed: nn.Module

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        # initialize pre-trained GloVe embeddings. VOCAB_SIZE + 1 to account for the OOV embedding
        _weight = torch.from_numpy(self._get_embeddings())
        self.embed = nn.Embedding(const.VOCAB_SIZE + 1, embedding_dim, 0, _weight=_weight)
        self.embed.to(device=const.DEVICE, dtype=torch.float)

    def _get_embeddings(self):
        if self.embedding_dim == 50:
            return np.load(const.EMBEDDINGS_50)['array']
        elif self.embedding_dim == 100:
            return np.load(const.EMBEDDINGS_100)['array']
        elif self.embedding_dim == 200:
            return np.load(const.EMBEDDINGS_200)['array']
        elif self.embedding_dim == 300:
            return np.load(const.EMBEDDINGS_300)['array']
        else:
            raise ValueError('Unknown embedding dimension.')

    def forward(self, doc_batch: torch.long):
        doc_embedding = self.embed(doc_batch)

        return self.encode(doc_embedding)

    @abstractmethod
    def encode(self, doc_embedding: torch.long):
        raise NotImplementedError


class MaxPoolEncoder(Encoder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)

    def encode(self, document_embeddings: torch.float) -> torch.float:
        return torch.unsqueeze(torch.max(document_embeddings, dim=1)[0], dim=1)


class MeanPoolEncoder(Encoder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)

    def encode(self, document_embeddings: torch.float) -> torch.float:
        return torch.unsqueeze(torch.mean(document_embeddings, dim=1), dim=1)


class GRUEncoder(Encoder):
    encoding_dim: int

    def __init__(self, encoding_dim: int):
        super().__init__(encoding_dim)

        self.encoding_dim = encoding_dim
        self.document_encoder = nn.GRU(
            input_size=encoding_dim,
            hidden_size=encoding_dim,
            batch_first=True
        )

    def encode(self, document_embeddings):
        (batch_size, _, _) = document_embeddings.shape
        (document_encoding, document_hn) = self.document_encoder(document_embeddings)
        return document_hn.view(batch_size, self.encoding_dim)
