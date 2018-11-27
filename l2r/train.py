import copy
import gc as garbage_collector
import pickle
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm
import main_constants as c
from services.index import Index, Tokenizer

EMBEDDING_DIMENSION = 15
ENCODER_HIDDEN_SIZE = 15


class Pointwise(nn.Module):

    tokenizer: Tokenizer
    token2id: Dict[str, int]

    def __init__(self, token2id):
        super().__init__()

        self.tokenizer = Tokenizer()
        self.token2id = token2id

        self.document_embedding = nn.Embedding(
            len(self.token2id),
            EMBEDDING_DIMENSION
        )

        self.query_embedding = nn.Embedding(
            len(self.token2id),
            EMBEDDING_DIMENSION
        )

        self.document_encoder = nn.GRU(
            input_size=EMBEDDING_DIMENSION,
            hidden_size=ENCODER_HIDDEN_SIZE
        )

        self.query_encoder = nn.GRU(
            input_size=EMBEDDING_DIMENSION,
            hidden_size=ENCODER_HIDDEN_SIZE
        )

        self.similarity = nn.CosineSimilarity(
            dim=2
        )

    def forward(self, query: str, document: str):

        query_tensor = self.prepare_input(query)
        document_tensor = self.prepare_input(document)

        query_embedding = torch.unsqueeze(self.query_embedding(query_tensor), dim=1)
        document_embedding = torch.unsqueeze(self.document_embedding(document_tensor), dim=1)

        (document_encoding, document_hn) = self.document_encoder(document_embedding)
        (query_encoding, query_hn) = self.query_encoder(query_embedding)

        score = self.similarity(document_hn, query_hn)

        return torch.abs(score).reshape((1,))

    def prepare_input(self, s: str) -> torch.LongTensor:

        tokenized = [self.token2id.get(token, 0) for token in self.tokenizer.tokenize(s)]

        return torch.LongTensor(tokenized)


def get_token2id() -> Dict[str, int]:

    index = Index()
    token2id = copy.deepcopy(index.token2id)
    index = None
    garbage_collector.collect()

    return token2id


def train() -> Pointwise:

    model = Pointwise(get_token2id())
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('created model, optimizer, loss', flush=True)

    with open(c.L2R_TRAINING_SET, 'rb') as file:
        training_set = pickle.load(file)

    print('loaded dataset', flush=True)

    for epoch in range(1):

        running_loss = 0

        for (query, document, target) in tqdm(training_set):

            optimizer.zero_grad()

            score = model(query, document)
            loss = criterion(score, torch.FloatTensor([target]))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch}: loss = {running_loss}', flush=True)

    return model


def evaluate(model: Pointwise) -> float:

    with open(c.L2R_TEST_SET, 'rb') as f:
        test_set = pickle.load(f)

    right = 0
    wrong = 0
    for (query, document, target) in test_set:

        prediction = model(query, document)
        if abs(prediction.item() - target) < 0.5:
            right += 1
        else:
            wrong += 1

    return right / (right + wrong)


def train_and_save():

    model = train()

    with open(c.L2R_MODEL, 'wb') as f:
        pickle.dump(model, f)


def load_and_evaluate():

    with open(c.L2R_MODEL, 'rb') as f:
        model = pickle.load(f)

    print(f'Accuracy: {round(evaluate(model), 4)}')
