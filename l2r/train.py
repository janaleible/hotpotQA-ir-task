import copy
import csv
import gc as garbage_collector
import os
import pickle
from abc import abstractmethod
from math import ceil
from typing import Dict
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
from tqdm import tqdm
import main_constants as c
from services.index import Index, Tokenizer

EMBEDDING_DIMENSION = 15
ENCODER_HIDDEN_SIZE = 15

nltk.download('punkt')
nltk.download('stopwords')


class Encoder(nn.Module):

    tokenizer: Tokenizer
    token2id: Dict[str, int]

    def __init__(self, token2id):
        super().__init__()

        self.tokenizer = Tokenizer()
        self.token2id = token2id

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    def prepare_input(self, documents: [str]) -> torch.LongTensor:

        tokenized_documents = [[self.token2id.get(token, 0) for token in self.tokenizer.tokenize(document)] for document in documents]
        max_document_length = max(len(doc) for doc in tokenized_documents)

        tensors = []
        for document in tokenized_documents:
            if len(document) < max_document_length:
                document += [0] * (max_document_length - len(document))
            tensors.append(torch.LongTensor(document))

        return dont_care_cuda(torch.stack(tensors).permute(1, 0))


class GRUEncoder(Encoder):

    def __init__(self, token2id):
        super().__init__(token2id)

        self.document_embedding = nn.Embedding(
            len(self.token2id),
            EMBEDDING_DIMENSION
        )

        self.document_encoder = nn.GRU(
            input_size=EMBEDDING_DIMENSION,
            hidden_size=ENCODER_HIDDEN_SIZE
        )

    def forward(self, documents):

        document_tensors = self.prepare_input(documents)
        document_embeddings = self.document_embedding(document_tensors)
        (document_encoding, document_hn) = self.document_encoder(document_embeddings)

        return document_hn


class Scorer(nn.Module):

    @abstractmethod
    def forward(self, document_encodings, query_encodings):
        raise NotImplementedError


class CosineScorer(Scorer):

    def __init__(self):
        super().__init__()

        self.similarity_measure = nn.CosineSimilarity(
            dim=2
        )

    def forward(self, document_encodings, query_encodings):
        return self.similarity_measure(document_encodings, query_encodings)


class Pointwise(nn.Module):

    epochs_trained: int

    def __init__(self,
        query_encoder: Encoder,
        document_encoder: Encoder,
        scorer: Scorer
    ):
        super().__init__()

        self.epochs_trained = 0

        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.scorer = scorer

    def forward(self, query: [str], document: [str]):

        query_hns = self.query_encoder(query)
        document_hns = self.document_encoder(document)

        score = self.scorer(document_hns, query_hns)

        return torch.abs(score)


def get_token2id() -> Dict[str, int]:

    index = Index()
    token2id = copy.deepcopy(index.token2id)
    index = None # make sure to lose reference to the index for memory reasons
    garbage_collector.collect()

    return token2id


def update_learning_progress(learning_progress: {}, epoch: int, loss: float, training_acc: float, test_acc: float):
    learning_progress['epoch'].append(epoch)
    learning_progress['loss'].append(loss)
    learning_progress['test_acc'].append(test_acc)
    learning_progress['training_acc'].append(training_acc)


def train(model: Pointwise, number_of_epochs: int =15) -> Pointwise:

    dont_care_cuda(model)
    criterion = dont_care_cuda(nn.BCELoss())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('created model, optimizer, loss', flush=True)

    with open(c.L2R_TRAINING_SET, 'rb') as file:
        training_set = pickle.load(file)

    print('loaded dataset', flush=True)

    if os.path.isfile(c.L2R_TMP_TRAIN_PROGRESS):
        with open(c.L2R_TMP_TRAIN_PROGRESS) as f:
            reader = csv.reader(f)
            stats = list(sorted(reader, key=lambda row: row[0])) # sort entries in csv by epoch, TODO: parse numbers

        # load learning progress if stats match current model, else remove
        if not int(stats[0][0]) == model.epochs_trained:
            os.remove(c.L2R_TMP_TRAIN_PROGRESS)

    for epoch in range(number_of_epochs):

        epoch_loss = 0
        correct_predictions = 0

        BATCH_SIZE = 5
        number_of_batches = ceil(len(training_set) / BATCH_SIZE)

        # for (query, document, target) in tqdm(training_set):
        for batch_index in range(number_of_batches):

            batch = training_set[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]

            queries = []
            documents = []
            targets = []

            for (query, document, target) in batch:
                queries.append(query)
                documents.append(document)
                targets.append(target)

            optimizer.zero_grad()

            score = model(queries, documents)
            loss = criterion(score, dont_care_cuda(torch.FloatTensor(targets)))
            correct_predictions += torch.sum(abs(torch.FloatTensor(targets) - score.reshape(5)) < 0.5).item()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        training_acc = correct_predictions / len(training_set)

        test_acc = evaluate(model)

        os.makedirs(c.L2R_MODEL_DIR, exist_ok=True)
        with open(c.L2R_TMP_TRAIN_PROGRESS, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss / len(training_set), training_acc, test_acc])

        model.epochs_trained += 1
        torch.save(model, c.L2R_INTERMEDIATE_MODEL.format(model.epochs_trained))
        print(f'Epoch {model.epochs_trained}: loss = {epoch_loss}', flush=True)

    return model


def evaluate(model: Pointwise) -> float:

    with open(c.L2R_TEST_SET, 'rb') as f:
        test_set = pickle.load(f)

    right = 0
    wrong = 0
    for (query, document, target) in test_set:

        prediction = model([query], [document])
        if abs(prediction.item() - target) < 0.5:
            right += 1
        else:
            wrong += 1

    return right / (right + wrong)


def train_and_save(number_of_epochs: int=15):

    token2id = get_token2id()

    query_encoder = GRUEncoder(token2id)
    document_encoder = GRUEncoder(token2id)
    scorer = CosineScorer()
    model = Pointwise(query_encoder, document_encoder, scorer)  # TODO: optionally load model and resume training

    model = train(model, number_of_epochs)

    os.makedirs(c.L2R_MODEL_DIR, exist_ok=True)
    with open(c.L2R_MODEL, 'wb') as f:
        pickle.dump(model, f)

    epochs = []
    loss = []
    training_acc = []
    test_acc = []

    with open(c.L2R_TMP_TRAIN_PROGRESS, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            epochs.append(int(row[0]))
            loss.append(float(row[1]))
            training_acc.append(float(row[2]))
            test_acc.append(float(row[3]))

    plt.plot(epochs, loss, color='tab:orange', label='average loss')
    plt.plot(epochs, training_acc, color='tab:blue', label='training set accuracy')
    plt.plot(epochs, test_acc, color='tab:green', label='test set accuracy')
    plt.xlabel('Epochs')
    plt.xticks(range(1,len(epochs)))
    plt.legend(loc='upper left')

    plt.savefig(c.L2R_LEARNING_PROGRESS_PLOT)


def load_and_evaluate():

    with open(c.L2R_MODEL, 'rb') as f:
        model = pickle.load(f)

    # print(f'Accuracy: {round(evaluate(model), 4)}')


def dont_care_cuda(source):

    if torch.cuda.is_available():
        return source.cuda()
    else:
        return source
