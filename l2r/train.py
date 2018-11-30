import copy
import csv
import gc as garbage_collector
import os
import pickle
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

class Pointwise(nn.Module):

    tokenizer: Tokenizer
    token2id: Dict[str, int]

    def __init__(self, device, token2id):
        super().__init__()

        self.tokenizer = Tokenizer()
        self.token2id = token2id

        self.device = device

        self.epochs_trained = 0

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

        return torch.LongTensor(tokenized).to(self.device)


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


def train(model: Pointwise, device, number_of_epochs: int =15) -> Pointwise:

    model.to(device)
    criterion = nn.BCELoss().to(device)
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
        if not stats[0][0] == model.epochs_trained:
            os.remove(c.L2R_TMP_TRAIN_PROGRESS)

    for epoch in range(number_of_epochs):

        epoch_loss = 0
        correct_predictions = 0

        for (query, document, target) in tqdm(training_set):

            optimizer.zero_grad()

            score = model(query, document)
            loss = criterion(score, torch.FloatTensor([target]).to(device))
            correct_predictions += (abs(score.item() - target) < 0.5)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        training_acc = correct_predictions / len(training_set)

        test_model = Pointwise(model.token2id)
        test_model.load_state_dict(model.state_dict())
        test_acc = evaluate(test_model)

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

        prediction = model(query, document)
        if abs(prediction.item() - target) < 0.5:
            right += 1
        else:
            wrong += 1

    return right / (right + wrong)


def train_and_save(number_of_epochs: int=15):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Pointwise(device, get_token2id()) # TODO: optionally load model and resume training

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

    print(f'Accuracy: {round(evaluate(model), 4)}')
