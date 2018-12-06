import copy
import csv
import gc as garbage_collector
import os
import pickle
from math import ceil
from typing import Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import main_constants as c
from retrieval.neural.models.rankers import Pointwise
from services.index import Index


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





def train_and_save(number_of_epochs: int=15):

    token2id = get_token2id()

    query_encoder = MaxPoolEncoder(token2id)
    document_encoder = MaxPoolEncoder(token2id)
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
