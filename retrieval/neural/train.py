import csv
import json
import os
import random
import shutil
from datetime import datetime
from typing import Tuple
import numpy
import pytrec_eval
from retrieval.evaluate import Evaluator, Run
from retrieval.neural.configs import Config
from retrieval.neural.dataset import TrainDataset, DevDataset
from torch.utils.data import DataLoader
import main_constants as const
from torch import nn, optim
import torch
from services import helpers
from services.index import Index

INDEX: Index
random.seed(42)
torch.random.manual_seed(42)
numpy.random.seed(42)


def run(config: Config) -> None:
    global INDEX
    INDEX = Index('tfidf')

    start = datetime.now()
    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=const.DEVICE)
    # noinspection PyCallingNonCallable
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)
    helpers.log(f'Loaded model and optimizer in {datetime.now() - start}.')

    train_dataset = TrainDataset(INDEX, config.train_candidate_db, config.train_question_set, True)
    dev_dataset = DevDataset(INDEX, config.dev_question_set)
    train_loader = DataLoader(train_dataset, const.BATCH_SIZE, True, collate_fn=TrainDataset.collate, num_workers=8)

    best_accuracy = _load_checkpoint(model, optimizer, config)
    remaining_epochs = config.epochs - model.epochs_trained
    for epoch in range(remaining_epochs):
        is_best = False

        # train
        train_loss, train_correct_predictions = _train_epoch(model, optimizer, train_loader, config)
        train_acc = train_correct_predictions / len(train_dataset)
        helpers.log(f'Epoch {model.epochs_trained}: Loss = {train_loss}')

        # evaluate
        # dev_acc, map_10, ndcg_10, recall_10 = _evaluate_epoch(model, dev_dataset)
        dev_acc = 0
        # save statistics
        # _save_statistics(config.name, epoch, train_loss, train_acc, dev_acc, map_10, ndcg_10, recall_10)
        _save_statistics(config.name, epoch, train_loss, train_acc, -1, -1, -1, -1)

        # save model
        model.epochs_trained += 1
        if dev_acc >= best_accuracy:
            best_accuracy = dev_acc
            is_best = True
        _save_checkpoint(config.name, model, optimizer, best_accuracy, is_best)

    _evaluate_epoch(model, dev_dataset)

    return


def _train_epoch(model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, config: Config):
    epoch_loss = 0
    correct_predictions = 0
    for idx, batch in enumerate(data_loader):
        (queries, documents, targets, _) = batch
        batch_size = len(queries)

        optimizer.zero_grad()

        scores = model(queries, documents)
        loss = model.criterion(scores, targets)
        if config.trainable:
            loss.backward()
            optimizer.step()

        # undo elementwise mean and save epoch loss
        epoch_loss += loss.item() * batch_size
        correct_predictions += torch.sum(torch.abs(targets - scores) < 0.5).item()
        del loss

    return epoch_loss / len(data_loader.dataset), correct_predictions


def _evaluate_epoch(model: nn.Module, dev_dataset: torch.utils.data.Dataset) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        correct_predictions = 0
        evaluator = Evaluator(const.DEV_DUMMY_TREC_REFERENCE, measures=pytrec_eval.supported_measures)
        eval_run = Run()
        for idx in range(len(dev_dataset)):
            (query, retrieved_docs, golden_docs, retrieved_titles, golden_titles, question_id) = dev_dataset[idx]
            ranking_dict = {}
            for i in range(len(retrieved_docs)):
                ranking_dict[retrieved_titles[i]] = model(torch.tensor(query).unsqueeze(dim=0),
                                                          torch.tensor(retrieved_docs[i]).unsqueeze(dim=0)).item()
            eval_run.add_question(question_id, ranking_dict)

            for doc in golden_docs:
                correct_predictions += torch.sum(
                    model(torch.tensor(query).unsqueeze(dim=0), torch.tensor(doc).unsqueeze(dim=0)) > 0.5).item()
            irrelevant_count = 0
            i = 0
            while irrelevant_count < 2:
                if retrieved_docs[i] not in golden_docs:
                    correct_predictions += torch.sum(
                        model(torch.tensor(query).unsqueeze(dim=0),
                              torch.tensor(retrieved_docs[i]).unsqueeze(dim=0)) < 0.5).item()
                    irrelevant_count += 1
                i += 1
        json.dump(eval_run, open('./data/run.json', 'w'), indent=True)
        _, trec_eval_agg = evaluator.evaluate(eval_run, save=False)
        dev_accuracy = correct_predictions / (4 * len(dev_dataset))

    return dev_accuracy, trec_eval_agg['map_cut_10'], trec_eval_agg['ndcg_cut_10'], trec_eval_agg['recall_10']


def _save_statistics(name: str, epoch: int, mean_epoch_loss: float, train_acc: float, dev_acc: float, map_10: float, ndcg_10: float, recall_10: float):
    os.makedirs(const.L2R_MODEL_DIR.format(name), exist_ok=True)
    with open(const.L2R_TRAIN_PROGRESS.format(name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, mean_epoch_loss, train_acc, dev_acc, map_10, ndcg_10, recall_10])


def _load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: Config):
    best_accuracy = 0
    start = datetime.now()
    if os.path.isfile(const.L2R_TRAIN_PROGRESS.format(config.name)):
        with open(const.L2R_MODEL.format(config.name), 'rb') as file:
            checkpoint = torch.load(file, map_location=const.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs_trained = checkpoint['epoch']

        best_accuracy = checkpoint['best_accuracy']
        helpers.log(f'Loading checkpoint from {const.L2R_MODEL.format(config.name)} in {datetime.now() - start}.')
    return best_accuracy


def _save_checkpoint(name: str, model: nn.Module, optimizer: optim.Optimizer, best_accuracy: float, is_best: bool):
    checkpoint = {
        'epoch': model.epochs_trained,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, const.L2R_MODEL.format(name))
    if is_best:
        shutil.copyfile(const.L2R_MODEL.format(name), const.L2R_BEST_MODEL.format(name))
