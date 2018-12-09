import csv
import os
import pickle
import random
import shutil
from datetime import datetime
from typing import Tuple, Dict
import numpy
import pytrec_eval

from services.evaluation import Evaluator, Run
from retrieval.neural.configs import Config
from retrieval.neural.dataset import QueryDocumentsDataset
from torch.utils.data import DataLoader
import main_constants as ct
from torch import nn, optim
import torch
from services import helpers

METRICS = Tuple[float, float, float, float, float, float, float]
random.seed(42)
torch.random.manual_seed(42)
numpy.random.seed(42)

INT2WID: Dict[int, int]
WID2TITLE: Dict[int, str]


def run(config: Config) -> None:
    start = datetime.now()
    os.makedirs(ct.L2R_MODEL_DIR.format(config.name), exist_ok=True)
    with open(ct.INT2WID, 'rb') as file:
        global INT2WID
        INT2WID = pickle.load(file)
    with open(ct.WID2TITLE, 'rb') as file:
        global WID2TITLE
        WID2TITLE = pickle.load(file)

    trec_eval_train = ct.L2R_EVAL.format(config.name, 'train')
    trec_eval_agg_train = ct.L2R_EVAL_AGG.format(config.name, 'train')
    trec_eval_dev = ct.L2R_EVAL.format(config.name, 'dev')
    trec_eval_agg_dev = ct.L2R_EVAL_AGG.format(config.name, 'dev')

    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
    # noinspection PyCallingNonCallable
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)
    helpers.log(f'Loaded maps, model, and optimizer in {datetime.now() - start}.')

    train_loader, dev_loader = _load_datasets()

    best_acc = _load_checkpoint(model, optimizer, config)
    remaining_epochs = config.epochs - model.epochs_trained
    for epoch in range(remaining_epochs):
        is_best = False
        last_epoch = (model.epochs_trained + 1) == config.epochs

        # train
        train_loss = _train_epoch(model, optimizer, train_loader, config)

        # only once every 10 epochs for speed.
        model.epochs_trained += 1
        if model.epochs_trained % 1 == 0:
            # evaluate and save statistics

            train_stats = _evaluate_epoch(model, ct.TRAIN_TREC_REFERENCE, train_loader,
                                          trec_eval_train, trec_eval_agg_train, last_epoch)
            dev_stats = _evaluate_epoch(model, ct.DEV_TREC_REFERENCE, dev_loader,
                                        trec_eval_dev, trec_eval_agg_dev, last_epoch)
            _save_epoch_stats(config.name, model.epochs_trained, train_loss, train_stats, dev_stats)
            # save model
            if dev_stats[0] >= best_acc:
                best_acc = dev_stats[0]
                is_best = True
            _save_checkpoint(config.name, model, optimizer, best_acc, is_best)

    return


def _load_datasets():
    train_dataset = QueryDocumentsDataset(ct.TRAIN_UNIGRAM_TFIDF_CANDIDATES)
    dev_dataset = QueryDocumentsDataset(ct.DEV_UNIGRAM_TFIDF_CANDIDATES)
    train_loader = DataLoader(train_dataset, ct.BATCH_SIZE, True,
                              collate_fn=QueryDocumentsDataset.collate, num_workers=8)
    dev_loader = DataLoader(dev_dataset, ct.BATCH_SIZE, True,
                            collate_fn=QueryDocumentsDataset.collate, num_workers=8)

    return train_loader, dev_loader


def _train_epoch(model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, config: Config) -> float:
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(data_loader):
        (questions, documents, targets, _, _) = batch
        batch_size = len(questions)
        questions = questions.to(device=ct.DEVICE)
        documents = documents.to(device=ct.DEVICE)
        targets = targets.to(device=ct.DEVICE)

        scores = model(questions, documents)
        loss = model.criterion(scores, targets)

        if config.trainable:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        # undo elementwise mean and save epoch loss
        epoch_loss += loss.item() * batch_size
        del loss

    return epoch_loss / len(data_loader.dataset)


def _evaluate_epoch(model: nn.Module, ref: str, data_loader: DataLoader, trec_eval: str,
                    trec_eval_agg: str, save: bool) -> METRICS:
    model.eval()
    epoch_run = Run()
    epoch_eval = Evaluator(ref, measures=pytrec_eval.supported_measures)
    acc = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            (questions, documents, targets, question_ids, document_ids) = batch
            questions = questions.to(device=ct.DEVICE)
            documents = documents.to(device=ct.DEVICE)
            targets = targets.to(device=ct.DEVICE)

            scores = model(questions, documents)

            for i in range(len(questions)):
                question_id = question_ids[i]
                document_id = document_ids[i]
                title = WID2TITLE[INT2WID[document_id]]
                epoch_run.update_ranking(question_id, title, scores[i].item())
            acc += torch.sum((torch.round(scores) == targets).to(dtype=torch.float))

        acc = acc / len(data_loader.dataset)
        _, trec_eval_agg = epoch_eval.evaluate(epoch_run, trec_eval, trec_eval_agg, save)
        return acc.item(), \
               trec_eval_agg['map_cut_10'], trec_eval_agg['ndcg_cut_10'], trec_eval_agg['recall_10'], \
               trec_eval_agg['map_cut_100'], trec_eval_agg['ndcg_cut_100'], trec_eval_agg['recall_100']


def _save_epoch_stats(name: str, epoch: int, train_loss: float,
                      train_stats: Tuple[float, ...], dev_stats: Tuple[float, ...]):
    with open(ct.L2R_TRAIN_PROGRESS.format(name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, *train_stats, *dev_stats])
    helpers.log(f'[Epoch {epoch:03d}]\t[Train Acc:\t{train_stats[0]:0.4f}]'
                f'[Train MAP@10:\t{train_stats[1]:0.4f}][Train NDCG@10:\t{train_stats[2]:0.4f}]'
                f'[Train Recall@10:\t{train_stats[3]:0.4f}]'
                f'[Train MAP@100:\t{train_stats[4]:0.4f}][Train NDCG@100:\t{train_stats[5]:0.4f}]'
                f'[Train Recall@100:\t{train_stats[6]:0.4f}]'
                f'[Train Loss:\t{train_loss:0.4f}]'
                )
    helpers.log(f'[Epoch {epoch:03d}]\t[Dev Acc:\t{dev_stats[0]:0.4f}]'
                f'[Dev MAP@10:\t{dev_stats[1]:0.4f}][Dev NDCG@10:\t{dev_stats[2]:0.4f}]'
                f'[Dev Recall@10:\t\t{dev_stats[3]:0.4f}]'
                f'[Dev MAP@100:\t{dev_stats[4]:0.4f}][Dev NDCG@100:\t{dev_stats[5]:0.4f}]'
                f'[Dev Recall@100:\t\t{dev_stats[6]:0.4f}]')


def _load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: Config):
    best_acc = 0
    start = datetime.now()
    if os.path.isfile(ct.L2R_TRAIN_PROGRESS.format(config.name)):
        with open(ct.L2R_MODEL.format(config.name), 'rb') as file:
            checkpoint = torch.load(file, map_location=ct.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs_trained = checkpoint['epoch']

        best_acc = checkpoint['best_accuracy']
        helpers.log(f'Loaded checkpoint from {ct.L2R_MODEL.format(config.name)} in {datetime.now() - start}.')
    return best_acc


def _save_checkpoint(name: str, model: nn.Module, optimizer: optim.Optimizer, best_accuracy: float, is_best: bool):
    checkpoint = {
        'epoch': model.epochs_trained,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, ct.L2R_MODEL.format(name))
    if is_best:
        shutil.copyfile(ct.L2R_MODEL.format(name), ct.L2R_BEST_MODEL.format(name))
