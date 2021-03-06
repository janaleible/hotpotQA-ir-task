import csv
import json
import os
import pickle
import random
import shutil
from datetime import datetime
from typing import Tuple, Dict
import numpy
import pytrec_eval

from services.evaluator import Evaluator
from services.run import Run
from retrieval.neural.configs import Config
from retrieval.neural.dataset import QueryDocumentsDataset
from torch.utils.data import DataLoader
import main_constants as ct
from torch import nn, optim
import torch
from services import helpers, parallel

METRICS = Tuple[Run, float, float, float, float, float, float, float, float, float, float, float]
random.seed(42)
torch.random.manual_seed(42)
numpy.random.seed(42)

with open(ct.INT2WID, 'rb') as file:
    INT2WID: Dict[int, int] = pickle.load(file)
with open(ct.WID2TITLE, 'rb') as file:
    WID2TITLE: Dict[int, str] = pickle.load(file)


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

    best_recall_100 = _load_checkpoint(model, optimizer, config)
    remaining_epochs = config.epochs - model.epochs_trained

    train_stats = _evaluate_epoch(model, ct.TRAIN_TREC_REFERENCE, train_loader,
                                  trec_eval_train, trec_eval_agg_train, False)
    dev_stats = _evaluate_epoch(model, ct.DEV_TREC_REFERENCE, dev_loader,
                                trec_eval_dev, trec_eval_agg_dev, False)

    # Hack to avoid having to adapt all index accesses below after adding run
    train_stats = train_stats[1:]
    dev_stats = dev_stats[1:]

    _save_epoch_stats(config.name, model.epochs_trained, -1, train_stats, dev_stats)

    for epoch in range(remaining_epochs):
        is_best = False
        last_epoch = (model.epochs_trained + 1) == config.epochs

        # train
        train_loss = _train_epoch(model, optimizer, train_loader, config)

        # only once every 10 epochs for speed.
        model.epochs_trained += 1
        if model.epochs_trained % 10 == 0:
            # evaluate and save statistics

            train_stats = _evaluate_epoch(model, ct.TRAIN_TREC_REFERENCE, train_loader,
                                          trec_eval_train, trec_eval_agg_train, last_epoch)
            dev_stats = _evaluate_epoch(model, ct.DEV_TREC_REFERENCE, dev_loader,
                                        trec_eval_dev, trec_eval_agg_dev, last_epoch)

            # Hack to avoid having to adapt all index accesses below after adding run
            train_run = train_stats[0]
            train_stats = train_stats[1:]
            dev_run = dev_stats[0]
            dev_stats = dev_stats[1:]

            _save_epoch_stats(config.name, model.epochs_trained, train_loss, train_stats, dev_stats)

            # save model
            if dev_stats[6] >= best_recall_100:
                best_recall_100 = dev_stats[6]
                is_best = True
            _save_checkpoint(config.name, model, optimizer, best_recall_100, is_best, train_run, dev_run)

    return


def _load_datasets():
    train_dataset = QueryDocumentsDataset(ct.TRAIN_FEATURES_DB)
    dev_dataset = QueryDocumentsDataset(ct.DEV_FEATURES_DB)
    train_loader = DataLoader(train_dataset, ct.BATCH_SIZE, True, pin_memory=True,
                              collate_fn=QueryDocumentsDataset.collate, num_workers=os.cpu_count())
    dev_loader = DataLoader(dev_dataset, ct.BATCH_SIZE, True, pin_memory=True,
                            collate_fn=QueryDocumentsDataset.collate, num_workers=os.cpu_count())

    return train_loader, dev_loader


def _train_epoch(model: nn.Module, optimizer: optim.Optimizer, data_loader: DataLoader, config: Config) -> float:
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(data_loader):
        (questions, documents, features, targets, _, _) = batch
        batch_size = len(questions)
        questions = questions.to(device=ct.DEVICE, non_blocking=True)
        documents = documents.to(device=ct.DEVICE, non_blocking=True)
        targets = targets.to(device=ct.DEVICE, non_blocking=True)
        features = features.to(device=ct.DEVICE, non_blocking=True)

        scores, encodings = model(questions, documents, features)
        loss = model.criterion(scores, targets)

        if config.trainable:
            loss.backward()
            # noinspection PyArgumentList
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

    final_scores = torch.empty((len(data_loader.dataset), 1), dtype=torch.float)
    question_ids = []
    document_ids = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            (questions, documents, features, targets, batch_question_ids, batch_document_ids) = batch
            questions = questions.to(device=ct.DEVICE, non_blocking=True)
            documents = documents.to(device=ct.DEVICE, non_blocking=True)
            features = features.to(device=ct.DEVICE, non_blocking=True)
            targets = targets.to(device=ct.DEVICE, non_blocking=True)

            batch_size = questions.shape[0]
            scores, encodings = model(questions, documents, features)
            acc += torch.sum((torch.round(scores) == targets).to(dtype=torch.float))

            question_ids.extend(batch_question_ids)
            document_ids.extend(batch_document_ids)
            if batch_size == ct.BATCH_SIZE:
                final_scores[idx * ct.BATCH_SIZE:(idx + 1) * ct.BATCH_SIZE] = scores
            else:
                final_scores[idx * ct.BATCH_SIZE:] = scores

    for batch_run in parallel.execute(_build_run, parallel.chunk(10000, zip(question_ids, document_ids, final_scores.numpy()))):
        epoch_run.update_rankings(batch_run)

    acc = acc / len(data_loader.dataset)
    _, trec_eval_agg = epoch_eval.evaluate(epoch_run, trec_eval, trec_eval_agg, save)

    return epoch_run, acc.item(), \
           trec_eval_agg['map_cut_10'], trec_eval_agg['ndcg_cut_10'], trec_eval_agg['recall_10'], \
           trec_eval_agg['map_cut_100'], trec_eval_agg['ndcg_cut_100'], trec_eval_agg['recall_100'], \
           trec_eval_agg['map_cut_1000'], trec_eval_agg['ndcg_cut_1000'], trec_eval_agg['recall_1000'], \
           trec_eval_agg['P_5']


def _build_run(batch: Tuple[int, Tuple[str, int, numpy.float]]) -> Run:
    batch_run = Run()
    idx, batch = batch
    for i, (question_id, document_id, score) in enumerate(batch):
        title = WID2TITLE[INT2WID[document_id]]
        batch_run.update_ranking(question_id, title, score.item())

    return batch_run


def _save_epoch_stats(name: str, epoch: int, train_loss: float,
                      train_stats: Tuple[float, ...], dev_stats: Tuple[float, ...]):
    with open(ct.L2R_TRAIN_PROGRESS.format(name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, *train_stats, *dev_stats])

    helpers.log(f'[Epoch {epoch:03d}][Train Acc: {train_stats[0]:0.4f}]'
                f'[Train MAP@10: {train_stats[1]:0.4f}][Train NDCG@10: {train_stats[2]:0.4f}]'
                f'[Train Precision@5: {train_stats[10]:0.4f}]'
                f'[Train Loss: {train_loss:0.4f}]'
                f'[Dev Acc: {dev_stats[0]:0.4f}]'
                f'[Dev MAP@10: {dev_stats[1]:0.4f}][Dev NDCG@10: {dev_stats[2]:0.4f}]'
                f'[Dev Recall@10: {dev_stats[3]:0.4f}]'
                f'[Dev MAP@100: {dev_stats[4]:0.4f}][Dev NDCG@100: {dev_stats[5]:0.4f}]'
                f'[Dev Recall@100: {dev_stats[6]:0.4f}]'
                f'[Dev Recall@1000: {dev_stats[9]:0.4f}]')


def _load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, config: Config):
    best_statistic = 0
    start = datetime.now()
    if os.path.isfile(ct.L2R_TRAIN_PROGRESS.format(config.name)):
        with open(ct.L2R_MODEL.format(config.name), 'rb') as file:
            checkpoint = torch.load(file, map_location=ct.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs_trained = checkpoint['epoch']

        best_statistic = checkpoint['best_statistic']
        helpers.log(f'Loaded checkpoint from {ct.L2R_MODEL.format(config.name)} in {datetime.now() - start}.')
    return best_statistic


def _save_checkpoint(name: str, model: nn.Module, optimizer: optim.Optimizer, best_statistic: float, is_best: bool,
                     train_run: Run, dev_run: Run):
    checkpoint = {
        'epoch': model.epochs_trained,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_statistic': best_statistic
    }
    torch.save(checkpoint, ct.L2R_MODEL.format(name))
    if is_best:
        shutil.copyfile(ct.L2R_MODEL.format(name), ct.L2R_BEST_MODEL.format(name))
        os.makedirs(ct.RUN_DIR.format(name), exist_ok=True)

        with open(ct.RESULT_RUN_PICKLE.format(name, 'train'), 'wb') as file:
            pickle.dump(train_run, file)
        with open(ct.RESULT_RUN_PICKLE.format(name, 'dev'), 'wb') as file:
            pickle.dump(dev_run, file)
        with open(ct.RESULT_RUN_JSON.format(name, 'train'), 'w') as file:
            json.dump(train_run, file)
        with open(ct.RESULT_RUN_JSON.format(name, 'dev'), 'w') as file:
            json.dump(dev_run, file)
