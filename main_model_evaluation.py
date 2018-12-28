import argparse

import numpy
import os
import pickle
from typing import Dict, Tuple

import pytrec_eval
import torch
from torch.utils import data
from tqdm import tqdm

import main_constants as ct
from retrieval.neural.configs import Config
from retrieval.neural.configs import models
from datetime import datetime
from retrieval.neural.dataset import QueryDocumentsDataset
from services import helpers, parallel
from services.evaluator import Evaluator
from services.run import Run

INT2WID: Dict[int, int]
WID2TITLE: Dict[int, str]


def run_eval(_set: str, config: Config):
    start = datetime.now()
    if _set == 'train':
        feature_db = ct.TRAIN_FEATURES_DB
        ref = ct.TRAIN_TREC_REFERENCE
    elif _set == 'dev':
        feature_db = ct.DEV_FEATURES_DB
        ref = ct.DEV_TREC_REFERENCE
    elif _set == 'test':
        feature_db = ct.TEST_FEATURES_DB
        ref = ct.TEST_TREC_REFERENCE
    else:
        raise ValueError(f'Unknown set {_set}.')
    with open(ct.INT2WID, 'rb') as file:
        global INT2WID
        INT2WID = pickle.load(file)
    with open(ct.WID2TITLE, 'rb') as file:
        global WID2TITLE
        WID2TITLE = pickle.load(file)

    trec_eval = ct.L2R_EVAL.format(config.name, 'test')
    trec_eval_agg = ct.L2R_EVAL_AGG.format(config.name, 'test')

    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
    # noinspection PyCallingNonCallable
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)
    _ = _load_checkpoint(model, optimizer, config)
    helpers.log(f'Loaded maps, model, and optimizer in {datetime.now() - start}.')

    test_data_set = QueryDocumentsDataset(feature_db)
    test_data_loader = data.DataLoader(test_data_set, ct.BATCH_SIZE, False,
                                       num_workers=os.cpu_count(), collate_fn=QueryDocumentsDataset.collate)

    model.eval()
    epoch_run = Run()
    epoch_eval = Evaluator(ref, measures=pytrec_eval.supported_measures)

    final_scores = torch.empty((len(test_data_loader.dataset), 1), dtype=torch.float)
    question_ids = []
    document_ids = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_data_loader)):
            (questions, documents, features, targets, batch_question_ids, batch_document_ids) = batch
            questions = questions.to(device=ct.DEVICE, non_blocking=True)
            documents = documents.to(device=ct.DEVICE, non_blocking=True)
            features = features.to(device=ct.DEVICE, non_blocking=True)

            batch_size = questions.shape[0]
            scores, encodings = model(questions, documents, features)

            question_ids.extend(batch_question_ids)
            document_ids.extend(batch_document_ids)
            if batch_size == ct.BATCH_SIZE:
                final_scores[idx * ct.BATCH_SIZE:(idx + 1) * ct.BATCH_SIZE] = scores
            else:
                final_scores[idx * ct.BATCH_SIZE:] = scores

    for batch_run in tqdm(parallel.execute(_build_run,
                                           parallel.chunk(10000,
                                                          zip(question_ids, document_ids, final_scores.numpy())))):
        epoch_run.update_rankings(batch_run)

    trec_eval, trec_eval_agg = epoch_eval.evaluate(epoch_run, trec_eval, trec_eval_agg, False)
    er_10 = 0
    for stats in trec_eval.values():
        er_10 += stats['recall_10'] == 1.0
    er_10 /= len(trec_eval)

    print(f'ndcg@10:\t\t{trec_eval_agg["ndcg_cut_10"]:.4f}')
    print(f'map@10:\t\t{trec_eval_agg["map_cut_10"]:.4f}')
    print(f'er@10:\t\t{er_10:.4f}')
    print(f'recall@10:\t\t{trec_eval_agg["recall_10"]:.4f}')
    print(f'recall@100:\t\t{trec_eval_agg["recall_100"]:.4f}')
    print(f'recall@1000:\t\t{trec_eval_agg["recall_1000"]:.4f}')


def _build_run(batch: Tuple[int, Tuple[str, int, numpy.float]]) -> Run:
    batch_run = Run()
    idx, batch = batch
    for i, (question_id, document_id, score) in enumerate(batch):
        # hack for taking care of double title. prefer the proper one to the disambiguation one regardless of what the
        # index says.
        wid = INT2WID[document_id]
        if wid == 38754454:
            wid = 2209045
        title = WID2TITLE[wid]
        batch_run.update_ranking(question_id, title, score.item())

    return batch_run


def _load_checkpoint(model, optimizer, config: Config):
    best_statistic = 0
    start = datetime.now()
    if os.path.isfile(ct.L2R_TRAIN_PROGRESS.format(config.name)):
        with open(ct.L2R_BEST_MODEL.format(config.name), 'rb') as file:
            checkpoint = torch.load(file, map_location=ct.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs_trained = checkpoint['epoch']

        best_statistic = checkpoint['best_statistic']
        helpers.log(f'Loaded checkpoint from {ct.L2R_MODEL.format(config.name)} in {datetime.now() - start}.')
    return best_statistic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['max_pool_llr_features_pw', 'max_pool_llr_embeddings_pw', 'max_pool_llr_full_pw'])
    args, _ = parser.parse_known_args()
    run_eval(args.dataset, models[args.model])
