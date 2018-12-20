import numpy
import os
import pickle
from typing import Dict, Tuple

import pytrec_eval
import torch
from torch.utils import data
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


def run_eval(config: Config):
    start = datetime.now()
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
    helpers.log(f'Loaded maps, model, and optimizer in {datetime.now() - start}.')

    test_data_set = QueryDocumentsDataset(ct.TEST_FEATURES_DB)
    test_data_loader = data.DataLoader(test_data_set, ct.BATCH_SIZE, False, num_workers=os.cpu_count())

    model.eval()
    epoch_run = Run()
    epoch_eval = Evaluator(ct.TEST_TREC_REFERENCE, measures=pytrec_eval.supported_measures)
    acc = 0

    final_scores = torch.empty((len(test_data_loader.dataset), 1), dtype=torch.float)
    question_ids = []
    document_ids = []
    with torch.no_grad():
        for idx, batch in enumerate(test_data_loader):
            (questions, documents, features, targets, batch_question_ids, batch_document_ids) = batch
            questions = questions.to(device=ct.DEVICE, non_blocking=True)
            documents = documents.to(device=ct.DEVICE, non_blocking=True)
            features = features.to(device=ct.DEVICE, non_blocking=True)
            targets = targets.to(device=ct.DEVICE, non_blocking=True)

            batch_size = questions.shape[0]
            scores = model(questions, documents, features)
            acc += torch.sum((torch.round(scores) == targets).to(dtype=torch.float))

            question_ids.extend(batch_question_ids)
            document_ids.extend(batch_document_ids)
            if batch_size == ct.BATCH_SIZE:
                final_scores[idx * ct.BATCH_SIZE:(idx + 1) * ct.BATCH_SIZE] = scores
            else:
                final_scores[idx * ct.BATCH_SIZE:] = scores

    for batch_run in parallel.execute(_build_run,
                                      parallel.chunk(10000, zip(question_ids, document_ids, final_scores.numpy()))):
        epoch_run.update_rankings(batch_run)

    acc = acc / len(test_data_loader.dataset)
    _, trec_eval_agg = epoch_eval.evaluate(epoch_run, trec_eval, trec_eval_agg, False)

    print('ndcg10', trec_eval_agg['ndcg_cut_10'])
    print('map10', trec_eval_agg['map_cut_10'])
    print('recall10', trec_eval_agg['recall_10'])
    print('recall100', trec_eval_agg['recall_100'])
    print('recall1000', trec_eval_agg['recall_1000'])


def _build_run(batch: Tuple[int, Tuple[str, int, numpy.float]]) -> Run:
    batch_run = Run()
    idx, batch = batch
    for i, (question_id, document_id, score) in enumerate(batch):
        title = WID2TITLE[INT2WID[document_id]]
        batch_run.update_ranking(question_id, title, score.item())

    return batch_run


if __name__ == '__main__':
    run_eval(models['max_pool_llr_full_pw'])
