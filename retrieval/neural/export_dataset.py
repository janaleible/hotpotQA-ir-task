import pickle
from datetime import datetime
import json
import os
from typing import Dict, Tuple

import numpy
import pytrec_eval
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import main_constants as ct
from retrieval.neural.configs import models, Config
from retrieval.neural.dataset import QueryDocumentsDataset
from services import helpers, parallel
from services.evaluator import Evaluator
from services.run import Run

with open(ct.INT2WID, 'rb') as f:
    INT2WID: Dict[int, int] = pickle.load(f)
with open(ct.WID2TITLE, 'rb') as f:
    WID2TITLE: Dict[int, str] = pickle.load(f)


def evaluate_test_set(model_name: str, output_dir: str):
    os.makedirs('./evaluation', exist_ok=True)

    config = models[model_name]
    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)

    _load_checkpoint(model, optimizer, config)

    dataset = QueryDocumentsDataset(ct.TEST_FEATURES_DB)
    data_loader = DataLoader(dataset, ct.BATCH_SIZE, False, collate_fn=QueryDocumentsDataset.collate,
                             num_workers=os.cpu_count(), pin_memory=True)

    model.eval()
    epoch_run = Run()
    epoch_eval = Evaluator(ct.TEST_TREC_REFERENCE, measures=pytrec_eval.supported_measures)

    final_scores = torch.empty((len(data_loader.dataset), 1), dtype=torch.float)
    question_ids = []
    document_ids = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data_loader)):
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

    for batch_run in parallel.execute(_build_run,
                                      parallel.chunk(10000, zip(question_ids, document_ids, final_scores.numpy()))):
        epoch_run.update_rankings(batch_run)

    trec_eval, trec_eval_agg = epoch_eval.evaluate(epoch_run, save=False)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, model_name + '_hotpot.json'), 'w') as file:
        # use DEV_HOTPOT because that corresponds to our test set. the actual hotpot test set is unlabeled.
        json.dump(epoch_run.to_json(ct.TEST_FEATURES_DB, ct.DEV_HOTPOT_SET), file, indent=True)

    print(json.dumps(trec_eval_agg, indent=True))


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


def _load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Config):
    best_statistic = 0
    start = datetime.now()
    if os.path.isfile(ct.L2R_TRAIN_PROGRESS.format(config.name)):
        with open(ct.L2R_BEST_MODEL.format(config.name), 'rb') as file:
            checkpoint = torch.load(file, map_location=ct.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.epochs_trained = checkpoint['epoch']

        best_statistic = checkpoint['best_statistic']
        helpers.log(f'Loaded checkpoint from {ct.L2R_BEST_MODEL.format(config.name)} in {datetime.now() - start}.')
    return best_statistic


if __name__ == '__main__':
    evaluate_test_set('max_pool_llr_full_pw', 'evaluation/')
