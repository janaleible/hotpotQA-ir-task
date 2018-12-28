from datetime import datetime
import json
import os

import torch
from torch.utils.data import DataLoader

import main_constants as ct
from retrieval.neural.configs import models, Config
from retrieval.neural.dataset import QueryDocumentsDataset
from retrieval.neural.train import _evaluate_epoch
from services import helpers


def evaluate_testset(model_name: str, output_dir: str):
    os.makedirs('./evaluation')

    config = models[model_name]
    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)

    _load_checkpoint(model, optimizer, config)

    dataset = QueryDocumentsDataset(ct.TEST_FEATURES_DB)
    data_loader = DataLoader(dataset, ct.BATCH_SIZE, True, pin_memory=True, collate_fn=QueryDocumentsDataset.collate,
                             num_workers=os.cpu_count())

    stats = _evaluate_epoch(model, ct.TEST_TREC_REFERENCE, data_loader, 'x', 'y', False)
    run = stats[0]
    stats = stats[1:]

    print(stats)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, model_name + '_hotpot.json'), 'w') as file:
        # use DEV_HOTPOT because that corresponds to our test set. the actual hotpot test set is unlabeled.
        json.dump(run.to_json(ct.TEST_FEATURES_DB, ct.DEV_HOTPOT_SET), file, indent=True)


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
    evaluate_testset('max_pool_llr_full_pw', 'evaluation/')
