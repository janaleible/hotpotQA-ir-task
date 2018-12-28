import json
import os
import pickle

from torch.utils.data import DataLoader

import main_constants as ct
from retrieval.neural.configs import models
from retrieval.neural.dataset import QueryDocumentsDataset
from retrieval.neural.train import _load_checkpoint, _evaluate_epoch


def evaluate_testset(model_name: str, output_dir: str):

    config = models[model_name]

    query_encoder = config.query_encoder(config.embedding_dim)
    document_encoder = config.document_encoder(config.embedding_dim)
    scorer = config.scorer(**config.scorer_kwargs)
    model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
    optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)

    _load_checkpoint(model, optimizer, config)

    with open(ct.INT2WID, 'rb') as file:
        INT2WID = pickle.load(file)
    with open(ct.WID2TITLE, 'rb') as file:
        WID2TITLE = pickle.load(file)

    dataset = QueryDocumentsDataset(ct.TEST_FEATURES_DB)
    dataloader = DataLoader(dataset, ct.BATCH_SIZE, True, pin_memory=True, collate_fn=QueryDocumentsDataset.collate,
                            num_workers=8)

    stats = _evaluate_epoch(model, ct.TEST_TREC_REFERENCE, dataloader, 'x', 'y', False)
    run = stats[0]
    stats = stats[1:]

    print(stats)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, model_name + '_hotpot.json'), 'w') as file:
        # use DEV_HOTPOT because that corresponds to our test set. the actual hotpot test set is unlabeled.
        json.dump(run.to_json(ct.TEST_FEATURES_DB, ct.DEV_HOTPOT_SET), file)


if __name__ == '__main__':

    evaluate_testset('max_pool_llr_embeddings_pw', 'evaluation/')
