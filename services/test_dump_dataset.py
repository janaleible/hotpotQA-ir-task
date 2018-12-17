import json
import pickle
#
from torch.utils.data import DataLoader

from retrieval.neural.configs import models
from retrieval.neural.dataset import QueryDocumentsDataset
from retrieval.neural.train import _load_checkpoint, _evaluate_epoch
import main_constants as ct
from services.run import Run

# config = models['max_pool_llr+features_pw']
#
# query_encoder = config.query_encoder(config.embedding_dim)
# document_encoder = config.document_encoder(config.embedding_dim)
# scorer = config.scorer(**config.scorer_kwargs)
# model = config.ranker(query_encoder, document_encoder, scorer).to(device=ct.DEVICE)
# optimizer = config.optimizer(model.parameters(), **config.optimizer_kwargs)
#
# best_acc = _load_checkpoint(model, optimizer, config)
#
# with open(ct.INT2WID, 'rb') as file:
#     INT2WID = pickle.load(file)
# with open(ct.WID2TITLE, 'rb') as file:
#     WID2TITLE = pickle.load(file)
#
# train_dataset = QueryDocumentsDataset(ct.TRAIN_FEATURES_DB)
# train_loader = DataLoader(train_dataset, ct.BATCH_SIZE, True, pin_memory=True, collate_fn=QueryDocumentsDataset.collate, num_workers=8)
#
# # dev_dataset = QueryDocumentsDataset(ct.DEV_FEATURES_DB)
# # dev_loader = DataLoader(dev_dataset, ct.BATCH_SIZE, True, pin_memory=True, collate_fn=QueryDocumentsDataset.collate, num_workers=8)
#
# train_stats = _evaluate_epoch(model, ct.TRAIN_TREC_REFERENCE, train_loader, 'trec_eval_train', 'trec_eval_agg_train', False)
# # dev_stats = _evaluate_epoch(model, ct.DEV_TREC_REFERENCE, dev_loader, 'trec_eval_dev', 'trec_eval_agg_dev', False)
#
#
# print(train_stats[1:])
# # print(dev_stats[1:])
#
# #
# # with open('run.pickle', 'rb') as file:
# #     run: Run = pickle.load(file)
# #
# #
# with open('devfullrun.hotopot', 'w') as file:
#     json.dump(run.to_json(ct.TRAIN_FEATURES_DB, ct.TRAIN_HOTPOT_SET), file)
# # json_data = run.to_json(ct.TRAIN_FEATURES_DB, ct.TRAIN_HOTPOT_SET)
# #
# # with open('test.json', 'w') as file:
# #     json.dump(json_data, file)

import pickle
from services.run import Run


with open('models/max_pool_llr+features_pw/runs/dev.full.pickle', 'rb') as file:
    run = pickle.load(file)

import json
import main_constants as ct


with open('devfullrun.hotopot', 'w') as file:
    json.dump(run.to_json(ct.DEV_FEATURES_DB, ct.TRAIN_HOTPOT_SET), file)
