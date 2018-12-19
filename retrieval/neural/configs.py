from typing import Dict, Any

from torch import nn, optim

from retrieval.neural.modules.encoders import Encoder, MaxPoolEncoder, MeanPoolEncoder, GRUEncoder
from retrieval.neural.modules.scorers import Scorer, AbsoluteCosineScorer, FullBilinearLogisticRegression, \
    FullLinearLogisticRegression, SemanticLinearLogisticRegression, FeatureLinearLogisticRegression
from retrieval.neural.modules.rankers import Pointwise
from torch.optim import SGD, Adam

import main_constants as const


class Config(object):
    name: str
    train_candidate_db: str
    train_question_set: str
    dev_question_set: str
    trainable: bool

    query_encoder: Encoder
    document_encoder: Encoder
    scorer: Scorer
    ranker: nn.Module
    optimizer: optim.Optimizer
    epochs: int

    scorer_kwargs: Dict[str, Any]
    optimizer_kwargs: Dict[str, Any]

    def __init__(self, name: str, train_candidate_db: str, train_question_set: str, dev_question_set: str,
                 trainable: bool,
                 query_encoder: Encoder, document_encoder: Encoder, scorer: Scorer, ranker: nn.Module,
                 optimizer: optim.Optimizer, epochs: float, embedding_dim: int,
                 scorer_kwargs: Dict[str, Any] = {}, optimizer_kwargs: Dict[str, Any] = {}):
        self.name = name
        self.trainable = trainable
        self.train_candidate_db = train_candidate_db
        self.train_question_set = train_question_set
        self.dev_question_set = dev_question_set
        self.optimizer = optimizer
        self.ranker = ranker
        self.scorer = scorer
        self.document_encoder = document_encoder
        self.epochs = epochs
        self.query_encoder = query_encoder
        self.embedding_dim = embedding_dim
        self.scorer_kwargs = scorer_kwargs
        self.optimizer_kwargs = optimizer_kwargs


models = {
    'max_pool_llr_full_pw': Config(**{'name': 'max_pool_llr_full_pw',
                                      'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                      'train_question_set': const.TRAIN_HOTPOT_SET,
                                      'dev_question_set': const.DEV_HOTPOT_SET,
                                      'query_encoder': MaxPoolEncoder,
                                      'document_encoder': MaxPoolEncoder,
                                      'scorer': FullLinearLogisticRegression,
                                      'ranker': Pointwise,
                                      'optimizer': Adam,
                                      'embedding_dim': 50,
                                      'epochs': 100,
                                      'trainable': True,
                                      'scorer_kwargs': {
                                          'in_features': 50 * 2 + 20
                                      }}
                                   ),
    'max_pool_llr_embeddings_pw': Config(**{'name': 'max_pool_llr_embeddings_pw',
                                            'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                            'train_question_set': const.TRAIN_HOTPOT_SET,
                                            'dev_question_set': const.DEV_HOTPOT_SET,
                                            'query_encoder': MaxPoolEncoder,
                                            'document_encoder': MaxPoolEncoder,
                                            'scorer': SemanticLinearLogisticRegression,
                                            'ranker': Pointwise,
                                            'optimizer': Adam,
                                            'embedding_dim': 50,
                                            'epochs': 100,
                                            'trainable': True,
                                            'scorer_kwargs': {
                                                'in_features': 50 * 2
                                            }}
                                         ),
    'max_pool_llr_features_pw': Config(**{'name': 'max_pool_llr_features_pw',
                                          'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                          'train_question_set': const.TRAIN_HOTPOT_SET,
                                          'dev_question_set': const.DEV_HOTPOT_SET,
                                          'query_encoder': MaxPoolEncoder,
                                          'document_encoder': MaxPoolEncoder,
                                          'scorer': FeatureLinearLogisticRegression,
                                          'ranker': Pointwise,
                                          'optimizer': Adam,
                                          'embedding_dim': 50,
                                          'epochs': 100,
                                          'trainable': True,
                                          'scorer_kwargs': {
                                              'in_features': 20
                                          }}
                                       ),
    'gru_llr_pw': Config(**{'name': 'gru_llr_pw',
                            'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                            'train_question_set': const.TRAIN_HOTPOT_SET,
                            'dev_question_set': const.DEV_HOTPOT_SET,
                            'query_encoder': GRUEncoder,
                            'document_encoder': GRUEncoder,
                            'scorer': SemanticLinearLogisticRegression,
                            'ranker': Pointwise,
                            'optimizer': Adam,
                            'embedding_dim': 50,
                            'epochs': 100,
                            'trainable': True,
                            'scorer_kwargs': {
                                'in_features': 50 * 2
                            }}),
    'mean_pool_llr_pw': Config(**{'name': 'mean_pool_llr_pw',
                                  'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                  'train_question_set': const.TRAIN_HOTPOT_SET,
                                  'dev_question_set': const.DEV_HOTPOT_SET,
                                  'query_encoder': MeanPoolEncoder,
                                  'document_encoder': MeanPoolEncoder,
                                  'scorer': SemanticLinearLogisticRegression,
                                  'ranker': Pointwise,
                                  'optimizer': Adam,
                                  'embedding_dim': 50,
                                  'epochs': 100,
                                  'trainable': True,
                                  'scorer_kwargs': {
                                      'in_features': 50 * 2
                                  }}),
    'max_pool_bllr_pw': Config(**{'name': 'max_pool_bllr_pw',
                                  'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                  'train_question_set': const.TRAIN_HOTPOT_SET,
                                  'dev_question_set': const.DEV_HOTPOT_SET,
                                  'query_encoder': MaxPoolEncoder,
                                  'document_encoder': MaxPoolEncoder,
                                  'scorer': SemanticLinearLogisticRegression,
                                  'ranker': Pointwise,
                                  'optimizer': Adam,
                                  'embedding_dim': 50,
                                  'epochs': 100,
                                  'trainable': True,
                                  'scorer_kwargs': {
                                      'in_features': 50 * 2
                                  }}
                               ),
    'mean_pool_bllr_pw': Config(**{'name': 'mean_pool_bllr_pw',
                                   'train_candidate_db': const.TRAIN_CANDIDATES_DB,
                                   'train_question_set': const.TRAIN_HOTPOT_SET,
                                   'dev_question_set': const.DEV_HOTPOT_SET,
                                   'query_encoder': MeanPoolEncoder,
                                   'document_encoder': MeanPoolEncoder,
                                   'scorer': SemanticLinearLogisticRegression,
                                   'ranker': Pointwise,
                                   'optimizer': Adam,
                                   'embedding_dim': 50,
                                   'epochs': 100,
                                   'trainable': True,
                                   'scorer_kwargs': {
                                       'in_features': 50 * 2
                                   }}),
}
