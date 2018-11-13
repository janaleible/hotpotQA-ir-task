import os

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
PREPROCESSED_DATA_DIR = os.path.join('.', 'data', 'preprocessed')

# TRAINING_SET = os.path.join('data', 'hotpot', 'train_1k.json')
TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')
PREPROCESSED_DB = os.path.join('.', 'data', 'preprocessed.sqlite')

MAPS_DIR = os.path.join('.', 'data', 'maps')
TOKEN2ID = os.path.join('.', 'data', 'maps', 'token2id.tar')
ID2TOKEN = os.path.join('.', 'data', 'maps', 'id2token.tar')
TITLE2ID = os.path.join('.', 'data', 'maps', 'title2id.tar')
ID2TITLE = os.path.join('.', 'data', 'maps', 'id2title.tar')

INDEX_DIR = os.path.join('.', 'data', 'index')
UNIGRAM_INDEX = os.path.join('.', 'data', 'index', 'unigram.tar')
BIGRAM_INDEX = os.path.join('.', 'data', 'index', 'bigram.tar')
