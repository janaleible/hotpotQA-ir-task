import os

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
PREPROCESSED_DATA_DIR = os.path.join('.', 'data', 'preprocessed')
INDEX_FILE = os.path.join('.', 'data', 'index.tar')

PREPROCESSED_DB = os.path.join('data', 'preprocessed.sqlite')

TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')
