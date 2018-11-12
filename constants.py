import os

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
PREPROCESSED_DATA_DIR = os.path.join('.', 'data', 'preprocessed')
INDEX_FILE = os.path.join('.', 'data', 'index.tar')

PREPROCESSED_DB = os.path.join('.', 'data', 'preprocessed.sqlite')

MAPS_DIR = os.path.join('.', 'data', 'maps')
TOKEN2ID = os.path.join('.', 'data', 'maps', 'token2id.tar')
ID2TOKEN = os.path.join('.', 'data', 'maps', 'id2token.tar')
TITLE2ID = os.path.join('.', 'data', 'maps', 'title2id.tar')
ID2TITLE = os.path.join('.', 'data', 'maps', 'id2title.tar')

INDEX_DIR = os.path.join('.', 'data', 'index')
UNIGRAM_INDEX = os.path.join('.', 'data', 'index', 'unigram.tar')
BIGRAM_INDEX = os.path.join('.', 'data', 'index', 'bigram.tar')