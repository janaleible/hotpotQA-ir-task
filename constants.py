import os

EOP = " 0eop0 "
EOS = " 0eos0 "

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
TREC_CORPUS_DIR = os.path.join('.', 'data', 'trec')

INDEX_DIR = os.path.join('.', 'data', 'index')
WID2TITLE = os.path.join('.', 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join('.', 'data', 'index', 'title2wid.tar')
INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri')

TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')


