import os
from types import SimpleNamespace

EOP = " 0eop0 "
EOS = " 0eos0 "

NO_INDEXES = 1
CHUNK_SIZE = 1000

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
TREC_CORPUS_DIR = os.path.join('.', 'data', 'trec')

INDEX_DIR = os.path.join('.', 'data', 'index')
WID2TITLE = os.path.join('.', 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join('.', 'data', 'index', 'title2wid.tar')
INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri')

TRAINING_SET = os.path.join('data', 'hotpot', 'train_100.json')
# TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')

FILTERED_DIR = os.path.join('data', 'filtered')
FILTERED_DB = os.path.join('data', 'filtered', 'db.sqlite')
FILTER_RESULTS = os.path.join('data', 'filtered', 'results.npy')

SQL = SimpleNamespace()
SQL.CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS filtered 
(id TEXT PRIMARY KEY, type TEXT, level TEXT, target_titles BLOB, result_int_ids BLOB)
"""
SQL.INSERT = """
INSERT INTO filtered VALUES (?, ?, ?, ?, ?)
"""
SQL.CHECK_EXISTS = """
SELECT id FROM filtered WHERE id in {}
"""