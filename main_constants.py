from types import SimpleNamespace
import os

EOP = " 0eop0 "
EOS = " 0eos0 "

CHUNK_SIZE = 50

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
TREC_CORPUS_DIR = os.path.join('.', 'data', 'trec')

INDEX_DIR = os.path.join('.', 'data', 'index')
WID2TITLE = os.path.join('.', 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join('.', 'data', 'index', 'title2wid.tar')
INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri')
INDRI_PARAMETERS = 'index.xml'

# TRAINING_SET = os.path.join('data', 'hotpot', 'train_100.json')
TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')

FILTERED_DIR = os.path.join('data', 'filtered')
BASELINE_FILTERED_DB = os.path.join('data', 'filtered', 'baseline.sqlite')

SQL = SimpleNamespace()
SQL.CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS filtered 
(id TEXT PRIMARY KEY, type TEXT, level TEXT, target_titles BLOB, result_int_ids BLOB)
"""
SQL.INSERT = """
INSERT INTO filtered VALUES (?, ?, ?, ?, ?)
"""
SQL.CHECK_EXISTS = """
SELECT id FROM filtered WHERE id = ?
"""
SQL.FETCH_ONE_RESULT = """
SELECT result_int_ids FROM filtered WHERE id = ?
"""