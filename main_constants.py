from types import SimpleNamespace
import os

EOP = " 0eop0 "
EOS = " 0eos0 "

CHUNK_SIZE = 10

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
TREC_CORPUS_DIR = os.path.join('.', 'data', 'trec')

INDEX_DIR = os.path.join('.', 'data', 'index')
WID2TITLE = os.path.join('.', 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join('.', 'data', 'index', 'title2wid.tar')

# WITHOUT STEMMING AND STOPPING
INDRI_PARAMETERS = 'index.xml'
INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri')

# WITH STEMMING AND STOPPING
# INDRI_PARAMETERS = 'index_stop_stem.xml'
# INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri_stop_stem')

TRAINING_SET = os.path.join('data', 'hotpot', 'train_100.json')
# TRAINING_SET = os.path.join('data', 'hotpot', 'train.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')

TRECEVAL_REFERENCE_DIR = os.path.join('data', 'trec_eval', 'reference')
TRECEVAL_REFERENCE_TRAIN = os.path.join('data', 'trec_eval', 'reference', 'train_reference.json')
TRECEVAL_REFERENCE_DEV = os.path.join('data', 'trec_eval', 'reference', 'dev_reference.json')

TRECEVAL_RESULTS_DIR = os.path.join('data', 'trec_eval', 'results')
TRECEVAL_RESULTS_TRAIN = os.path.join('data', 'trec_eval', 'results', 'train_results.json')
TRECEVAL_RESULTS_DEV = os.path.join('data', 'trec_eval', 'results', 'dev_results.json')

TRECEVAL_EVALUATION_DIR = os.path.join('data', 'trec_eval', 'evaluation')
TRECEVAL_EVALUATION_TRAIN = os.path.join('data', 'trec_eval', 'evaluation', 'train_evaluation.json')
TRECEVAL_EVALUATION_DEV = os.path.join('data', 'trec_eval', 'evaluation', 'dev_evaluation.json')

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
