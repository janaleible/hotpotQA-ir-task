import os
import torch

# use tmp dir on cluster, project root locally

BASE_DIR = (os.environ['TMPDIR'] if (os.environ.get('SLURM_JOBID') is not None) else '.')
HOME_DIR = (os.environ['HOME'] if (os.environ.get('SLURM_JOBID') is not None) else '.')

# switch between dummy and full data setting
SETTING = 'full'
# SETTING = 'dummy'

GRAND_CHUNK = 7000
THREAD_NO = 2
PROCESS_CONTEXT = 'fork'

# data processing constants
CHUNK_SIZE = 1000
EOP = " 0eop0 "
EOS = " 0eos0 "

RELEVANT_DOCUMENTS = 2

# Data constants
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TREC_CORPUS_DIR = os.path.join(BASE_DIR, 'data', 'trec')
DOCUMENT_DB = os.path.join(BASE_DIR, 'data', 'documents', 'documents.sqlite')

# Index constants
INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
WID2TITLE = os.path.join(BASE_DIR, 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join(BASE_DIR, 'data', 'index', 'title2wid.tar')
WID2INT = os.path.join(BASE_DIR, 'data', 'index', 'wid2int.tar')
INT2WID = os.path.join(BASE_DIR, 'data', 'index', 'int2wid.tar')
TOKEN2ID = os.path.join(BASE_DIR, 'data', 'index', 'token2id.tar')
ID2TOKEN = os.path.join(BASE_DIR, 'data', 'index', 'id2token.tar')
ID2DF = os.path.join(BASE_DIR, 'data', 'index', 'id2df.tar')
ID2TF = os.path.join(BASE_DIR, 'data', 'index', 'id2tf.tar')
PYNDRI2GLOVE = os.path.join(INDEX_DIR, 'pyndri.glove.tar')
GLOVE2PYNDRI = os.path.join(INDEX_DIR, 'glove2pyndri.tar')

INDRI_PARAMETERS = os.path.join(BASE_DIR, 'index.xml')
INDRI_INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index', 'indri')

# hotpot constants
HOTPOT_DIR = os.path.join(BASE_DIR, 'data', 'hotpot')
TRAIN_HOTPOT_SET = os.path.join(HOTPOT_DIR, f'train_{SETTING}.json')
DEV_HOTPOT_SET = os.path.join(HOTPOT_DIR, f'dev_{SETTING}.json')

# term-based retrieval constants
TERM_RETRIEVALS_DIR = os.path.join(BASE_DIR, 'data', 'term_retrievals')
OVERLAP_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'overlap')
UNIGRAM_TFIDF_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'uni_tfidf')
BIGRAM_TFIDF_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'bi_tfidf')
PRF_LM_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'prf_lm')

# embeddings constants
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'data', 'embeddings')
EMBEDDINGS_50_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d.txt')
EMBEDDINGS_100_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d.txt')
EMBEDDINGS_200_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d.txt')
EMBEDDINGS_300_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d.txt')
EMBEDDINGS_50 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d.npz')
EMBEDDINGS_100 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d.npz')
EMBEDDINGS_200 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d.npz')
EMBEDDINGS_300 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d.npz')

# model constants
MODEL_BASE_DIR = os.path.join(HOME_DIR, 'models')
L2R_MODEL_DIR = os.path.join(MODEL_BASE_DIR, '{}') # Make sure models are saved to permanent storage during training
L2R_MODEL = os.path.join(L2R_MODEL_DIR, 'checkpoint.pt')
L2R_BEST_MODEL = os.path.join(L2R_MODEL_DIR, 'checkpoint_best.pt')
L2R_TRAIN_PROGRESS = os.path.join(L2R_MODEL_DIR, 'learning_progress.csv')
L2R_EVAL = os.path.join(L2R_MODEL_DIR, 'trec_eval_{}.json')
L2R_EVAL_AGG = os.path.join(L2R_MODEL_DIR, 'trec_eval_agg_{}.json')

L2R_LEARNING_PROGRESS_PLOT = os.path.join(L2R_MODEL_DIR, 'learning_progress.pdf')
REPORT_LEARNING_PROGRESS_PLOT = os.path.join('../report/figures/training_{}.pdf')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 400000
BATCH_SIZE = 2048

# candidate constants
TRAIN_DEV_SPLIT = 90 if SETTING == 'dummy' else 85000
TRAIN_NO_CANDIDATES = 10
DEV_NO_CANDIDATES = 1000
TEST_NO_CANDIDATES = 1000
CANDIDATES_CHUNK = 1
CANDIDATES_DIR = os.path.join(BASE_DIR, 'data', 'candidates')

TRAIN_CANDIDATES_DB = os.path.join(CANDIDATES_DIR, f'tfidf.train.{SETTING}.sqlite')
DEV_CANDIDATES_DB = os.path.join(CANDIDATES_DIR, f'tfidf.dev.{SETTING}.sqlite')
TEST_CANDIDATES_DB = os.path.join(CANDIDATES_DIR, f'tfidf.test.{SETTING}.sqlite')

CANDIDATES_TABLE_NAME = 'candidates'
CANDIDATE_COLUMNS = ['question_id', 'type', 'level', 'doc_iid', 'doc_wid', 'doc_title',
                     'question_text', 'doc_text', 'question_tokens', 'doc_tokens',
                     'tfidf', 'relevance']

# reference constants
TRAIN_TREC_REFERENCE = os.path.join(BASE_DIR, 'data', 'trec_eval', f'train_{SETTING}_reference.json')
DEV_TREC_REFERENCE = os.path.join(BASE_DIR, 'data', 'trec_eval', f'dev_{SETTING}_reference.json')
TEST_TREC_REFERENCE = os.path.join(BASE_DIR, 'data', 'trec_eval', f'test_{SETTING}_reference.json')

# entity recognition constants
E2I = {
    "PER": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3
}
I2E = {
    0: "PER",
    1: "LOC",
    2: "ORG",
    3: "MISC"
}

# features constants
FEATURE_EXTRACTORS = [
    'entity',
    'ibm1',
    'nibm1',
    'bigram',
    'nbigram',
    'doclen',
    'qword'
]
FEATURE_BASE_COLUMN_NAMES = [
    'query_id',
    'type',
    'level',
    'doc_id',
    'doc_wid',
    'doc_title',
    'question_text',
    'document_text',
    'query_tokens',
    'doc_tokens',
    'tfidf',
]
FEATURE_TARGET_COLUMN_NAME = 'relevant'

TRAIN_FEATURES_CHUNK = 10
DEV_FEATURES_CHUNK = 10
TEST_FEATURES_CHUNK = 10
FEATURES_DIR = os.path.join(BASE_DIR, 'data', 'features')

TRAIN_FEATURES_DB = os.path.join(FEATURES_DIR, f'train.{SETTING}.feature.db')
DEV_FEATURES_DB = os.path.join(FEATURES_DIR, f'dev.{SETTING}.feature.db')
TEST_FEATURES_DB = os.path.join(FEATURES_DIR, f'test.{SETTING}.feature.db')


TRANSLATION_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'translation')
IBM_MODEL = os.path.join(TRANSLATION_MODEL_DIR, f'ibm1_{SETTING}.pickle')

RUN_DIR = os.path.join(L2R_MODEL_DIR, 'runs')
RESULT_HOTPOT = os.path.join(RUN_DIR, 'hotpot.{}.' + SETTING + '.json')  # format with dev|train
RESULT_RUN_PICKLE = os.path.join(RUN_DIR, '{}.' + SETTING + '.pickle')  # format with dev|train
RESULT_RUN_JSON = os.path.join(RUN_DIR, '{}.' + SETTING + '.run')  # format with dev|train
