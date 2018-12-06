import os
import torch

# use tmp dir on cluster, project root locally
BASE_DIR = (os.environ['TMPDIR'] if (os.environ.get('SLURM_JOBID') is not None) else '.')

EOP = " 0eop0 "
EOS = " 0eos0 "

CHUNK_SIZE = 10

RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TREC_CORPUS_DIR = os.path.join(BASE_DIR, 'data', 'trec')

INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
WID2TITLE = os.path.join(BASE_DIR, 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join(BASE_DIR, 'data', 'index', 'title2wid.tar')
WID2INT = os.path.join(BASE_DIR, 'data', 'index', 'wid2int.tar')
INT2WID = os.path.join(BASE_DIR, 'data', 'index', 'int2wid.tar')
TOKEN2ID = os.path.join(BASE_DIR, 'data', 'index', 'token2id.tar')
ID2TOKEN = os.path.join(BASE_DIR, 'data', 'index', 'id2token.tar')
ID2DF = os.path.join(BASE_DIR, 'data', 'index', 'id2df.tar')
ID2TF = os.path.join(BASE_DIR, 'data', 'index', 'id2tf.tar')

# INDEX
INDRI_PARAMETERS = os.path.join(BASE_DIR, 'index.xml')
INDRI_INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index', 'indri')

HOTPOT_DIR = os.path.join(BASE_DIR, 'data', 'hotpot')
# TRAINING_SET = os.path.join(HOTPOT_DIR, 'train_dummy.json')
TRAINING_SET = os.path.join(HOTPOT_DIR, 'train_full.json')
DEV_DISTRACTOR_SET = os.path.join(HOTPOT_DIR, 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join(HOTPOT_DIR, 'dev_fullwiki.json')

TERM_RETRIEVALS_DIR = os.path.join(BASE_DIR, 'data', 'term_retrievals')
OVERLAP_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'overlap')
UNIGRAM_TFIDF_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'uni_tfidf')
BIGRAM_TFIDF_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'bi_tfidf')
PRF_LM_DIR = os.path.join(TERM_RETRIEVALS_DIR, 'prf_lm')

EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'data', 'embeddings')
EMBEDDINGS_50_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d.txt')
EMBEDDINGS_100_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d.txt')
EMBEDDINGS_200_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d.txt')
EMBEDDINGS_300_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d.txt')
EMBEDDINGS_50 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d.npz')
EMBEDDINGS_100 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d.npz')
EMBEDDINGS_200 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d.npz')
EMBEDDINGS_300 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d.npz')

L2R_DATA_DIR = os.path.join(BASE_DIR, 'data', 'l2r')
L2R_TRAINING_SET = os.path.join(L2R_DATA_DIR, 'training_set_dummy.tar')
L2R_TEST_SET = os.path.join(L2R_DATA_DIR, 'test_set_dummy.tar')
# L2R_TRAINING_SET = os.path.join(L2R_DATA_DIR, 'training_set_full.tar')
# L2R_TEST_SET = os.path.join(L2R_DATA_DIR, 'test_set_full.tar')


L2R_MODEL_DIR = os.path.join(BASE_DIR, 'models', '{}')
L2R_MODEL = os.path.join(L2R_MODEL_DIR, 'checkpoint.pt')
L2R_BEST_MODEL = os.path.join(L2R_MODEL_DIR, 'checkpoint_best.pt')
L2R_TRAIN_PROGRESS = os.path.join(L2R_MODEL_DIR, 'learning_progress.csv')
L2R_LEARNING_PROGRESS_PLOT = os.path.join(L2R_MODEL_DIR, 'learning_progress.pdf')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100

DUMMY_UNIGRAM_TFIDF_DB = os.path.join(TERM_RETRIEVALS_DIR, 'uni_tfidf.dummy', 'retrievals.sqlite')
DUMMY_TRAIN_QUESTION_SET = os.path.join(HOTPOT_DIR, 'train_dummy.json')
DUMMY_DEV_QUESTION_SET = os.path.join(HOTPOT_DIR, 'dev_dummy.json')

DEV_DUMMY_TREC_REFERENCE = os.path.join(BASE_DIR, 'data', 'trec_eval', 'reference', 'dev_dummy_reference.json')
