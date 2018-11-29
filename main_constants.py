import os

# use tmp dir on cluster, project root locally
BASE_DIR = (os.environ['TMPDIR'] if (os.environ.get('SLURM_JOBID') is not None) else '.')

EOP = " 0eop0 "
EOS = " 0eos0 "

CHUNK_SIZE = 1000

RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TREC_CORPUS_DIR = os.path.join(BASE_DIR, 'data', 'trec')

INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index')
WID2TITLE = os.path.join(BASE_DIR, 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join(BASE_DIR, 'data', 'index', 'title2wid.tar')
WID2INT = os.path.join(BASE_DIR, 'data', 'index', 'wid2int.tar')
INT2WID = os.path.join(BASE_DIR, 'data', 'index', 'int2wid.tar')

# INDEX WITHOUT STEMMING AND STOPPING
INDRI_PARAMETERS = os.path.join(BASE_DIR, 'index.xml')
INDRI_INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index', 'indri')

# INDEX WITH STEMMING AND STOPPING
# INDRI_PARAMETERS = os.path.join(BASE_DIR, 'index_stop_stem.xml')
# INDRI_INDEX_DIR = os.path.join(BASE_DIR, 'data', 'index', 'indri_stop_stem')

HOTPOT_DIR = os.path.join(BASE_DIR, 'data', 'hotpot')
TRAINING_SET = os.path.join(HOTPOT_DIR, 'train_dummy.json')
# TRAINING_SET = os.path.join(HOTPOT_DIR, 'train_full.json')
DEV_DISTRACTOR_SET = os.path.join(HOTPOT_DIR, 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join(HOTPOT_DIR, 'dev_fullwiki.json')

FILTERS_DIR = os.path.join(BASE_DIR, 'data', 'filters')
FILTERS_OVERLAP_DIR = os.path.join(FILTERS_DIR, 'overlap')
FILTERS_UNIGRAM_TFIDF_DIR = os.path.join(FILTERS_DIR, 'uni_tfidf')
FILTERS_BIGRAM_TFIDF_DIR = os.path.join(FILTERS_DIR, 'bi_tfidf')
FILTERS_PRF_LM_DIR = os.path.join(FILTERS_DIR, 'prf_lm')

RANKERS_DIR = os.path.join(BASE_DIR, 'data', 'rankers')
# RANKERS_OVERLAP_DB = os.path.join(RANKERS_DIR, 'overlap')
# RANKERS_UNIGRAM_TFIDF_DB = os.path.join(RANKERS_DIR, 'uni_tfidf')
# RANKERS_BIGRAM_TFIDF_DB = os.path.join(RANKERS_DIR, 'bi_tfidf')
# RANKERS_PRF_LM_DB = os.path.join(RANKERS_DIR, 'prf_lm')

TRECEVAL_REFERENCE_DIR = os.path.join(BASE_DIR, 'data', 'trec_eval', 'reference')
TRECEVAL_REFERENCE_TRAIN_FULL = os.path.join(TRECEVAL_REFERENCE_DIR, 'train.full.json')
TRECEVAL_REFERENCE_TRAIN_DUMMY = os.path.join(TRECEVAL_REFERENCE_DIR, 'train.dummy.json')
TRECEVAL_REFERENCE_DEV = os.path.join(TRECEVAL_REFERENCE_DIR, 'dev.json')

TRECEVAL_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'trec_eval', 'results')
TRECEVAL_RESULTS_TRAIN = os.path.join(TRECEVAL_RESULTS_DIR, 'train_results.json')
TRECEVAL_RESULTS_DEV = os.path.join(TRECEVAL_RESULTS_DIR, 'dev_results.json')

TRECEVAL_EVALUATION_DIR = os.path.join(BASE_DIR, 'data', 'trec_eval', 'evaluation')
TRECEVAL_EVALUATION_TRAIN = os.path.join(TRECEVAL_EVALUATION_DIR, 'train_evaluation.json')
TRECEVAL_EVALUATION_DEV = os.path.join(TRECEVAL_EVALUATION_DIR, 'dev_evaluation.json')

EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'data', 'embeddings')
EMBEDDINGS_50_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d.txt')
EMBEDDINGS_100_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d.txt')
EMBEDDINGS_200_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d.txt')
EMBEDDINGS_300_TXT = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d.txt')
EMBEDDINGS_50 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.50d')
EMBEDDINGS_100 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.100d')
EMBEDDINGS_200 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.200d')
EMBEDDINGS_300 = os.path.join(EMBEDDINGS_DIR, 'glove.6B.300d')

L2R_DATA_DIR = os.path.join(BASE_DIR, 'data', 'l2r')
L2R_TRAINING_SET = os.path.join(L2R_DATA_DIR, 'training_set_full.tar')
L2R_TEST_SET = os.path.join(L2R_DATA_DIR, 'test_set_full.tar')
# L2R_TRAINING_SET = os.path.join(L2R_DIR, 'training_set_dummy.tar')
# L2R_TEST_SET = os.path.join(L2R_DIR, 'test_set_dummy.tar')

L2R_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'l2r')
L2R_MODEL = os.path.join(L2R_MODEL_DIR, 'model.pt')
L2R_INTERMEDIATE_MODEL = os.path.join(L2R_MODEL_DIR, 'model_{}.pt')
L2R_STATE_DICT = os.path.join(L2R_MODEL_DIR, 'model_state_dict.pt')
L2R_TMP_TRAIN_PROGRESS = os.path.join(L2R_MODEL_DIR, 'tmp_learning_progress.csv')
L2R_LEARNING_PROGRESS_PLOT = os.path.join(L2R_MODEL_DIR, 'learning_progess.png')
