from types import SimpleNamespace
import os

EOP = " 0eop0 "
EOS = " 0eos0 "

CHUNK_SIZE = 1000

RAW_DATA_DIR = os.path.join('.', 'data', 'raw')
TREC_CORPUS_DIR = os.path.join('.', 'data', 'trec')

INDEX_DIR = os.path.join('.', 'data', 'index')
WID2TITLE = os.path.join('.', 'data', 'index', 'wid2title.tar')
TITLE2WID = os.path.join('.', 'data', 'index', 'title2wid.tar')
WID2INT = os.path.join('.', 'data', 'index', 'wid2int.tar')
INT2WID = os.path.join('.', 'data', 'index', 'int2wid.tar')

# INDEX WITHOUT STEMMING AND STOPPING
INDRI_PARAMETERS = 'index.xml'
INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri')

# INDEX WITH STEMMING AND STOPPING
# INDRI_PARAMETERS = 'index_stop_stem.xml'
# INDRI_INDEX_DIR = os.path.join('.', 'data', 'index', 'indri_stop_stem')

HOTPOT_DIR = os.path.join('data', 'hotpot')
TRAINING_SET = os.path.join('data', 'hotpot', 'train_dummy.json')
# TRAINING_SET = os.path.join('data', 'hotpot', 'train_full.json')
DEV_DISTRACTOR_SET = os.path.join('data', 'hotpot', 'dev_distractor.json')
DEV_FULLWIKI_SET = os.path.join('data', 'hotpot', 'dev_fullwiki.json')

FILTERS_DIR = os.path.join('data', 'filters')
FILTERS_OVERLAP_DIR = os.path.join('data', 'filters', 'overlap')
FILTERS_UNIGRAM_TFIDF_DIR = os.path.join('data', 'filters', 'uni_tfidf')
FILTERS_BIGRAM_TFIDF_DIR = os.path.join('data', 'filters', 'bi_tfidf')
FILTERS_PRF_LM_DIR = os.path.join('data', 'filters', 'prf_lm')

RANKERS_DIR = os.path.join('data', 'rankers')
# RANKERS_OVERLAP_DB = os.path.join('data', 'rankers', 'overlap')
# RANKERS_UNIGRAM_TFIDF_DB = os.path.join('data', 'rankers', 'uni_tfidf')
# RANKERS_BIGRAM_TFIDF_DB = os.path.join('data', 'rankers', 'bi_tfidf')
# RANKERS_PRF_LM_DB = os.path.join('data', 'rankers', 'prf_lm')

TRECEVAL_REFERENCE_DIR = os.path.join('data', 'trec_eval', 'reference')
TRECEVAL_REFERENCE_TRAIN_FULL = os.path.join('data', 'trec_eval', 'reference', 'train.full.json')
TRECEVAL_REFERENCE_TRAIN_DUMMY = os.path.join('data', 'trec_eval', 'reference', 'train.dummy.json')
TRECEVAL_REFERENCE_DEV = os.path.join('data', 'trec_eval', 'reference', 'dev.json')

TRECEVAL_RESULTS_DIR = os.path.join('data', 'trec_eval', 'results')
TRECEVAL_RESULTS_TRAIN = os.path.join('data', 'trec_eval', 'results', 'train_results.json')
TRECEVAL_RESULTS_DEV = os.path.join('data', 'trec_eval', 'results', 'dev_results.json')

TRECEVAL_EVALUATION_DIR = os.path.join('data', 'trec_eval', 'evaluation')
TRECEVAL_EVALUATION_TRAIN = os.path.join('data', 'trec_eval', 'evaluation', 'train_evaluation.json')
TRECEVAL_EVALUATION_DEV = os.path.join('data', 'trec_eval', 'evaluation', 'dev_evaluation.json')

EMBEDDINGS_DIR = os.path.join('data', 'embeddings')
EMBEDDINGS_50_TXT = os.path.join('data', 'embeddings', 'glove.6B.50d.txt')
EMBEDDINGS_100_TXT = os.path.join('data', 'embeddings', 'glove.6B.100d.txt')
EMBEDDINGS_200_TXT = os.path.join('data', 'embeddings', 'glove.6B.200d.txt')
EMBEDDINGS_300_TXT = os.path.join('data', 'embeddings', 'glove.6B.300d.txt')
EMBEDDINGS_50 = os.path.join('data', 'embeddings', 'glove.6B.50d')
EMBEDDINGS_100 = os.path.join('data', 'embeddings', 'glove.6B.100d')
EMBEDDINGS_200 = os.path.join('data', 'embeddings', 'glove.6B.200d')
EMBEDDINGS_300 = os.path.join('data', 'embeddings', 'glove.6B.300d')
EMBEDDINGS = os.path.join('data', 'embeddings', 'glove.42B.300d')
EMBEDDINGS_TXT = os.path.join('data', 'embeddings', 'glove.42B.300d.txt')

L2R_DIR = os.path.join('data', 'l2r')
L2R_TRAINING_SET = os.path.join(L2R_DIR, 'training_set.tar')
L2R_TEST_SET = os.path.join(L2R_DIR, 'test_set.tar')
L2R_MODEL = os.path.join(L2R_DIR, 'model.tar')