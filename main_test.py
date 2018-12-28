import argparse
import os

import main_constants
from retrieval.neural.export_dataset import evaluate_testset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['overlap', 'uni_tfidf', 'bi_tfidf', 'prf_lm', 'max_pool_bllr_pw', 'max_pool_llr_pw',
                                 'mean_pool_bllr_pw', 'mean_pool_llr_pw', 'gru_llr_pw',
                                 'max_pool_llr_features_pw', 'max_pool_llr_embeddings_pw', 'max_pool_llr_full_pw', 'all']
                        , help='Which model to create a plot for.')
    parser.add_argument('-o', '--outputdir', type=str, default='test',
                        choices=['report', 'show', 'save'], help='Directory to save the hotpot files in')
    args, _ = parser.parse_known_args()

    for model in os.listdir(main_constants.MODEL_BASE_DIR):
        if os.path.isfile(main_constants.L2R_BEST_MODEL.format(model)):
            if args.model == 'all' or model == args.model:
                evaluate_testset(model, args.outputdir)
