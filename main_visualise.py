import argparse
import os

import main_constants
from retrieval.neural import configs
from retrieval.neural.visualise import plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['overlap', 'uni_tfidf', 'bi_tfidf', 'prf_lm', 'max_pool_bllr_pw', 'max_pool_llr_pw',
                                 'mean_pool_bllr_pw', 'mean_pool_llr_pw', 'gru_llr_pw',
                                 'max_pool_llr_features_pw', 'max_pool_llr_embeddings_pw', 'max_pool_llr_full_pw', 'all']
                        , help='Which model to create a plot for.')
    parser.add_argument('-o', '--output', type=str, default='show',
                        choices=['report', 'show', 'save'], help='What to do with the plot')
    args, _ = parser.parse_known_args()

    for model in os.listdir(main_constants.MODEL_BASE_DIR):

        if args.model == 'all' or model == args.model:
            if os.path.isfile(main_constants.L2R_TRAIN_PROGRESS.format(model)):

                if args.output == 'show':
                    output_path = None
                elif args.output == 'report':
                    output_path = main_constants.REPORT_LEARNING_PROGRESS_PLOT
                elif args.output == 'save':
                    output_path = main_constants.L2R_LEARNING_PROGRESS_PLOT
                else:
                    raise ValueError('unknown output option')

                plot(configs.models[model], show=(args.output == 'show'), output_path=output_path)
