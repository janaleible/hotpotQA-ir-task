from retrieval import term
from retrieval import neural
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--group', type=str, required=True, choices=['term', 'neural'],
                        help='What to do.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['overlap', 'uni_tfidf', 'bi_tfidf', 'prf_lm', 'max_pool_bllr_pw', 'max_pool_llr_pw',
                                 'mean_pool_bllr_pw', 'mean_pool_llr_pw', 'gru_llr_pw',
                                 'max_pool_llr_features_pw', 'max_pool_llr_embeddings_pw', 'max_pool_llr_full_pw']
                        , help='What retrieval model to use.')
    args, _ = parser.parse_known_args()
    command = f'{args.group}@{args.model}'

    if args.group == 'neural':
        neural.train.run(neural.configs.models[args.model])
    elif command == 'term@overlap':
        term.variants.overlap.process()
    elif command == 'term@uni_tfidf':
        term.variants.uni_tfidf.process()
    elif command == 'term@bi_tfidf':
        term.variants.bi_tfidf.process()
    elif command == 'term@prf_lm':
        term.variants.prf_lm.process()
    else:
        raise ValueError(f'Unknown command: {command}')
