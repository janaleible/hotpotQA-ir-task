import argparse
from retrieval.neural import train
from retrieval.neural.configs import configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True, choices=['data', 'train', 'eval', 'all'],
                        help='Choose task')
    parser.add_argument('-c', '--config', type=str, required=True, choices=['max_pool_abs_cosine_pw', 'mean_pool_bllr_pw', 'max_pool_llr_pw', 'gru_llr_pw', 'mean_pool_llr_pw'],
                        help='Configuration to load.')
    args, _ = parser.parse_known_args()

    # if args.action in ['data', 'all']:
    #     build_l2r_dataset()

    if args.action in ['train', 'all']:
        train.run(configs[args.config])

    # if args.action in ['eval', 'all']:
    #     load_and_evaluate()
