import argparse

from l2r.build_dataset import build_l2r_dataset
from l2r.train import train_and_save, load_and_evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True, choices=['data', 'train', 'eval', 'all'],
                        help='Choose task')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=15,
                        help='number of epochs to train the model for. Only use with action "train".')
    args, _ = parser.parse_known_args()

    if args.action in ['data', 'all']:
        build_l2r_dataset()

    if args.action in ['train', 'all']:
        train_and_save(args.epochs)

    if args.action in ['eval', 'all']:
        load_and_evaluate()
