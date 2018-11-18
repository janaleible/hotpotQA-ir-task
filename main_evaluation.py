from main_constants import *
from retrieval import statistics
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True, choices=['baseline'],
                        help='What retrieval model to use.')
    parser.add_argument('-a', '--action', type=str, required=True, choices=['acc'],
                        help='What to do.')
    args, _ = parser.parse_known_args()
    command = f'{args.model}@{args.action}'

    if command == 'baseline@acc':
        statistics.accuracies(TRAINING_SET, BASELINE_FILTERED_DB)
