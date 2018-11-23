import retrieval
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--action', type=str, required=True, choices=['filter'],
                        help='What to do.')
    parser.add_argument('-m', '--model', type=str, required=True, choices=['overlap', 'uni_tfidf', 'bi_tfidf', 'prf_lm']
                        , help='What retrieval model to use.')
    args, _ = parser.parse_known_args()
    command = f'{args.action}@{args.model}'

    retrieval.evaluate.process(command)
