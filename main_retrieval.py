from retrieval import baseline
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True, choices=['baseline'],
                        help='What retrieval model to use.')
    parser.add_argument('-a', '--action', type=str, required=True, choices=['filter', 'tfidf'],
                        help='What to do.')
    args, _ = parser.parse_known_args()
    command = f'{args.model}@{args.action}'

    if command == 'baseline@filter':
        baseline.filter.top_5000()

    if command == 'baseline@tfidf':
        baseline.ranking.main()


