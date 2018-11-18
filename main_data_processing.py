from data_processing import trec, titles, index
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True, choices=['trec', 'title', 'index','all'],
                        help='Data processing action. trec=build TREC corpus from first paragraph of each document.')
    parser.add_argument('-u', '--use_less_memory', type=bool, default=False, help='Use less memory. Useful for building TREC corpus.')
    args, _ = parser.parse_known_args()

    if args.action in ['title', 'all']:
        titles.build()
    if args.action in ['trec', 'all']:
        trec.build(args.use_less_memory)
    if args.action in ['index', 'all']:
        index.build()
