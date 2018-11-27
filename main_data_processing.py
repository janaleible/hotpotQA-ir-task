from data_processing import trec, titles, ids, embeddings
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True,
                        choices=['trec', 'titles', 'ids', 'embeddings', 'all'],
                        help='Data processing action. trec=build TREC corpus from first paragraph of each document.')
    parser.add_argument('-u', '--use_less_memory', type=bool, default=True,
                        help='Use less memory. Useful for building TREC corpus.')
    parser.add_argument('-s', '--sub_action', type=str, help='Parameter control for actions.',
                        choices=['E6B.50', 'E6B.100', 'E6B.200', 'E6B.300'])
    args, _ = parser.parse_known_args()

    if args.action in ['titles', 'all']:
        titles.build()
    if args.action in ['trec', 'all']:
        trec.build(args.use_less_memory)
    if args.action in ['ids', 'all']:
        ids.build()
    if args.action in ['embeddings', 'all']:
        embeddings.build(args.sub_action)
