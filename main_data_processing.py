from data_processing import trec, title_maps, id_maps, embeddings, reference, candidates, features
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True,
                        choices=['trec', 'titles', 'ids', 'embeddings', 'reference', 'candidates', 'features', 'all'],
                        help='Data processing action. trec=build TREC corpus from first paragraph of each document.')
    parser.add_argument('-u', '--use_less_memory', type=bool, default=True,
                        help='Use less memory. Useful for building TREC corpus.')
    parser.add_argument('-s', '--sub_action', type=str, help='Parameter control for actions.',
                        choices=['E6B.50', 'E6B.100', 'E6B.200', 'E6B.300', 'skip'])
    args, _ = parser.parse_known_args()

    if args.action in ['trec', 'all']:
        trec.build(args.use_less_memory)
    if args.action in ['titles', 'all']:
        title_maps.build()
    if args.action in ['ids', 'all']:
        id_maps.build()
    if args.action in ['embeddings', 'all']:
        embeddings.build(args.sub_action)
    if args.action in ['reference', 'all']:
        reference.build()
    if args.action in ['candidates', 'all']:
        if hasattr(args, 'sub_action'):
            candidates.build(bool(args.sub_action))
    if args.action in ['features', 'all']:
        features.build()
