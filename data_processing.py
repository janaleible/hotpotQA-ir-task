from data_processing import extract
from data_processing import inverted_index
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True,
                        choices=['e', 'bm', 'bi'], help='Data processing action. e=extract, bi=build inverted index')
    parser.add_argument('-g', '--group', type=str, help='The group to use for index. If none, will use whole dataset.')
    args, _ = parser.parse_known_args()

    if args.action == 'e':
        extract.title_and_first_paragraph()
    elif args.action == 'bm':
        inverted_index.build_mappings()
    elif args.action == 'bi':
        inverted_index.build_index()

