from data_processing import extract
from data_processing import inverted_index
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, required=True,
                        choices=['e', 'bii'], help='Data processing action. e=extract, bi=build inverted index')
    args, _ = parser.parse_known_args()

    if args.action == 'e':
        extract.title_and_first_paragraph()
    elif args.action == 'bii':
        inverted_index.build()
