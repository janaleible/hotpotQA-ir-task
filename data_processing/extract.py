import nltk

from data_processing.inverted_index import stemmer, stopwords
from data_processing.sql import *
from services import parallel
from constants import *
from glob import glob
import itertools as it
import sqlite3
import logging
import pickle
import json
import bz2
import re
import os

logging.basicConfig(level='INFO')


def remove_links(text: str):
    text = text.replace('</a>', ' ')
    text = re.sub('<a href=[^>]*>', '', text)

    return text


def title_and_first_paragraph():
    file_paths = sorted(glob(os.path.join(RAW_DATA_DIR, '*')))
    done_count = 0

    connection = sqlite3.connect(PREPROCESSED_DB)
    c = connection.cursor()

    c.execute(CREATE_TABLE_IF_NOT_EXISTS)
    connection.commit()
    logging.info(f'\t[{os.getpid()}]\tStarting extraction. Total to process: {len(file_paths):03d}')
    for group, group_inserts in parallel.execute(_process_folder, file_paths):
        c.executemany(INSERT_EXTRACTED_DOC, group_inserts)
        connection.commit()
        done_count += 1
    logging.info(f'\t[{os.getpid()}]\tFinished extraction.\t[{done_count:03d}/{len(file_paths):03d}]')
    connection.close()


def _process_folder(folder_path: str):
    group = folder_path.split('/')[-1]
    data = []

    file_paths = glob(os.path.join(folder_path, '*.bz2'))
    for file_path in file_paths:
        with bz2.BZ2File(file_path) as file:
            for line in file:
                article = json.loads(line.decode('utf-8'))

                paragraphs = []
                char_count = 0
                paragraph_index = 0

                while char_count < 500 and paragraph_index < len(article['text']):
                    plaintext = [remove_links(sentence) for sentence in article['text'][paragraph_index]]
                    paragraphs.append(plaintext)
                    char_count += sum(len(sentence) for sentence in plaintext)
                    paragraph_index += 1

                doc_string = " ".join(["".join(sentences) for sentences in paragraphs])

                tokens = [stemmer.stem(token.lower()) for token in nltk.word_tokenize(doc_string) if
                          token.lower() not in stopwords]
                doc = (article['id'], article['title'], doc_string, pickle.dumps(paragraphs), pickle.dumps(tokens))
                data.append(doc)
    logging.info(f'\t[{os.getpid()}]\tFinished processing folder {group}.')

    return group, data


if __name__ == '__main__':
    title_and_first_paragraph()
