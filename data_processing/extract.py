from data_processing.sql import CREATE_TABLE_IF_NOT_EXISTS, insert_string
from services import parallel
from constants import *
from glob import glob

import sqlite3
import logging
import pickle
import json
import bz2
import re
import os


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
    connection.close()


    for _ in parallel.execute(_process_folder, file_paths):
        done_count += 1
    logging.info(f'Finished extraction. Successfully processed: [{done_count:03d}/{len(file_paths):03d}]')


def _process_folder(folder_path: str):
    group = folder_path.split('/')[-1]
    data = {}

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

                data[article['title']] = paragraphs

    connection = sqlite3.connect(PREPROCESSED_DB)
    cursor = connection.cursor()

    for (title, article) in data:
        cursor.execute(insert_string(title, pickle.dumps(article)))

    connection.commit()
    connection.close()

    logging.info(f'Finished processing folder {group}.')

    return group, data


if __name__ == '__main__':
    title_and_first_paragraph()
