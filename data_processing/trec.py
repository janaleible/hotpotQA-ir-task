"""This module builds the TREC corpus that will be passed to the Indri index. Read on for important details."""
import pandas as pd

from unidecode import unidecode

from main_constants import EOP, EOS, RAW_DATA_DIR, TREC_CORPUS_DIR, DOCUMENT_DB
from typing import Dict, Any, Tuple, List
from datetime import datetime
from services import parallel, helpers
from lxml import etree
from glob import glob
import sqlite3
import logging
import json
import bz2
import os
import re

logging.basicConfig(level='INFO')
CHUNK_SIZE = 100000
USE_LESS_MEMORY: bool


def build(use_less_memory: bool):
    """Build the corpus of TREC documents files asynchronously from the HotpotQA raw wiki data.

    Expects the uncompressed HotpotQA raw wiki data available in the ``./data/raw`` folder. Folders are processed in
    order. Resulting documents are collected and sorted according to their ids. Each persisted file carries
    ``CHUNK_SIZE`` documents and is named as ``{first_doc_id_in_file}@{last_doc_id_in_file}``.

    If asked to use less memory, it will defer persistence to the child processes.

    :param use_less_memory: Whether to use less memory by not sorting documents and instead persisting them under a file
    with the same name as the folder from which the raw data originate.
    :return: None.
    """
    global USE_LESS_MEMORY
    USE_LESS_MEMORY = use_less_memory

    assert os.path.exists(RAW_DATA_DIR), f'Cannot find raw data in {os.path.abspath(RAW_DATA_DIR)}'
    os.makedirs(os.path.abspath(TREC_CORPUS_DIR), exist_ok=True)

    folder_paths = sorted(glob(os.path.join(RAW_DATA_DIR, '*')))
    doc_triples = []

    # create document database
    helpers.log('Creating documents database.')
    db = sqlite3.connect(DOCUMENT_DB)
    cursor: sqlite3.Cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, text TEXT)")
    db.commit()
    cursor.close()
    db.close()

    dfs = []

    helpers.log('Extracting TREC documents.')
    if USE_LESS_MEMORY:
        for _ in parallel.execute(_process_raw_data_folder, folder_paths):
            pass
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tExtraction done.')
    else:
        for doc_triples_by_folder in parallel.execute(_process_raw_data_folder, folder_paths):
            doc_triples.extend(doc_triples_by_folder)
            doc_triples = sorted(doc_triples, key=lambda triple: triple[0])
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tExtraction done.')

        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tPersisting TREC documents.')
        for _ in parallel.execute(_process_doc_triples, parallel.chunk(100000, doc_triples)):
            pass
        logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tPersistence done.')
    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tFinished building TREC corpus.')

    return


def _process_raw_data_folder(folder_path: str):
    """Load documents from the JSON collections line by line.

    Extract document id, title, and first paragraph from each document in each JSON. Create TREC documents and collect
    in a list.

    If ``USE_LESS_MEMORY`` is set to ``True``, the TREC documents will be persisted to disk under a file named after the
    folder from which the files originate. If set to ``False`` the documents will be returned for further processing in
    the main thread.

    Store the document string in a database for later reference.

    :param folder_path: The path to the folder where the compressed JSON collection of raw wiki data lies.
    :return: A sorted collection of (document_id, document_title, trec_document_string)
    """
    doc_count = 0
    doc_pairs: List[Tuple[int, str]] = []
    doc_triples = []
    file_paths = sorted(glob(os.path.join(folder_path, '*.bz2')))
    for file_path in file_paths:
        with bz2.BZ2File(file_path) as file:
            for line in file:
                doc = json.loads(line.decode('utf-8'))
                doc_id, doc_title, doc_str = _extract_doc(doc)
                doc_pairs.append((doc_id, doc_str))
                doc_count += 1
                # doc_triples.append((doc_id, doc_title, _build_trec(doc_id, doc_title, doc_str)))

    folder = folder_path.split("/")[-1]
    helpers.log(f'Extracted documents from folder {folder}.')

    db = sqlite3.connect(DOCUMENT_DB)
    cursor = db.cursor()
    cursor.executemany("INSERT INTO documents (id, text) VALUES (?, ?)", doc_pairs)
    db.commit()
    cursor.close()
    db.close()

    helpers.log(f'Persisted {doc_count} documents to database.')

    if USE_LESS_MEMORY:
        file_name = os.path.join(TREC_CORPUS_DIR, f'{folder}.trectext')
        # _process_doc_triples(doc_triples, file_name)
    else:
        return doc_triples


def _process_doc_triples(doc_triples: List[Tuple[int, str, str]], file_name: str = None):
    """Persist TREC documents to disk.

    File name is given by the first and last document id in the collection.
    :param doc_triples: A collection of (document_id. document_title, document_trec_string)
    :param file_name: The name of the file where the documents will be persisted. If not specified, will be composed of
    the first and last id of the document in the collection.
    :return: None
    """
    if file_name is None:
        file_name = os.path.join(TREC_CORPUS_DIR, f'{doc_triples[0][0]:08d}@{doc_triples[-1][0]:08d}.trectext')
    doc_str = list(zip(*doc_triples))[2]

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write("".join(doc_str))

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tPersisted {file_name.split("/")[-1]}.')

    return


def _extract_doc(doc: Dict[str, Any]):
    """Extract document data from a raw document JSON.

    Paragraphs are added until the total number of characters exceeds 500 or the document is finished.

    Paragraphs are separated by *0EOP0*. Sentences are separated by *0EOS0*. The first paragraph is always the title
    This is skipped in this step because the title will be added later when the TREC document is built. Skipping it here
    prevents duplicates.

    :param doc: The document JSON parsed as a dictionary.
    :return: A tuple (document_id, document_title, trec_document_string)
    """
    paragraphs = []
    char_count = 0
    paragraph_index = 1
    while char_count < 500 and paragraph_index < len(doc['text']):
        plain_text = [__remove_links(sentence) for sentence in doc['text'][paragraph_index]]
        paragraphs.append(plain_text)
        char_count += sum(len(sentence) for sentence in plain_text)
        paragraph_index += 1

    doc_string = f"{EOP}".join([f"{EOS}".join(sentences) for sentences in paragraphs])
    doc_string = unidecode(doc_string)

    return int(doc['id']), doc['title'], doc_string


def _build_trec(doc_id: int, title: str, doc_str: str):
    """Build a TREC text document with an id, a title, and some text. Document in built in XML style.

    The title is added inside the text element so that *Indri Structured Query* field restriction can be used in
    retrieval. More information here: https://sourceforge.net/p/lemur/wiki/The%20Indri%20Query%20Language/
    :param doc_id: An external document id as found in the raw wiki data.
    :param title: The title of the document. Should be unique but isn't in the raw data.
    :param doc_str: The document string.
    :return: An XML-style TREC text document ready to be passed to Indri for indexing.
    """
    doc = etree.Element('DOC')
    doc_no = etree.SubElement(doc, 'DOCNO')
    doc_no.text = str(doc_id)
    doc_text = etree.SubElement(doc, 'TEXT')
    doc_title = etree.SubElement(doc_text, 'TITLE')
    doc_title.text = title
    doc_title.tail = doc_str

    return etree.tostring(doc, encoding='unicode', pretty_print=True)


def __remove_links(text: str):
    """Remove links of from a text but keep the display text.

    :param text: a string possibly with links.
    :return: a string that has no links.
    """
    text = text.replace('</a>', ' ')
    text = re.sub('<a href=[^>]*>', '', text)

    return text
