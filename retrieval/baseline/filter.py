from retrieval.filtering import bigram_unigram_5000
from dataset.dataset import Dataset, Question
from typing import Dict, Tuple, List
from retrieval.index import Index
from datetime import datetime
from services import parallel
from main_constants import *
import logging
import sqlite3
import pickle

logging.basicConfig(level='INFO')
INDEX: Index


def top_5000():
    """Filter the collection of 5 million document to just the top 5000 at most according to bigram/unigram
    filtering per question. Processed in parallel."""
    global_start = datetime.now()
    global INDEX
    INDEX = Index()

    logging.info(f'{__log()}[Loading dataset in chunks.]\t[Data file: {TRAINING_SET}]\t[Chunk size: {CHUNK_SIZE}]')
    start = datetime.now()
    os.makedirs(FILTERED_DIR, exist_ok=True)
    training_set = Dataset(TRAINING_SET)
    with sqlite3.connect(BASELINE_FILTERED_DB) as db:
        cursor = db.cursor()
        cursor.execute(SQL.CREATE_TABLE)
        db.commit()
    batches = list(enumerate(parallel.chunk(CHUNK_SIZE, training_set)))
    no_batches = len(batches)
    no_docs = len(training_set)
    no_filtered = 0
    end = datetime.now()
    logging.info(f'{__log()}[Finished loading in {end - start}.]\t[Batches: {no_batches}]\t[Documents: {no_docs}]')

    logging.info(f'{__log()}[Filtering documents.][Workers: {os.cpu_count()}]')
    start = datetime.now()
    for filtered_in_batch in parallel.execute(_process_question_batch, batches):
        no_filtered += filtered_in_batch
    end = datetime.now()
    logging.info(f'{__log()}[Finished filtering in {end - start}.]\t[Filtered {no_filtered}/{no_docs}]')

    global_end = datetime.now()
    logging.info(f'{__log()}[Finished entire procedure in {global_end - global_start}.]')


def _process_question_batch(question_numbered_batch: Tuple[int, Tuple[Question]]) -> int:
    """If the batch was not previously processed, filter a batch and persist to SQLite database."""
    (batch_no, question_batch), filtered = question_numbered_batch, 0
    already_processed = _check_already_processed(question_numbered_batch)

    if len(already_processed) == len(question_batch):
        logging.info(f'{__log()}[Batch {batch_no} already processed. Skipping.]')

        return 0

    for question in question_batch:
        if already_processed.get(question.id, False):
            continue

        filtered_docs = bigram_unigram_5000(question.question, INDEX)
        _persist_filter_result(question, filtered_docs)
        filtered += 1
    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Filtered {filtered}/{len(question_batch)} questions.]')

    return filtered


def _check_already_processed(question_numbered_batch: Tuple[int, Tuple[Question]]) -> Dict[str, bool]:
    """Check if the batch is already in the database."""
    batch_no, question_batch = question_numbered_batch
    already_processed = {}
    with sqlite3.connect(BASELINE_FILTERED_DB) as db:
        for question in question_batch:
            cursor = db.cursor()
            cursor.execute(SQL.CHECK_EXISTS, (question.id,))
            exists = cursor.fetchone()
            cursor.close()
            if exists is not None:
                already_processed[question.id] = True

    return already_processed


def _persist_filter_result(question: Question, filtered_docs: List[int]):
    """Persist the results to a SQLite database"""
    with sqlite3.connect(BASELINE_FILTERED_DB) as db:
        cursor = db.cursor()
        row = (question.id, question.type, question.level, pickle.dumps(question.gold_articles),
               pickle.dumps(filtered_docs))
        cursor.execute(SQL.INSERT, row)
        cursor.close()
        db.commit()


def __log():
    """Misc logging formatting."""
    return f'[{datetime.now()}]\t[{os.getpid()}]\t'
