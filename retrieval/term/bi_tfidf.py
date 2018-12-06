import nltk

from services import parallel, helpers
from retrieval.term.dataset import Question
from services.index import Index
from datetime import datetime
from main_constants import *
from retrieval import retrieve
from typing import Tuple, List
import collections as cl
import os

INDEX: Index
DIR_NAME = os.path.join(f'{BIGRAM_TFIDF_DIR}.{helpers.training_set_id()}')
DB_NAME = os.path.join(DIR_NAME, 'retrievals.sqlite')


def process():
    """Filter the collection of 5 million document to just the top 5000 at most according to bigram/unigram
    filtering per question. Processed in parallel."""
    global_start = datetime.now()
    global INDEX
    INDEX = Index(env='tfidf')
    os.makedirs(DIR_NAME)
    (batches, no_batches, no_queries), total_retrieved = retrieve.load_dataset_batches(), 0
    retrieve.create_retrieval_db(DB_NAME)

    helpers.log(f'Retrieving documents. Workers: {os.cpu_count()}')
    start = datetime.now()
    for batch_retrieval in parallel.execute(_process_question_batch, batches):
        total_retrieved += batch_retrieval
    end = datetime.now()
    helpers.log(f'Finished retrieval in {end - start}. Filtered {total_retrieved}/{no_queries}')

    global_end = datetime.now()
    helpers.log(f'Finished process in {global_end - global_start}.')


def _process_question_batch(question_numbered_batch: Tuple[int, Tuple[Question]]) -> int:
    """If the batch was not previously processed, filter a batch and persist to SQLite database."""
    (no, questions), retrieved = question_numbered_batch, 0

    already_processed = retrieve.check_already_processed(DB_NAME, question_numbered_batch)
    if len(already_processed) == len(questions):
        helpers.log(f'Batch {no} already processed. Skipping.')
        return 0

    for question in questions:
        if already_processed.get(question.id, False):
            continue
        retrieval = _full_bigram_query(question)
        retrieve.persist_retrieval(DB_NAME, question, retrieval)
        retrieved += 1
        helpers.log(f'Retrieved questions: {retrieved}/{len(questions)}.')

    return retrieved


def _full_bigram_query(question: Question) -> List[Tuple[int, float]]:
    results = cl.defaultdict(float)
    for bigram in nltk.bigrams(INDEX.tokenize(question.question)):
        first, second = bigram
        result = INDEX.bigram_query(first, second, request=5000)
        for (_id, score) in result:
            results[_id] += score

    return sorted(list(zip(results.keys(), results.values())), key=lambda x: x[1], reverse=True)
