from retrieval.term.dataset import Question
from services.index import Index
from collections import Counter
from main_constants import *
from datetime import datetime
from services import parallel, helpers
from retrieval.term import retrieve
from typing import List, Tuple
import nltk
import os

INDEX: Index
DIR_NAME = os.path.join(f'{OVERLAP_DIR}.{helpers.training_set_id()}')
DB_NAME = os.path.join(DIR_NAME, 'retrievals.sqlite')


def process():
    """Filter the collection of 5 million document to just the top 5000 at most according to bigram/unigram
    filtering per question. Processed in parallel."""
    global_start = datetime.now()
    global INDEX
    INDEX = Index(env='default')
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

        retrieval = _bigram_unigram_5000(question.question, INDEX)
        retrieve.persist_retrieval(DB_NAME, question, retrieval)
        retrieved += 1
    helpers.log(f'Retrieved questions: {retrieved}/{len(questions)}.')

    return retrieved


def _bigram_unigram_5000(query: str, index: Index, n: int = 5000) -> List[Tuple[int, float]]:
    """ Retrieves the at most n candidates from the full set of articles based on query-document pair bigram/unigram
    matches. Uses pre-built inverted index. Assumed to be equivalent to Algorithm 2, Appendix C of HotpotQA paper.
    Possible mismatches:
        -- unigram/bigrams counts in our case are considered only over first at most 500 characters of the article. Not
        clear if they use full article or not.
        -- implementation does not follow algorithm exactly since that seems very inefficient. We made it better, but
        maybe some edge-cases result in different results.

    :param query: A string of words to match.
    :param index: The prebuilt inverted index.
    :param n: The control threshold
    :return: A list of at most 5000 candidates.
    """

    # tokenize, step, filter stopwords and collect unigrams and bigrams
    tokenized_query = index.tokenize(query)
    query_unigrams = set(tokenized_query)
    query_bigrams = set(nltk.bigrams(tokenized_query))

    # count the overlapping n-gram for each query-document pair
    overlap_set = Counter()
    for bigram in query_bigrams:
        for (doc_id, _) in index.bigram_query(bigram[0], bigram[1], request=10000):
            overlap_set[doc_id] += 1
    for unigram in query_unigrams:
        for (doc_id, _) in index.unigram_query(unigram, request=10000):
            overlap_set[doc_id] += 1

    # Get the best n+1 documents and filter all the ones that have a count equal to the smallest count in the list.
    most_common = overlap_set.most_common(n + 1)
    candidates = filter(lambda t: t[1] > most_common[-1][1], most_common)

    return [candidate for candidate in candidates]
