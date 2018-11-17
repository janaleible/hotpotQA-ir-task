import logging
import os
import pickle
import sqlite3
from datetime import datetime
from typing import List
import numpy
import constants
from dataset.Dataset import Dataset, Question
from retrieval.filtering import unigram_bigram_filter
from retrieval.index import Index
from services import parallel

logging.basicConfig(level='INFO')


def accuracy(true_positives: int, all_observations: int) -> float:
    if all_observations == 0 and true_positives == 0:
        return 0

    return true_positives / all_observations


def _process_question_batch(question_numbered_batch: List[Question]) -> numpy.ndarray:
    comparison = 0
    bridge = 1
    hard = 0
    medium = 1
    easy = 2
    found_articles = numpy.zeros((2, 3, 3))

    batch_no, question_batch = question_numbered_batch
    index_no = batch_no % constants.NO_INDEXES

    already_processed = {}
    with sqlite3.connect(constants.FILTERED_DB) as db:
        cursor = db.cursor()
        cursor.execute(constants.SQL.CHECK_EXISTS.format(tuple(map(lambda q: q['id'], question_batch))))
        results = cursor.fetchall()
        if len(results) == len(question_batch):
            logging.info(
                f'[{datetime.now()}]\t[{os.getpid()}]\t[Batch {batch_no} already processed. Results size: {len(results)}. Batch size {len(question_batch)} Skipping.]')
            return found_articles
        for (q_id,) in results:
            already_processed[q_id] = True
        cursor.close()

    index = Index(index_no)
    filtered = 0
    for question in question_batch:
        if already_processed.get(question['id'], False):
            continue

        gold_article_ids = set()
        [[gold_article_ids.add(index.external2internal(idx)) for idx in index.title2wid[title]] for title in
         question['gold']]

        filtered_articles = unigram_bigram_filter(question['text'], index)

        number_of_articles_found = len(gold_article_ids.intersection(filtered_articles))

        question_type = comparison if question['type'] == 'comparison' else bridge
        question_level = hard if question['level'] == 'hard' else medium if question['level'] == 'medium' else easy

        found_articles[question_type][question_level][number_of_articles_found] += 1
        with sqlite3.connect(constants.FILTERED_DB) as db:
            cursor = db.cursor()
            cursor.execute(constants.SQL.INSERT, (question['id'], question['type'], question['level'],
                                                  pickle.dumps(question['gold']),
                                                  pickle.dumps(filtered_articles)))
            cursor.close()
            db.commit()
        filtered += 1
    del index
    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Done filtering {filtered} questions.]')

    return found_articles


def filter_top_5000():
    logging.info(f'{constants.TRAINING_SET}, {constants.CHUNK_SIZE}')
    start = datetime.now()
    training_set = Dataset(constants.TRAINING_SET)
    # indices for results array
    comparison = 0
    bridge = 1
    hard = 0
    medium = 1
    easy = 2

    # shape: (question type [comparison, bridge], level [hard, medium, easy], number of found gold articles [0, 1, 2])
    found_articles = numpy.zeros((2, 3, 3))

    os.makedirs(constants.FILTERED_DIR, exist_ok=True)
    with sqlite3.connect(constants.FILTERED_DB) as db:
        cursor = db.cursor()
        cursor.execute(constants.SQL.CREATE_TABLE)
        db.commit()
    data = []
    for question in training_set:
        data.append({
            'id': question.id,
            'text': question.question,
            'answer': question.answer,
            'level': question.level,
            'type': question.type,
            'gold': question.gold_articles,
            'context': question.context
        })

    batches = list(enumerate(parallel.chunk(constants.CHUNK_SIZE, data)))

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Starting filtering.]')
    for found_articles_batch in parallel.execute(_process_question_batch, batches):
        found_articles += found_articles_batch

    fully_accurate = accuracy(int(numpy.sum(found_articles[:, :, 2])), int(numpy.sum(found_articles[:, :, :])))
    somewhat_accurate = accuracy(int(numpy.sum(found_articles[:, :, 1:])), int(numpy.sum(found_articles[:, :, :])))
    accurate_hard_questions = accuracy(int(numpy.sum(found_articles[:, hard, 2])),
                                       int(numpy.sum(found_articles[:, hard, :])))
    accurate_medium_questions = accuracy(int(numpy.sum(found_articles[:, medium, 2])),
                                         int(numpy.sum(found_articles[:, medium, :])))
    accurate_easy_questions = accuracy(int(numpy.sum(found_articles[:, easy, 2])),
                                       int(numpy.sum(found_articles[:, easy, :])))
    accurate_comparison_questions = accuracy(int(numpy.sum(found_articles[comparison, :, 2])),
                                             int(numpy.sum(found_articles[comparison, :, :])))
    accurate_bridge_questions = accuracy(int(numpy.sum(found_articles[bridge, :, 2])),
                                         int(numpy.sum(found_articles[bridge, :, :])))

    print(f'Finished processing in {datetime.now() - start}')
    print(f'Fully accurate: {round(fully_accurate, 4)}')
    print(f'Somewhat accurate: {round(somewhat_accurate, 4)}')
    print('-' * 10)
    print(f'easy question accuracy: {round(accurate_easy_questions, 4)}')
    print(f'medium question accuracy: {round(accurate_medium_questions, 4)}')
    print(f'hard question accuracy: {round(accurate_hard_questions, 4)}')
    print('-' * 10)
    print(f'comparison question accuracy: {round(accurate_comparison_questions, 4)}')
    print(f'bridge question accuracy: {round(accurate_bridge_questions, 4)}')

    numpy.save(constants.FILTER_RESULTS, found_articles)


if __name__ == '__main__':
    filter_top_5000()
