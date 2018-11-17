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
CHUNK_SIZE = 20


def accuracy(true_positives: int, all_observations: int) -> float:
    return true_positives / all_observations


def _process_question_batch(question_batch: List[Question]) -> numpy.ndarray:
    index = Index()
    comparison = 0
    bridge = 1
    hard = 0
    medium = 1
    easy = 2
    found_articles = numpy.zeros((2, 3, 3))

    data = []
    with sqlite3.connect(constants.FILTERED_DB) as db:
        cursor = db.cursor()

        for question in question_batch:
            gold_article_ids = set()
            [[gold_article_ids.add(index.external2internal(idx)) for idx in index.title2wid[title]] for title in
             question.gold_articles]

            filtered_articles = unigram_bigram_filter(question.question, index)

            number_of_articles_found = len(gold_article_ids.intersection(filtered_articles))

            question_type = comparison if question.type == 'comparison' else bridge
            question_level = hard if question.level == 'hard' else medium if question.level == 'medium' else easy

            found_articles[question_type][question_level][number_of_articles_found] += 1

            try:
                cursor.execute(constants.SQL.INSERT, (question.id, question.type, question.level,
                                                      pickle.dumps(question.gold_articles),
                                                      pickle.dumps(filtered_articles)))
                db.commit()
            except Exception as e:
                print((question.id, question.type, question.level,
                       pickle.dumps(question.gold_articles),
                       pickle.dumps(filtered_articles)))
                print(e)

            data.append((question.id, question.type, question.level,
                         pickle.dumps(question.gold_articles),
                         pickle.dumps(filtered_articles)))

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Done filtering {len(question_batch)} questions.]')

    return found_articles


def filter_top_5000():
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

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Starting filtering.]')
    for found_articles_batch in parallel.execute(_process_question_batch, parallel.chunk(CHUNK_SIZE, training_set)):
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
    try:
        filter_top_5000()
    except Exception as error:
        INDEX.index.close()
        raise
