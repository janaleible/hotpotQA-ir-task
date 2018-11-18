from dataset.dataset import Dataset, Question
from typing import Set, List, Tuple
from retrieval.index import Index
from datetime import datetime
from main_constants import *
from tqdm import tqdm
import numpy as np
import logging
import sqlite3
import pickle

logging.basicConfig(level='INFO')
INDEX: Index
COMPARISON = 0
BRIDGE = 1
HARD = 0
MEDIUM = 1
EASY = 2


def accuracies(dataset: str, database: str):
    """Compute accuracy over a dataset given a retrieval database.
    Statistics are split over question types and levels."""
    dataset = Dataset(dataset)
    global INDEX
    INDEX = Index()

    # shape: (question type [comparison, bridge], level [hard, medium, easy], number of found gold articles [0, 1, 2])
    hits = np.zeros((2, 3, 3))
    with sqlite3.connect(database) as conn:
        for question in tqdm(dataset, unit='questions'):
            target = _extract_target(question)
            prediction = _fetch_prediction(conn, question)
            _update_hits(prediction, target, question, hits)

        full, half, full_hard, full_medium, full_easy, full_comparison, full_bridge = _accuracies(hits)

    logging.info(f'[{datetime.now()}]\t[Full Accuracy: {round(full, 4)}]')
    logging.info(f'[{datetime.now()}]\t[Half Accuracy: {round(half, 4)}]')
    logging.info(f'[{"-" * 10}]')
    logging.info(f'[{datetime.now()}]\t[Easy Question Full Accuracy: {round(full_easy, 4)}]')
    logging.info(f'[{datetime.now()}]\t[Medium Question Full Accuracy: {round(full_medium, 4)}]')
    logging.info(f'[{datetime.now()}]\t[Hard Question Full Accuracy: {round(full_hard, 4)}]')
    logging.info(f'[{"-" * 10}]')
    logging.info(f'[{datetime.now()}]\t[Comparison Question Full Accuracy: {round(full_comparison, 4)}]')
    logging.info(f'[{datetime.now()}]\t[Bridge Question Full Accuracy: {round(full_bridge, 4)}]')


def _extract_target(question: Question) -> Set[int]:
    """Extract indri internal ids of target documents."""
    target = set()
    for title in question.gold_articles:
        for external_doc_id in INDEX.title2wid[title]:
            target.add(INDEX.external2internal(external_doc_id))

    return target


def _fetch_prediction(conn: sqlite3.Connection, question: Question) -> List[int]:
    cursor = conn.cursor()
    cursor.execute(SQL.FETCH_ONE_RESULT, (question.id,))
    (prediction,) = cursor.fetchone()

    return pickle.loads(prediction)


def _update_hits(prediction: List[int], target: Set[int], question: Question, hits: np.ndarray) -> None:
    """Compute hits@x split by level, type, and overlap count between target and prediction."""
    overlap_count = len(target.intersection(prediction))
    question_type = COMPARISON if question.type == 'comparison' else BRIDGE
    question_level = HARD if question.level == 'hard' else MEDIUM if question.level == 'medium' else EASY
    hits[question_type][question_level][overlap_count] += 1

    return


def _accuracies(hits: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Compute accuracy statistics across all questions split by type, level, and overlap count."""
    full = _accuracy(hits[:, :, 2].sum(), hits[:, :, :].sum())
    half = _accuracy(hits[:, :, 1:].sum(), hits[:, :, :].sum())
    full_hard = _accuracy(hits[:, HARD, 2].sum(), hits[:, HARD, :].sum())
    full_medium = _accuracy(hits[:, MEDIUM, 2].sum(), hits[:, MEDIUM, :].sum())
    full_easy = _accuracy(hits[:, EASY, 2].sum(), hits[:, EASY, :].sum())
    full_comparison = _accuracy(hits[COMPARISON, :, 2].sum(), hits[COMPARISON, :, :].sum())
    full_bridge = _accuracy(hits[BRIDGE, :, 2].sum(), hits[BRIDGE, :, :].sum())

    return full, half, full_hard, full_medium, full_easy, full_comparison, full_bridge


def _accuracy(true_positives: int, all_observations: int) -> float:
    """Calculate single accuracy value."""
    if all_observations == 0:
        return 0.0

    return true_positives / all_observations
