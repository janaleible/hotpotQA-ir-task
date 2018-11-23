from dataset.dataset import Question, Dataset
from typing import List, Tuple, Dict
from services import parallel, sql
from datetime import datetime
from services import helpers
from main_constants import *
import sqlite3
import pickle


def load_dataset_batches() -> Tuple[List[List[Question]], int, int]:
    """Load the dataset in batches of ``CHUNK_SIZE`` and calculate lengths."""
    helpers.log(f'Loading dataset in chunks. Data file: {TRAINING_SET}. Chunk size: {CHUNK_SIZE}.')
    start = datetime.now()

    training_set = Dataset.from_file(TRAINING_SET)
    batches = parallel.chunk(CHUNK_SIZE, training_set.questions)
    no_batches = len(batches)
    no_docs = len(training_set)

    end = datetime.now()
    helpers.log(f'Finished loading in {end - start}. Batches: {no_batches}. Documents: {no_docs}.')

    return batches, no_batches, no_docs


def create_retrieval_db(db_name: str) -> None:
    """Create the sqlite database where retrieval results will be persisted."""
    helpers.log(f'Creating retrieval SQL table. Database: {db_name}')
    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()
        cursor.execute(sql.create_table())
        db.commit()

    return


def persist_retrieval(db_name: str, question: Question, filtered_docs: List[Tuple[int, float]]) -> None:
    """Persist the results to a SQLite database"""

    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()
        row = (question.id, question.type, question.level, pickle.dumps(question.gold_articles),
               pickle.dumps(filtered_docs))
        cursor.execute(sql.insert_row(), row)
        cursor.close()
        db.commit()

    return


def check_already_processed(db_name: str, question_numbered_batch: Tuple[int, Tuple[Question]]) -> Dict[str, bool]:
    """Check if the batch is already in the database."""
    batch_no, question_batch = question_numbered_batch
    already_processed = {}
    with sqlite3.connect(db_name) as db:
        for question in question_batch:
            cursor = db.cursor()
            cursor.execute(sql.get_question_id(), (question.id,))
            exists = cursor.fetchone()
            cursor.close()
            if exists is not None:
                already_processed[question.id] = True

    return already_processed
