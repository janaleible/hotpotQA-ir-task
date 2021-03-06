import copy
import os
import sqlite3
from typing import List, Any, Tuple, Dict, Callable

from retrieval.feature_extractors.BigramOverlapFeatureExtractor import BigramOverlapFeatureExtractor
from retrieval.feature_extractors.DocumentLengthFeatureExtractor import DocumentLengthFeatureExtractor
from retrieval.feature_extractors.EntityExtractor import EntityExtractor
from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from retrieval.feature_extractors.IBM1FeatureExtractor import IBM1FeatureExtractor
from retrieval.feature_extractors.QuestionWordFeatureExtractor import QuestionWordFeatureExtractor
from services import parallel, helpers, sql
from services.index import Index
from datetime import datetime
import main_constants as constants
import json

INDEX: Index
EXTRACTORS: List[FeatureExtractor]
COLUMNS: List[str]


def rows_to_db(_set: str, rows: List[Any]):
    if _set == 'train':
        db_path = constants.TRAIN_FEATURES_DB
    elif _set == 'dev':
        db_path = constants.DEV_FEATURES_DB
    elif _set == 'test':
        db_path = constants.TEST_FEATURES_DB
    else:
        raise ValueError(f'Unknown set. {_set}')
    done = False
    while not done:
        try:
            connection = sqlite3.connect(db_path)
            cursor = connection.cursor()
            cursor.executemany(sql.insert_features(COLUMNS), [tuple(row) for row in rows])
            connection.commit()
            cursor.close()
            connection.close()
            done = True
        except Exception as e:
            helpers.log(e)


def build():
    assert constants.TRAIN_FEATURES_CHUNK > 1
    assert constants.DEV_FEATURES_CHUNK > 1

    global INDEX
    INDEX = Index('tfidf')
    helpers.log('Loaded index.')

    global EXTRACTORS
    EXTRACTORS = []
    if 'entity' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(EntityExtractor(INDEX))
    if 'ibm1' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(IBM1FeatureExtractor(normalized=False))
    if 'nibm1' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(IBM1FeatureExtractor(normalized=True))
    if 'bigram' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(BigramOverlapFeatureExtractor(normalized=False))
    if 'nbigram' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(BigramOverlapFeatureExtractor(normalized=True))
    if 'qword' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(QuestionWordFeatureExtractor())
    if 'doclen' in constants.FEATURE_EXTRACTORS:
        EXTRACTORS.append(DocumentLengthFeatureExtractor())
    helpers.log('Loaded extractors.')

    global COLUMNS
    COLUMNS = copy.copy(constants.FEATURE_BASE_COLUMN_NAMES)
    COLUMNS.extend(feature for extractor in EXTRACTORS for feature in extractor.feature_name)
    COLUMNS.append(constants.FEATURE_TARGET_COLUMN_NAME)
    helpers.log('Loaded column names.')

    os.makedirs(constants.FEATURES_DIR, exist_ok=True)
    iterator: List[Tuple[str, str, Callable]] = [
        # (constants.TRAIN_CANDIDATES_DB, constants.TRAIN_FEATURES_DB, constants.TRAIN_FEATURES_CHUNK),
        # (constants.DEV_CANDIDATES_DB, constants.DEV_FEATURES_DB, constants.DEV_FEATURES_CHUNK),
        (constants.TEST_CANDIDATES_DB, constants.TEST_FEATURES_DB, constants.TEST_FEATURES_CHUNK)
    ]

    for (candidate_db_path, feature_db_path, chunk) in iterator:
        start_time = datetime.now()
        _set = candidate_db_path.split("/")[-1].split(".")[1]

        done = False
        feature_db = sqlite3.connect(feature_db_path)
        cursor = feature_db.cursor()
        while not done:
            try:
                cursor.execute(sql.create_features_table(COLUMNS))
                feature_db.commit()
                done = True
            except Exception as e:
                helpers.log(e)
        helpers.log(f'Created {_set} features table.')

        (start,) = cursor.execute('SELECT MAX(id) FROM features').fetchone()
        start = start if start is not None else 0  # first id in the database
        cursor.close()
        feature_db.close()
        helpers.log(f'Starting feature build at {start}.')

        candidate_db = sqlite3.connect(candidate_db_path)
        cursor = candidate_db.cursor()
        (stop,) = cursor.execute('SELECT COUNT(*) FROM candidates').fetchone()  # last id in the database
        cursor.close()
        candidate_db.close()
        id_range = range(start + 1, stop + 1)
        helpers.log(f'Retrieved {len(id_range)} candidate indices for {_set} set.')

        total_count = 0
        _set_generator = parallel.chunk(chunk, zip([_set] * len(id_range), id_range))
        _batched_set_generator = parallel.chunk(constants.GRAND_CHUNK, _set_generator)
        for grand_batch_idx, _batch_set in _batched_set_generator:
            grand_batch_count = 0
            for batch_count in parallel.execute(_build_candidates, _batch_set, _as='process'):
                grand_batch_count += batch_count
                total_count += batch_count
            helpers.log(f'Processed {_set} batch of features no {grand_batch_idx} with {grand_batch_count} pairs.')
        helpers.log(f'Created {_set} features set with {total_count} pairs in {datetime.now() - start_time}')


def _build_candidates(numbered_batch: Tuple[int, Tuple[str, Dict[str, Any]]]) -> int:
    try:
        start_time = datetime.now()

        batch_index, batch = numbered_batch
        _set, start = batch[0]
        _, stop = batch[-1]

        if _set == 'train':
            candidate_db_path = constants.TRAIN_CANDIDATES_DB
            feature_db_path = constants.TRAIN_FEATURES_DB
        elif _set == 'dev':
            candidate_db_path = constants.DEV_CANDIDATES_DB
            feature_db_path = constants.DEV_FEATURES_DB
        elif _set == 'test':
            candidate_db_path = constants.TEST_CANDIDATES_DB
            feature_db_path = constants.TEST_FEATURES_DB
        else:
            raise ValueError(f'Unknown dataset {_set}.')
        done = False

        candidate_db = sqlite3.connect(candidate_db_path)
        candidate_cursor = candidate_db.cursor()
        candidate_rows = candidate_cursor.execute(sql.fetch_candidate_batch, (start, stop)).fetchall()
        candidate_cursor.close()
        candidate_db.close()

        batch_count = 0
        rows = []
        feature_db = sqlite3.connect(feature_db_path)
        feature_cursor = feature_db.cursor()
        for candidate_row in candidate_rows:
            (_id, question_id, _type, level, doc_iid, doc_wid, doc_title,
             question_text, doc_text, question_tokens, doc_tokens, tfidf, relevance) = candidate_row
            exists = feature_cursor.execute('SELECT id FROM features WHERE id = ?', (_id,)).fetchone()
            if exists is not None:
                continue

            row: List[str] = [_id, question_id, _type, level, doc_iid, doc_wid, doc_title,
                              question_text, doc_text, question_tokens, doc_tokens, tfidf]
            _extract_features(row, EXTRACTORS, json.loads(question_text), json.loads(doc_text))
            row.append(relevance)
            rows.append(row)
            batch_count += 1
        rows_to_db(_set, rows)
        helpers.log(f'Processed batch {batch_index} of {batch_count} pairs in {datetime.now() - start_time}')

        feature_cursor.close()
        feature_db.close()

        return batch_count
    except Exception as e:
        helpers.log(e)


def _extract_features(row: List[str], extractors: List[FeatureExtractor], question: str, document: str) -> None:
    features: List[str] = []

    for extractor in extractors:
        feature = extractor.extract(question, document)
        if isinstance(feature, list):
            features.extend(feature)
        else:
            features.append(json.dumps(feature))

    row.extend(features)


if __name__ == '__main__':
    build()
