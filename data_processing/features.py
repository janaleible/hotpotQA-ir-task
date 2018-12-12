import copy
import os
import pickle
import sqlite3
from typing import List, Any, Tuple, Dict, Callable

from retrieval.feature_extractors.BigramOverlapFeatureExtractor import BigramOverlapFeatureExtractor
from retrieval.feature_extractors.DocumentLengthFeatureExtractor import DocumentLengthFeatureExtractor
from retrieval.feature_extractors.EntityExtractor import EntityExtractor
from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from retrieval.feature_extractors.IBM1FeatureExtractor import IBM1FeatureExtractor
from retrieval.feature_extractors.QuestionWordFeatureExtractor import QuestionWordFeatureExtractor
from services import parallel, helpers
from services.index import Index
from datetime import datetime
import main_constants as constants
import pandas as pandas
import json

INDEX: Index
EXTRACTORS: List[FeatureExtractor]
COLUMNS: List[str]


def pandas_to_db(_set: str, dataframe: pandas.DataFrame):
    if _set == 'train':
        db_path = constants.TRAIN_FEATURES_DB
    elif _set == 'dev':
        db_path = constants.DEV_FEATURES_DB
    else:
        raise ValueError(f'Unknown set. {_set}')

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        f'CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(col + " TEXT" for col in dataframe.columns.values) })')
    connection.commit()

    cursor.executemany(f'INSERT INTO features {", ".join(col + " TEXT" for col in dataframe.columns.values) } VALUES ({", ".join(["?"] * len(dataframe.columns.values))})',
                       [tuple(row) for (i, row) in dataframe.iterrows()])
    connection.commit()


def build():
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
        (constants.TRAIN_CANDIDATES_DB, constants.TRAIN_FEATURES_CHUNK),
        (constants.DEV_CANDIDATES_DB, constants.DEV_FEATURES_CHUNK)
    ]

    for (question_set_db, chunk) in iterator:

        _set = question_set_db.split("/")[-1].split(".")[1]

        start = datetime.now()

        connection = sqlite3.connect(question_set_db)
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM candidates')
        question_set = cursor.fetchall()

        total_count = 0
        _set_generator = parallel.chunk(chunk, zip([_set] * len(question_set), question_set))
        for batch_count in map(_build_candidates, _set_generator):
            total_count += batch_count

        helpers.log(f'Created {_set} candidate set with {total_count} questions in {datetime.now() - start}')


def _build_candidates(numbered_batch: Tuple[int, Tuple[str, Dict[str, Any]]]) -> int:
    start = datetime.now()

    batch_index, batch = numbered_batch
    data_frames = []
    _set = None
    for candidate_idx, (_set, (
            _id, question_id, _type, level, doc_iid, doc_wid, doc_title, question_text, doc_text, question_tokens,
            doc_tokens, tfidf, relevance)) in enumerate(batch):
        candidate_df = pandas.DataFrame(index=pandas.RangeIndex(0, len(batch)), columns=COLUMNS)

        # document -> row
        row: List[str] = [question_id, _type, level, doc_iid, doc_wid, doc_title,
                          question_text, doc_text, question_tokens, doc_tokens, tfidf]
        _extract_features(row, EXTRACTORS, json.loads(question_text), json.loads(doc_text))
        row.append(relevance)

        candidate_df.loc[candidate_idx] = row

        data_frames.append(candidate_df)

    helpers.log(f'Processed batch {batch_index} in {datetime.now() - start}')
    pandas_to_db(_set, pandas.concat(data_frames, ignore_index=True))

    return len(batch)


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
