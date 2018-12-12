import copy
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


def pandas_to_db(dataframe: pandas.DataFrame):

    connection = sqlite3.connect(constants.TRAIN_FEATURES_DB)
    cursor = connection.cursor()
    cursor.execute(f'CREATE TABLE IF NOT EXISTS features ({", ".join(col + " TEXT" for col in dataframe.columns.values) })')
    connection.commit()

    cursor.executemany(f'INSERT INTO features VALUES ({", ".join(["?"]*len(dataframe.columns.values))})',
                       [tuple(row) for (i, row) in dataframe.iterrows()])
    connection.commit()


def build():

    iterator: List[Tuple[str, str, Callable]] = [
        (constants.TRAIN_CANDIDATES_DB, constants.TRAIN_CANDIDATES_PICKLE, constants.TRAIN_FEATURES_DB, 10),
        (constants.DEV_CANDIDATES_DB, constants.DEV_CANDIDATES_PICKLE, constants.DEV_CANDIDATES_DB, 2)
    ]

    for (question_set_db, feature_pickle_path, feature_db_path, chunk) in iterator:

        _set = question_set_db.split("/")[-1].split(".")[1]

        start = datetime.now()

        connection = sqlite3.connect(question_set_db)
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM candidates')
        question_set = cursor.fetchall()

        data_frames = []

        _set_generator = parallel.chunk(constants.CHUNK_SIZE, zip([_set] * len(question_set), question_set))

        for question_candidate_df in map(_build_candidates, _set_generator):
            data_frames.append(question_candidate_df)

        candidates_df = pandas.concat(data_frames, ignore_index=True)

        pandas.to_pickle(candidates_df, feature_pickle_path, compression='gzip')
        pandas_to_db(candidates_df)

        helpers.log(f'Created {_set} candidate set in {datetime.now() - start}')


def _build_candidates(numbered_batch: Tuple[int, Tuple[str, Dict[str, Any]]]) -> pandas.DataFrame:

    start = datetime.now()

    extractors = []
    INDEX = Index('tfidf')

    if 'entity' in constants.FEATURE_EXTRACTORS:
        extractors.append(EntityExtractor(INDEX))
    if 'ibm1' in constants.FEATURE_EXTRACTORS:
        extractors.append(IBM1FeatureExtractor(normalized=False))
    if 'nibm1' in constants.FEATURE_EXTRACTORS:
        extractors.append(IBM1FeatureExtractor(normalized=True))
    if 'bigram' in constants.FEATURE_EXTRACTORS:
        extractors.append(BigramOverlapFeatureExtractor(normalized=False))
    if 'nbigram' in constants.FEATURE_EXTRACTORS:
        extractors.append(BigramOverlapFeatureExtractor(normalized=True))
    if 'qword' in constants.FEATURE_EXTRACTORS:
        extractors.append(QuestionWordFeatureExtractor())
    if 'doclen' in constants.FEATURE_EXTRACTORS:
        extractors.append(DocumentLengthFeatureExtractor())

    columns = copy.copy(constants.FEATURE_BASE_COLUMN_NAMES)
    columns.extend(feature for extractor in extractors for feature in extractor.feature_name)
    columns.append(constants.FEATURE_TARGET_COLUMN_NAME)

    batch_index, batch = numbered_batch
    data_frames = []

    for candidate_idx, (_set, (id, question_id, type, level, doc_iid, doc_wid, doc_title, question_text, doc_text, question_tokens, doc_tokens, tfidf, relevance)) in enumerate(batch):

        candidate_df = pandas.DataFrame(index=pandas.RangeIndex(0, len(batch)), columns=columns)

        # document -> row
        row: List[str] = [question_id, type, level, doc_iid, doc_wid, doc_title, question_tokens, doc_tokens, tfidf]
        _extract_features(row, extractors, json.loads(question_text), json.loads(doc_text))
        row.append(relevance)

        candidate_df.iloc[candidate_idx] = row

        data_frames.append(candidate_df)

    helpers.log(f'Processed batch {batch_index} in {datetime.now() - start}')
    return pandas.concat(data_frames, ignore_index=True)


def _extract_features(row: List[str], extractors: List[FeatureExtractor], question: str, document: str) -> None:

    features: List[str] = []

    for extractor in extractors:
        feature = extractor.extract(question, document)
        if isinstance(feature, list): features.extend(feature)
        else: features.append(json.dumps(feature))

    row.extend(features)


if __name__ == '__main__':
    build()
