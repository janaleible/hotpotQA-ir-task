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
import main_constants as ct
import pandas as pandas
import json


def pandas_to_db(dataframe: pandas.DataFrame):

    connection = sqlite3.connect(ct.FEATURE_EXTRACTION_DB)
    cursor = connection.cursor()
    cursor.execute(f'CREATE TABLE IF NOT EXISTS features ({", ".join(col + " TEXT" for col in dataframe.columns.values) })')
    connection.commit()

    cursor.executemany(f'INSERT INTO features VALUES ({", ".join(["?"]*len(dataframe.columns.values))})',
                       [tuple(row) for (i, row) in dataframe.iterrows()])
    connection.commit()


def build():

    iterator: List[Tuple[str, str, Callable]] = [
        (ct.TRAIN_HOTPOT_SET, ct.TRAIN_CANDIDATES_DB, 10),
        (ct.DEV_HOTPOT_SET, ct.DEV_CANDIDATES_DB, 2)
    ]

    for (question_set_path, candidate_set_path, chunk) in iterator:

        _set = question_set_path.split("/")[-1].split("_")[0]

        start = datetime.now()

        with open(question_set_path, 'r') as file:
            question_set = json.load(file)

        data_frames = []

        _set_generator = parallel.chunk(chunk, zip([_set] * len(question_set), question_set))

        for question_candidate_df in map(_build_candidates, _set_generator):
            data_frames.append(question_candidate_df)

        candidates_df = pandas.concat(data_frames, ignore_index=True)

        pandas.to_pickle(candidates_df, candidate_set_path, compression='gzip')
        pandas_to_db(candidates_df)

        helpers.log(f'Created {_set} candidate set in {datetime.now() - start}')


def _build_candidates(numbered_batch: Tuple[int, Tuple[str, Dict[str, Any]]]) -> pandas.DataFrame:

    start = datetime.now()

    extractors = []
    INDEX = Index('tfidf')

    if 'entity' in ct.EXTRACTORS:
        extractors.append(EntityExtractor(INDEX))
    if 'ibm1' in ct.EXTRACTORS:
        extractors.append(IBM1FeatureExtractor(normalized=False))
    if 'nibm1' in ct.EXTRACTORS:
        extractors.append(IBM1FeatureExtractor(normalized=True))
    if 'bigram' in ct.EXTRACTORS:
        extractors.append(BigramOverlapFeatureExtractor(normalized=False))
    if 'nbigram' in ct.EXTRACTORS:
        extractors.append(BigramOverlapFeatureExtractor(normalized=True))
    if 'qword' in ct.EXTRACTORS:
        extractors.append(QuestionWordFeatureExtractor())
    if 'doclen' in ct.EXTRACTORS:
        extractors.append(DocumentLengthFeatureExtractor())


    columns = copy.copy(ct.BASE_COLUMN_NAMES)
    columns.extend(feature for extractor in extractors for feature in extractor.feature_name)
    columns.append(ct.TARGET_COLUMN_NAME)

    batch_index, batch = numbered_batch
    data_frames = []

    for _set, question in batch:
        if _set == 'train':
            number_of_candidates = ct.TRAIN_NO_CANDIDATES
        elif _set == 'dev':
            number_of_candidates = ct.DEV_NO_CANDIDATES
        else:
            raise ValueError(f'Unknown set {_set}.')

        _id = question['_id']
        _type = question['type']
        _level = question['level']
        _str = question['question']
        relevant_titles = list(map(lambda item: item[0], question['supporting_facts']))

        candidate_df = pandas.DataFrame(index=pandas.RangeIndex(0, number_of_candidates), columns=columns)

        # store relevant documents row
        relevant_doc_ids = set(INDEX.wid2int[INDEX.title2wid[title]] for title in relevant_titles)
        for (candidate_idx, doc_id) in enumerate(relevant_doc_ids):
            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level), json.dumps(doc_id)]
            _extract_base(row, INDEX, doc_id, _str)
            _extract_features(row, INDEX, extractors, _str, doc_id)
            row.append(json.dumps(1))  # relevance

            candidate_df.iloc[candidate_idx] = row

        # store irrelevant documents row in order scored by tf-idf until reached candidate_set length
        result_idx = 0
        candidate_idx = ct.RELEVANT_DOCUMENTS
        results = INDEX.unigram_query(_str, number_of_candidates)

        while candidate_idx < number_of_candidates:
            (doc_id, _) = results[result_idx]
            title = INDEX.wid2title[INDEX.int2wid[doc_id]]
            target = int(title in relevant_titles)
            if target == 1:
                result_idx += 1
                continue


            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level), json.dumps(doc_id)]
            _extract_base(row, INDEX, doc_id, _str)
            _extract_features(row, INDEX, extractors, _str, doc_id)
            row.append(json.dumps(target))

            candidate_df.iloc[candidate_idx] = row
            candidate_idx += 1
            result_idx += 1

        data_frames.append(candidate_df)

    helpers.log(f'Processed batch {batch_index} in {datetime.now() - start}')
    return pandas.concat(data_frames, ignore_index=True)



def _extract_base(row: List[str], index: Index, doc_id: int, question: str) -> None:

    document_tokens = index.get_document_by_int_id(doc_id)
    doc_wid = index.int2wid[doc_id]
    doc_title = index.wid2title[doc_wid]

    query = index.tokenize(question)
    query_tokens = [index.token2id[token] for token in query]

    row.extend([json.dumps(doc_wid), json.dumps(doc_title), json.dumps(query_tokens), json.dumps(document_tokens)])


def _extract_features(row: List[str], index: Index, extractors: List[FeatureExtractor], question: str, doc_id: int) -> None:

    features: List[str] = []

    doc_wid = index.int2wid[doc_id]
    doc_title = index.wid2title[doc_wid]

    doc_str = index.get_pretty_document_by_title(doc_title)

    for extractor in extractors:
        feature = extractor.extract(question, doc_str)
        if isinstance(feature, list): features.extend(feature)
        else: features.append(json.dumps(feature))

    row.extend(features)


if __name__ == '__main__':
    build()
