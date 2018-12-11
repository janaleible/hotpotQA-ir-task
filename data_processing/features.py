import sqlite3
from typing import List, Any, Tuple, Dict, Callable

from retrieval.feature_extractors.EntityExtractor import EntityExtractor
from retrieval.feature_extractors.FeatureExtractor import FeatureExtractor
from services import parallel, helpers
from services.index import Index
from datetime import datetime
import main_constants as ct
import pandas as pd
import json


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
        dfs = []
        _set_generator = parallel.chunk(chunk, zip([_set] * len(question_set), question_set))
        for question_candidate_df in map(_build_candidates, _set_generator):
            dfs.append(question_candidate_df)
        candidates_df = pd.concat(dfs, ignore_index=True)
        pd.to_pickle(candidates_df, candidate_set_path, compression='gzip')
        helpers.log(f'Created {_set} candidate set in {datetime.now() - start}')


def _build_candidates(numbered_batch: Tuple[int, Tuple[str, Dict[str, Any]]]) -> pd.DataFrame:
    start = datetime.now()
    EXTRACTORS = []
    INDEX = Index('tfidf')
    for extractor_name in ct.EXTRACTORS:
        if extractor_name == 'entity':
            EXTRACTORS.append(EntityExtractor(INDEX))
    COLUMNS = [x for x in ct.BASE_COLUMN_NAMES]
    COLUMNS.extend([extractor.feature_name for extractor in EXTRACTORS])
    COLUMNS.extend(ct.TARGET_COLUMN_NAME)

    no, batch = numbered_batch
    dfs = []
    for _set, question in batch:
        if _set == 'train':
            no_candidates = ct.TRAIN_NO_CANDIDATES
        elif _set == 'dev':
            no_candidates = ct.DEV_NO_CANDIDATES
        else:
            raise ValueError(f'Unknown set {_set}.')

        _id = question['_id']
        _type = question['type']
        _level = question['level']
        _str = question['question']
        relevant_titles = list(map(lambda item: item[0], question['supporting_facts']))

        candidate_df = pd.DataFrame(index=pd.RangeIndex(0, no_candidates), columns=COLUMNS)

        # store relevant documents row
        relevant_doc_ids = set(INDEX.wid2int[INDEX.title2wid[title]] for title in relevant_titles)
        for (candidate_idx, doc_id) in enumerate(relevant_doc_ids):
            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
            _extract_base(row, INDEX, doc_id, _str)
            _extract_features(row, INDEX, EXTRACTORS, _str, doc_id)
            row.append(json.dumps(1))  # relevance

            candidate_df.iloc[candidate_idx] = row

        # store irrelevant documents row in order scored by tf-idf until reached candidate_set length
        result_idx = 0
        candidate_idx = ct.RELEVANT_DOCUMENTS
        results = INDEX.unigram_query(_str, no_candidates)
        while candidate_idx < no_candidates:
            (doc_id, _) = results[result_idx]
            title = INDEX.wid2title[INDEX.int2wid[doc_id]]
            target = int(title in relevant_titles)
            if target == 1:
                result_idx += 1
                continue

            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
            _extract_base(row, INDEX, doc_id, _str)
            _extract_features(row, INDEX, EXTRACTORS, _str, doc_id)
            row.append(json.dumps(target))

            candidate_df.iloc[candidate_idx] = row
            candidate_idx += 1
            result_idx += 1

        dfs.append(candidate_df)

    helpers.log(f'Processed batch {no} in {datetime.now() - start}')
    return pd.concat(dfs, ignore_index=True)


def _extract_base(row: List[str], index: Index, doc_id: int, question: str) -> None:
    document = index.get_document_by_int_id(doc_id)
    query = index.tokenize(question)
    query = [index.token2id[token] for token in query]
    row.extend([json.dumps(doc_id), json.dumps(query), json.dumps(document)])


def _extract_features(row: List[str], index: Index, extractors: List[FeatureExtractor], question: str, doc_id: int) -> None:
    features: List[str] = []
    doc_wid = index.int2wid[doc_id]
    # (doc_str,) = df[]
    doc_str = "Anarchism is a political philosophy  that advocates self-governed  societies based on voluntary institutions. 0eos0  These are often described as stateless societies , although several authors have defined them more specifically as institutions based on non-hierarchical  free associations . 0eos0  Anarchism holds the state  to be undesirable, unnecessary and harmful. 0eop0 While anti-statism  is central, anarchism specifically entails opposing authority  or hierarchical organisation  in the conduct of all human relations, including--but not limited to--the state system. 0eos0  Anarchism is usually considered a far-left  ideology and much of anarchist economics  and anarchist legal philosophy  reflects anti-authoritarian interpretations  of communism , collectivism , syndicalism , mutualism  or participatory economics ."
    for extractor in extractors:
        feature = json.dumps(extractor.extract(question, doc_str))
        features.append(feature)

    row.extend(features)


if __name__ == '__main__':
    build()
