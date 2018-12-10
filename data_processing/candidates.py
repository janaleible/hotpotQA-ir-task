from typing import List, Any, Tuple, Dict, Callable
from services import parallel, helpers
from services.index import Index
from datetime import datetime
import main_constants as ct
import pandas as pd
import json

INDEX: Index
EXTRACTORS = List[Callable]


def build(extractors: List[Callable]):
    global INDEX
    INDEX = Index('tfidf')
    global EXTRACTORS
    EXTRACTORS = [extractor(INDEX) for extractor in extractors]
    ct.BASE_COLUMN_NAMES.extend(extractor.feature_name for extractor in EXTRACTORS)

    iterator: List[Tuple[str, str, Callable]] = [
        (ct.TRAIN_HOTPOT_SET, ct.TRAIN_UNIGRAM_TFIDF_CANDIDATES),
        (ct.DEV_HOTPOT_SET, ct.DEV_UNIGRAM_TFIDF_CANDIDATES)
    ]
    for (question_set_path, candidate_set_path) in iterator:
        _set = question_set_path.split("/")[-1].split("_")[0]
        start = datetime.now()
        with open(question_set_path, 'r') as file:
            question_set = json.load(file)
        dfs = []
        _set_generator = enumerate(zip([_set] * len(question_set), question_set))
        for question_candidate_df in parallel.execute(_build_candidates, _set_generator):
            dfs.append(question_candidate_df)
        candidates_df = pd.concat(dfs)
        pd.to_pickle(candidates_df, candidate_set_path, compression='gzip')
        helpers.log(f'Created {_set} candidate set in {datetime.now() - start}')


def get_question_df_first(enumerated_question):
    no, question = enumerated_question
    _id = question['_id']
    _question = question['question']
    _relevant_titles = map(lambda item: item[0], question['supporting_facts'])

    candidate_df = pd.DataFrame(index=pd.RangeIndex(0 + no * 1000, (no + 1) * 1000),
                                columns=['question_id', 'document_id', 'question', 'document', 'target'])
    _results = INDEX.unigram_query(_question, 1000)
    for (_idx, _result) in enumerate(_results):
        (_doc_id, _) = _result
        _document = list(INDEX.get_document_by_int_id(_doc_id))

        _title = INDEX.wid2title[INDEX.int2wid[_doc_id]]
        _target = int(_title in _relevant_titles)

        _query = INDEX.tokenize(_question)
        _query = [INDEX.token2id[token] for token in _query]

        candidate_df.iloc[_idx] = [_id, _doc_id, json.dumps(_query), json.dumps(_document), _target]

    helpers.log(f'Processed question {no}')

    return candidate_df


def _build_candidates(enumerated_question: Tuple[int, Tuple[str, Dict[str, Any]]]) -> pd.DataFrame:
    no, (_set, question) = enumerated_question
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
    candidate_df = pd.DataFrame(index=pd.RangeIndex(0 + no * no_candidates, (no + 1) * no_candidates),
                                columns=ct.BASE_COLUMN_NAMES)

    # store relevant documents row
    relevant_doc_ids = set(INDEX.wid2int[INDEX.title2wid[title]] for title in relevant_titles)
    for (candidate_idx, doc_id) in enumerate(relevant_doc_ids):
        row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
        document = _extract_base(row, doc_id, _str)
        _extract_features(row, _str, document)
        row.append(json.dumps(1))  # relevance

        candidate_df.iloc[candidate_idx] = row

    # store irrelevant documents partial row in order scored by tf-idf until reached candidate_set length
    result_idx = 0
    candidate_idx = ct.RELEVANT_DOCUMENTS
    results = INDEX.unigram_query(_str, 1000)
    while candidate_idx < no_candidates:
        (doc_id, _) = results[result_idx]
        title = INDEX.wid2title[INDEX.int2wid[doc_id]]
        target = int(title in relevant_titles)
        if target == 1:
            result_idx += 1
            continue

        row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
        document = _extract_base(row, doc_id, _str)
        _extract_features(row, _str, document)
        row.append(json.dumps(target))

        candidate_df.iloc[candidate_idx] = row
        candidate_idx += 1
        result_idx += 1

    helpers.log(f'Processed question {no}')

    return candidate_df


def _extract_base(row: List[str], doc_id: int, question: str) -> Tuple[int, ...]:
    document = INDEX.get_document_by_int_id(doc_id)
    query = INDEX.tokenize(question)
    query = [INDEX.token2id[token] for token in query]
    row.extend([json.dumps(doc_id), json.dumps(query), json.dumps(document)])

    return document


def _extract_features(row: List[str], question: str, document: Tuple[int]) -> None:
    features: List[str] = {}
    for extractor in EXTRACTORS:
        feature = json.dumps(extractor.extract(question, INDEX.doc_str(document, True, False)))
        features.append(feature)

    row.extend(features)


if __name__ == '__main__':
    build([])
