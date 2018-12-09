import json
import pandas as pd

from services import parallel, helpers
from services.index import Index

INDEX: Index


def build():
    global INDEX
    INDEX = Index('tfidf')
    with open('./data/hotpot/dev_dummy.json', 'r') as file:
        question_set = json.load(file)
    dfs = []
    for question_candidate_df in parallel.execute(get_question_df_second, enumerate(question_set)):
        dfs.append(question_candidate_df)
    candidates_df = pd.concat(dfs)
    pd.to_pickle(candidates_df, './data/candidates.tfidf.gzip', compression='gzip')


def get_question_df_first(enumerated_question):
    no, question = enumerated_question
    _id = question['_id']
    _question = question['question']
    _relevant_titles = map(lambda item: item[0], question['supporting_facts'])

    candidate_df = pd.DataFrame(index=pd.RangeIndex(0 + no * 1000, (no+1) * 1000),
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


def get_question_df_second(enumerated_question):
    no, question = enumerated_question
    _id = question['_id']
    _question = question['question']
    _relevant_titles = list(map(lambda item: item[0], question['supporting_facts']))

    candidate_df = pd.DataFrame(index=pd.RangeIndex(0 + no * 1000, (no + 1) * 1000),
                                columns=['question_id', 'document_id', 'question', 'document', 'target'])

    relevant_doc_ids = set(INDEX.wid2int[INDEX.title2wid[title]] for title in _relevant_titles)
    for (_idx, _doc_id) in enumerate(relevant_doc_ids):
        _document = list(INDEX.get_document_by_int_id(_doc_id))

        _title = INDEX.wid2title[INDEX.int2wid[_doc_id]]
        _target = 1

        _query = INDEX.tokenize(_question)
        _query = [INDEX.token2id[token] for token in _query]

        candidate_df.iloc[_idx] = [_id, _doc_id, json.dumps(_query), json.dumps(_document), _target]

    _results = INDEX.unigram_query(_question, 1000)
    count = 0
    for (_idx, _result) in enumerate(_results):
        if count == 998:
            break

        (_doc_id, _) = _result
        _title = INDEX.wid2title[INDEX.int2wid[_doc_id]]

        if _title in _relevant_titles:
            continue

        _document = list(INDEX.get_document_by_int_id(_doc_id))
        _target = int(_title in _relevant_titles)

        _query = INDEX.tokenize(_question)
        _query = [INDEX.token2id[token] for token in _query]

        candidate_df.iloc[2 + count] = [_id, _doc_id, json.dumps(_query), json.dumps(_document), _target]
        count += 1

    helpers.log(f'Processed question {no}')

    return candidate_df


if __name__ == '__main__':
    build()
