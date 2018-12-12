import json
import math
import os

import sqlite3
from typing import List, Tuple, Callable, Set, Dict, Any
import main_constants as ct
from datetime import datetime
import collections as cl

from services import parallel, helpers, sql
from services.index import Index

INDEX: Index
COLUMNS = ct.CANDIDATE_COLUMNS
QUESTION_COUNTS: Dict[str, int]


def build():
    global INDEX, COLUMNS, QUESTION_COUNTS
    INDEX = Index('tfidf')
    helpers.log('Loaded index.')

    os.makedirs(ct.CANDIDATES_DIR, exist_ok=True)
    with open(ct.TRAIN_HOTPOT_SET, 'r') as file:
        question_set = json.load(file)
        train_question_set = question_set[:ct.TRAIN_DEV_SPLIT]
        dev_question_set = question_set[ct.TRAIN_DEV_SPLIT:]

    iterator: List[Tuple[str, str, Callable]] = [
        (train_question_set, 'train', ct.TRAIN_CANDIDATES_DB, ct.TRAIN_CANDIDATES_CHUNK),
        (dev_question_set, 'dev', ct.DEV_CANDIDATES_DB, ct.DEV_CANDIDATES_CHUNK)
    ]

    for (_set, split, candidate_db_path, chunk) in iterator:
        start = datetime.now()

        db = sqlite3.connect(candidate_db_path)
        cursor = db.cursor()
        cursor.execute(sql.create_candidate_table)
        db.commit()
        helpers.log('Created candidates table.')

        QUESTION_COUNTS = cursor.execute(sql.count_question_rows).fetchall()
        QUESTION_COUNTS = {json.loads(_id): _count for (_id, _count) in QUESTION_COUNTS}
        helpers.log(f'Retrieved question counts for {len(QUESTION_COUNTS)} questions.')

        cursor.close()
        db.close()

        helpers.log(f'Creating {split} candidate set with {len(_set)} question.')
        total_count = 0
        _set_generator = parallel.chunk(chunk, zip([split] * len(_set), _set))
        for batch_count in parallel.execute(_build_candidates, _set_generator):
            total_count += batch_count
        helpers.log(f'Created {split} candidate set with {total_count} questions in {datetime.now() - start}')


def _build_candidates(numbered_batch: Tuple[int, List[Dict[str, Any]]]) -> int:
    start = datetime.now()
    (no, batch), db, cursor = numbered_batch, None, None
    processed_count = 0
    skipped_count = 0

    for split, question in batch:
        if split == 'train':
            no_candidates = ct.TRAIN_NO_CANDIDATES
            candidate_db_path = ct.TRAIN_CANDIDATES_DB
        elif split == 'dev':
            no_candidates = ct.DEV_NO_CANDIDATES
            candidate_db_path = ct.DEV_CANDIDATES_DB
        else:
            raise ValueError(f'Unknown set {split}.')

        _id = question['_id']
        _type = question['type']
        _level = question['level']
        _str = question['question']
        relevant_titles = list(map(lambda item: item[0], question['supporting_facts']))

        if QUESTION_COUNTS.get(_id, 0) == no_candidates:
            skipped_count += 1
            continue

        # store relevant documents row
        rows: List[List[str]] = []
        relevant_doc_iids = set(INDEX.wid2int[INDEX.title2wid[title]] for title in relevant_titles)
        for (candidate_idx, doc_iid) in enumerate(relevant_doc_iids):
            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
            doc_wid, doc_title = _extract_doc_identifiers(row, INDEX, doc_iid)
            doc_text = _extract_text(row, _str, doc_wid)
            doc_tokens, question_tokens = _extract_tokens(row, INDEX, _str, doc_iid)
            tfidf_score = _extract_tfidf_score(row, INDEX, doc_tokens, question_tokens)
            relevance = _extract_relevance(row, doc_iid, relevant_doc_iids)

            rows.append(row)

        # store irrelevant documents row in order scored by tf-idf until reached candidate_set length
        result_idx = 0
        candidate_idx = ct.RELEVANT_DOCUMENTS
        results = INDEX.unigram_query(_str, no_candidates)
        while candidate_idx < no_candidates:
            (doc_iid, tfidf_score) = results[result_idx]

            row: List[str] = [json.dumps(_id), json.dumps(_type), json.dumps(_level)]
            relevance = _extract_relevance(row, doc_iid, relevant_doc_iids, False)
            if relevance == 1:
                result_idx += 1
                continue

            doc_wid, doc_title = _extract_doc_identifiers(row, INDEX, doc_iid)
            doc_text = _extract_text(row, _str, doc_wid)
            doc_tokens, question_tokens = _extract_tokens(row, INDEX, _str, doc_iid)

            row.append(json.dumps(tfidf_score))
            row.append(json.dumps(relevance))

            rows.append(row)
            candidate_idx += 1
            result_idx += 1

        if db is None:
            db = sqlite3.connect(candidate_db_path)
            cursor = db.cursor()
        cursor.executemany(sql.insert_candidate, rows)
        db.commit()
        processed_count += 1

    if db is not None:
        cursor.close()
        db.close()

    end = datetime.now()
    helpers.log(f'Processed batch {no} in {end - start}. Processed {processed_count}. Skipped {skipped_count}')

    return len(batch)


def _extract_doc_identifiers(row: List[str], index: Index, doc_id: int) -> Tuple[int, str]:
    doc_wid = index.int2wid[doc_id]
    doc_title = index.wid2title[doc_wid]

    row.extend([json.dumps(doc_id), json.dumps(doc_wid), json.dumps(doc_title)])

    return doc_wid, doc_title


def _extract_text(row: List[str], question_text: str, doc_wid: int) -> Tuple[str]:
    db = sqlite3.connect(ct.DOCUMENT_DB)
    cursor = db.cursor()
    (doc_text,) = cursor.execute("SELECT text from documents WHERE id = ?", (doc_wid,)).fetchone()

    row.extend([json.dumps(question_text), json.dumps(doc_text)])

    return doc_text


def _extract_tokens(row: List[str], index: Index, question_text: str, doc_iid: int) -> Tuple[List[int], List[int]]:
    doc_tokens = list(index.get_document_by_int_id(doc_iid))
    query_tokens = [index.token2id.get(token, 0) for token in index.tokenize(question_text)]

    row.extend([json.dumps(query_tokens), json.dumps(doc_tokens)])

    return doc_tokens, query_tokens


def _extract_tfidf_score(row: List[str], index: Index, doc_tokens: List[int], question_tokens: List[int]) -> float:
    """Implementation according to http://www.lemurproject.org/lemur/tfidf.pdf"""
    doc_len = len(doc_tokens)
    tfidf = 0.0
    question_token_counts = cl.Counter(question_tokens)
    doc_token_counts = cl.Counter(doc_tokens)
    for token in question_token_counts:
        if token not in doc_token_counts:
            continue
        q_token_count = question_token_counts[token]
        d_token_count = doc_token_counts[token]
        tfidf += __question_tf_fn(q_token_count) * \
                 __doc_tf_fn(d_token_count, index, doc_len) * \
                 __idf_fn(token, index) ** 2

    row.append(json.dumps(tfidf))

    return tfidf


def _extract_relevance(row: List[str], doc_iid: int, relevant_doc_iids: Set[int], store: bool = True) -> int:
    relevance = int(doc_iid in relevant_doc_iids)

    if store:
        row.append(json.dumps(relevance))

    return relevance


def __question_tf_fn(token_count: int) -> float:
    return 1000 * token_count / (token_count + 1000)


def __doc_tf_fn(token_count: int, index: Index, doc_len: int) -> float:
    return 1.2 * token_count / (token_count + 1.2 * (1 - 0.75 + 0.75 * doc_len / index.avg_doc_len))


def __idf_fn(token: int, index: Index):
    if index.id2df.get(token, -1) == -1:
        return 0
    return math.log(index.index.document_count() / index.id2df[token])
