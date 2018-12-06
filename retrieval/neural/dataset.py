import json
import pickle
import random
import sqlite3
from datetime import datetime
from typing import Tuple, Dict, Any, List

import torch

from services import sql, helpers
from torch.utils import data
from services.index import Index


class TestDataset(data.Dataset):
    index: Index
    questions: List[Dict[str, Any]]

    def __init__(self, index: Index, question_set: str):
        start = datetime.now()

        with open(question_set, 'r') as file:
            self.questions = json.load(file)
        self.index = index
        helpers.log(f'Initialized DevDataset in {datetime.now() - start}')

    def __getitem__(self, item) -> Tuple[List[int], List[Tuple[int, ...]], str]:
        question = self.questions[item]
        question_id = question['_id']
        query = self.index.tokenize(question['question'])
        query = list(self.index.token2id[token] for token in query)

        retrieval = self.index.unigram_query(question['question'], request=1000)
        retrieved_docs = [self.index.get_document_by_int_id(int_id) for (int_id,) in retrieval]

        return query, retrieved_docs, question_id

    def __len__(self):
        return len(self.questions)


class DevDataset(data.Dataset):
    index: Index
    questions: List[Dict[str, Any]]

    def __init__(self, index: Index, question_set: str):
        start = datetime.now()

        with open(question_set, 'r') as file:
            self.questions = json.load(file)
        self.index = index
        helpers.log(f'Initialized DevDataset in {datetime.now() - start}')

    def __getitem__(self, item) -> Tuple[
        List[int], List[Tuple[int, ...]], List[Tuple[int, ...]], List[str], List[str], str]:
        question = self.questions[item]
        question_id = question['_id']
        query = self.index.tokenize(question['question'])
        query = list(self.index.token2id[token] for token in query)

        retrieval = self.index.unigram_query(question['question'], request=1000)
        retrieved_docs = []
        retrieved_doc_titles = []
        for int_id, _ in retrieval:
            retrieved_docs.append(self.index.get_document_by_int_id(int_id))
            retrieved_doc_titles.append(self.index.wid2title[self.index.int2wid[int_id]])
        relevant_docs = []
        relevant_doc_titles = []
        for (title, _) in question['supporting_facts']:
            relevant_docs.append(self.index.get_document_by_title(title))
            relevant_doc_titles.append(title)

        return query, retrieved_docs, relevant_docs, retrieved_doc_titles, relevant_doc_titles, question_id

    def __len__(self):
        return len(self.questions)


class TrainDataset(data.Dataset):
    db: str
    index: Index
    questions: List[Dict[str, Any]]
    deterministic: bool

    def __init__(self, index: Index, candidates_db: str, question_set: str, deterministic: bool):
        start = datetime.now()
        self.db = candidates_db
        with open(question_set, 'r') as file:
            self.questions = json.load(file)
        self.index = index
        self.deterministic = deterministic
        helpers.log(f'Initialized TrainDataset in {datetime.now() - start}')

    def __getitem__(self, item: int) -> Tuple[List[int], List[int], int, str]:
        """Returns query, document, relevance"""
        question = self.questions[item]
        question_id = question['_id']
        query = self.index.tokenize(question['question'])
        query = tuple(self.index.token2id[token] for token in query)

        # flip a coin to return a relevant or irrelevant document
        coin = random.randint(0, 1)
        if coin == 1:
            if self.deterministic:
                # get the first relevant document
                document_title = question['supporting_facts'][0][0]
            else:
                # roll a dice to chose a relevant document to return
                document_title = random.choice(question['supporting_facts'])[0]
            document = self.index.get_document_by_title(document_title, )

            target = 1
        else:
            with sqlite3.connect(self.db) as db:
                (q_id, target_titles, doc_int_ids) = db.cursor() \
                    .execute(sql.get_question_by_qid(), (question_id,)) \
                    .fetchone()
                target_titles = pickle.loads(target_titles)
                doc_int_ids = tuple(_id for (_id, _) in pickle.loads(doc_int_ids)[:1000])
                target_ext_ids = tuple(self.index.title2wid[title] for title in target_titles)
                target_int_ids = tuple(self.index.external2internal(_id) for _id in target_ext_ids)
                if self.deterministic:
                    i = 0
                    doc_int_id = doc_int_ids[i]
                    while doc_int_id in target_int_ids:
                        i += 1
                        doc_int_id = doc_int_ids[i]
                else:
                    # roll a dice to choose an irrelevant document
                    doc_int_id = random.choice(doc_int_ids)
                    while doc_int_id in target_int_ids:
                        doc_int_id = random.choice(doc_int_ids)
                document = self.index.get_document_by_int_id(doc_int_id)

                target = 0

        return list(query), list(document), target, question['_id']

    @staticmethod
    def collate(batch: Tuple[Any]):
        queries, documents, targets, question_ids = list(zip(*batch))
        max_query_length = max(map(lambda query: len(query), queries))
        max_document_length = max(map(lambda doc: len(doc), documents))

        queries = list(queries)
        documents = list(documents)
        for i in range(len(queries)):
            queries[i] += [0] * (max_query_length - len(queries[i]))
            documents[i] += [0] * (max_document_length - len(documents[i]))

        return torch.tensor(queries), \
               torch.tensor(documents), \
               torch.tensor(targets, dtype=torch.float).unsqueeze(dim=1), question_ids

    def __len__(self):
        return len(self.questions)


if __name__ == '__main__':
    _db = './data/term_retrievals/uni_tfidf.dummy/retrievals.sqlite'
    _s = './data/hotpot/train_dummy.json'
    ds = TrainDataset(_db, _s)
    dl = data.DataLoader(ds, 5, True, num_workers=0, collate_fn=TrainDataset.collate)
    for b in dl:
        print(b)
        break
