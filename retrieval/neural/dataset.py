import sqlite3
from typing import Tuple, Any, List
from datetime import datetime
from services import helpers
from torch.utils import data
import main_constants as const
import pandas as pd
import torch
import json

ITEM = Tuple[List[int], List[Tuple[int, ...]], List[Tuple[int, ...]], List[str], List[str], str]


# noinspection PyCallingNonCallable
class QueryDocumentsDataset(data.Dataset):
    data: pd.DataFrame
    cut_off: int

    _features = ['query_id', 'doc_id', 'query_tokens', 'doc_tokens', 'tfidf', 'entity_match_PER', 'entity_match_LOC',
        'entity_match_ORG', 'entity_match_MISC', 'ibm1', 'nibm1', 'bigram', 'nbigram', 'qwhat', 'qwhich', 'qwho', 'qin',
        'qare', 'qis', 'qwas', 'qwhen', 'qwhere', 'qhow', 'doclen', 'relevant']

    def __init__(self, database: str):
        start = datetime.now()

        connection = sqlite3.connect(database)
        cursor = connection.cursor()

        self.data = cursor.execute(f'SELECT {", ".join(self._features)} FROM features').fetchall()


        helpers.log(f'Initialized {database.split(".")[-3]} dataset in {datetime.now() - start}')

    def __getitem__(self, item: int) -> Tuple[List[int], List[int], List[float], int, str, int]:
        """Returns query, document, relevance"""
        row = [json.loads(entry) for entry in self.data[item]]


        question_id = row[0]
        document_id = row[1]
        question = row[2]
        document = row[3]
        features = row[4:24]
        target = row[24]

        min_tfidf = 0
        max_tfidf = 2544.1113472397183
        features[0] = self.normalize(features[0], min_tfidf, max_tfidf)

        min_ent_PER = 0
        max_ent_PER = 4
        features[1] = self.normalize(features[1], min_ent_PER, max_ent_PER)

        min_ent_LOC = 0
        max_ent_LOC = 4
        features[2] = self.normalize(features[2], min_ent_LOC, max_ent_LOC)

        min_ent_ORG = 0
        max_ent_ORG = 4
        features[3] = self.normalize(features[3], min_ent_ORG, max_ent_ORG)

        min_ent_MISC = 0
        max_ent_MISC = 5
        features[4] = self.normalize(features[4], min_ent_MISC, max_ent_MISC)

        min_ibm1 = 0
        max_ibm2 = 0.009390061985520014
        features[5] = self.normalize(features[5], min_ibm1, max_ibm2)

        min_nibm1 = 0
        max_nibm2 = 0.0023475154963800036
        features[6] = self.normalize(features[6], min_nibm1, max_nibm2)

        min_bigram = 0
        max_bigram = 81
        features[7] = self.normalize(features[7], min_bigram, max_bigram)

        min_nbigram = 0
        max_nbigram = 0.9655172413793104
        features[8] = self.normalize(features[8], min_nbigram, max_nbigram)

        # skip one-hot encoded question words

        min_doclen = 1
        max_doclen = 1488
        features[19] = self.normalize(features[19], min_doclen, max_doclen)

        question = self._filter(question)
        document = self._filter(document)

        return question, document, features, target, question_id, document_id

    def normalize(self, value, min, max):
        return (value - min) / (max - min)

    def _filter(self, _ids: List[int]):
        return list(map(lambda _id: _id if _id < const.VOCAB_SIZE else 0, _ids))

    @staticmethod
    def collate(batch: Tuple[Any]) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, List[str], List[int]]:

        queries, documents, features, targets, question_ids, document_ids = list(zip(*batch))

        max_query_length = max(map(lambda query: len(query), queries))
        max_document_length = max(map(lambda doc: len(doc), documents))

        queries = list(queries)
        documents = list(documents)
        for i in range(len(queries)):
            queries[i] += [0] * (max_query_length - len(queries[i]))
            documents[i] += [0] * (max_document_length - len(documents[i]))

        queries = torch.tensor(queries)
        documents = torch.tensor(documents)
        features = torch.tensor(features)
        targets = torch.tensor(targets, dtype=torch.float).unsqueeze(dim=1)
        return queries, documents, features, targets, question_ids, document_ids

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from tqdm import tqdm

    _s = '../../data/features/train.dummy.feature.db'
    ds = QueryDocumentsDataset(_s)
    dl = data.DataLoader(ds, 100, False, num_workers=4, collate_fn=QueryDocumentsDataset.collate)
    pbar = tqdm(total=100 * 1000)
    for x in dl:
        pbar.update(100)
