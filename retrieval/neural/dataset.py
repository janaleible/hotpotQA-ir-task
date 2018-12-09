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

    def __init__(self, path: str):
        start = datetime.now()
        self.data = pd.read_pickle(path, compression='gzip')[:2000]
        helpers.log(f'Initialized {path.split(".")[-3]} dataset in {datetime.now() - start}')

    def __getitem__(self, item: int) -> Tuple[List[int], List[int], int, str, int]:
        """Returns query, document, relevance"""
        row = self.data.iloc[item]
        question_id, document_id, question, document, target = row

        question = self._filter(json.loads(question))
        document = self._filter(json.loads(document))

        return question, document, target, question_id, document_id

    def _filter(self, _ids: List[int]):
        return list(map(lambda _id: _id if _id < const.VOCAB_SIZE else 0, _ids))

    @staticmethod
    def collate(batch: Tuple[Any]) -> Tuple[torch.tensor, torch.tensor, torch.tensor, List[str], List[int] ]:
        queries, documents, targets, question_ids, document_ids = list(zip(*batch))
        max_query_length = max(map(lambda query: len(query), queries))
        max_document_length = max(map(lambda doc: len(doc), documents))

        queries = list(queries)
        documents = list(documents)
        for i in range(len(queries)):
            queries[i] += [0] * (max_query_length - len(queries[i]))
            documents[i] += [0] * (max_document_length - len(documents[i]))

        queries = torch.tensor(queries)
        documents = torch.tensor(documents)
        targets = torch.tensor(targets, dtype=torch.float).unsqueeze(dim=1)
        return queries, documents, targets, question_ids, document_ids

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from tqdm import tqdm

    _s = './data/candidates/tfidf.train.dummy.gzip'
    ds = QueryDocumentsDataset(_s)
    dl = data.DataLoader(ds, 100, False, num_workers=4, collate_fn=QueryDocumentsDataset.collate)
    pbar = tqdm(total=100 * 1000)
    for x in dl:
        pbar.update(100)
