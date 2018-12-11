import pickle
from typing import List, Tuple, Dict

from main_constants import *
from services import parallel, helpers
from services.index import Index

INDEX: Index


def build():
    global INDEX
    INDEX = Index(env='default')
    batches = parallel.chunk(CHUNK_SIZE, INDEX.document_int_ids())
    helpers.log(f'Building maps for {INDEX.count()} documents.')
    int2wid = {}
    wid2int = {}
    for batch_maps in parallel.execute(_process_batch, batches):
        batch_int2wid, batch_wid2int = batch_maps
        int2wid.update(batch_int2wid)
        wid2int.update(batch_wid2int)

    with open(TOKEN2ID, 'wb')as file:
        pickle.dump(INDEX.token2id, file)
    with open(ID2TOKEN, 'wb')as file:
        pickle.dump(INDEX.id2token, file)
    with open(ID2DF, 'wb')as file:
        pickle.dump(INDEX.id2df, file)
    with open(ID2TF, 'wb')as file:
        pickle.dump(INDEX.id2tf, file)
    with open(ID2TOKEN, 'wb')as file:
        pickle.dump(INDEX.id2token, file)
    with open(INT2WID, 'wb') as file:
        pickle.dump(int2wid, file)
    with open(WID2INT, 'wb') as file:
        pickle.dump(wid2int, file)

    helpers.log(f'Finished building maps. Mapped {len(int2wid)}/{INDEX.index.document_count()}')


def _process_batch(batch: Tuple[int, List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    no, batch = batch
    wid2int = {}
    int2wid = {}
    for int_doc_id in batch:
        wid = INDEX.get_wid(int_doc_id)
        wid2int[wid] = int_doc_id
        int2wid[int_doc_id] = wid

    helpers.log(f'Finished batch. Mapped {len(wid2int)}.')
    return int2wid, wid2int
