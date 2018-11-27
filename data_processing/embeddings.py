from tqdm import tqdm

from services.index import Index
from main_constants import *
import numpy as np
import bcolz

INDEX: Index


def build(emb: str):
    if emb == 'E6B.50':
        embeddings = EMBEDDINGS_50
        embeddings_txt = EMBEDDINGS_50_TXT
    elif emb == 'E6B.100':
        embeddings = EMBEDDINGS_100
        embeddings_txt = EMBEDDINGS_100_TXT
    elif emb == 'E6B.200':
        embeddings = EMBEDDINGS_200
        embeddings_txt = EMBEDDINGS_200_TXT
    elif emb == 'E6B.300':
        embeddings = EMBEDDINGS_300
        embeddings_txt = EMBEDDINGS_300_TXT
    else:
        raise ValueError(f'Unknown embedding specification {emb}')

    global INDEX
    INDEX = Index(env='default')
    dim = int(emb.split('.')[-1])
    embeddings = bcolz.carray(np.zeros((len(INDEX.token2id), dim)), rootdir=embeddings, mode='w')
    count = 0
    with open(embeddings_txt, 'rb') as f:
        for l in tqdm(f.readlines()):
            line = l.decode().split()
            word = INDEX.normalize(line[0])
            try:
                idx = INDEX.token2id[word]
                embedding = np.array(line[1:]).astype(np.float)
                embeddings[idx] = embedding
                count += 1
            except KeyError:
                pass

    print(count, len(INDEX.token2id))
    embeddings.flush()
