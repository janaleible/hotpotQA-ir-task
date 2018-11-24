from tqdm import tqdm

from services import helpers
from services.index import Index
from main_constants import *
import numpy as np
import bcolz

INDEX: Index


def build():
    global INDEX
    INDEX = Index(env='default')
    embeddings = bcolz.carray(np.zeros((len(INDEX.token2id), 300)), rootdir=EMBEDDINGS, mode='w')
    count = 0
    with open(EMBEDDINGS_TXT, 'rb') as f:
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
