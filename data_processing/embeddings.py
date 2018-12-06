from tqdm import tqdm
import pickle
from services import helpers
from main_constants import *
import numpy as np
import bcolz


def build(emb: str):
    if emb == 'E6B.50':
        embeddings_file = EMBEDDINGS_50
        embeddings_txt = EMBEDDINGS_50_TXT
    elif emb == 'E6B.100':
        embeddings_file = EMBEDDINGS_100
        embeddings_txt = EMBEDDINGS_100_TXT
    elif emb == 'E6B.200':
        embeddings_file = EMBEDDINGS_200
        embeddings_txt = EMBEDDINGS_200_TXT
    elif emb == 'E6B.300':
        embeddings_file = EMBEDDINGS_300
        embeddings_txt = EMBEDDINGS_300_TXT
    else:
        raise ValueError(f'Unknown embedding specification {emb}')

    with open(TOKEN2ID, 'rb') as file:
        token2id = pickle.load(file)

    dim = int(emb.split('.')[-1])
    # embeddings = bcolz.carray(np.random.normal(loc=0.0, scale=0.5, size=(len(token2id), dim)), rootdir=embeddings, mode='w')
    # embeddings = bcolz.carray(np.zeros((len(token2id), dim)), rootdir=embeddings, mode='w')
    embeddings = np.random.normal(loc=0.0, scale=0.5, size=(len(token2id), dim))

    with open(embeddings_txt, 'rb') as f:
        embeddings[0] = np.zeros((1, dim))
        count = 0
        for line in tqdm(f.readlines()):
            line = line.decode().split()
            word = line[0]
            if token2id.get(word, -1) != -1:
                idx = token2id[word]
                embedding = np.array(line[1:]).astype(np.float)
                embeddings[idx] = embedding
                count += 1
    helpers.log(f'{count}/{len(token2id)}')
    np.savez_compressed(embeddings_file, array=embeddings)
