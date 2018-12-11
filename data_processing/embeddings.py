from tqdm import tqdm
import pickle
from main_constants import *
import numpy as np


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

    embeddings = np.zeros((VOCAB_SIZE + 1, dim))

    with open(embeddings_txt, 'rb') as f:

        pyndri2glove = {0: 0}
        glove2pyndri = {0: 0}
        glove_index = 1  # start at 1 to reserve 0 as unk

        for row_num, line in enumerate(tqdm(f.readlines())):

            line = line.decode().split()
            word = line[0]
            pyndri_index = token2id.get(word, -1)

            if not pyndri_index == -1:
                embeddings[glove_index] = np.array(line[1:]).astype(np.float)

                pyndri2glove[pyndri_index] = glove_index
                glove2pyndri[glove_index] = pyndri_index

                glove_index += 1

        # fill embeddings[0] with average of all embeddings for unknown token
        # (as of https://groups.google.com/d/msg/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
        # filter out all-zero rows first
        embeddings[0] = np.mean(embeddings[np.any(embeddings, axis=1)], axis=0)

        # generate random embeddings for the most common words in pyndri that do not appear in glove
        pyndri_index = 1
        while len(pyndri2glove) <= VOCAB_SIZE:

            if pyndri_index not in pyndri2glove.keys():
                embeddings[glove_index] = np.random.normal(loc=0.0, scale=0.5, size=(1, dim))

                pyndri2glove[pyndri_index] = glove_index
                glove2pyndri[glove_index] = pyndri_index

                glove_index += 1

            pyndri_index += 1

    with open(PYNDRI2GLOVE, 'wb') as file:
        pickle.dump(pyndri2glove, file)

    with open(GLOVE2PYNDRI, 'wb') as file:
        pickle.dump(glove2pyndri, file)

    np.savez_compressed(embeddings_file, array=embeddings)
