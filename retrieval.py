import pickle

from data_processing import inverted_index
from retrieval import filtering

if __name__ == '__main__':
    with open('./data/preprocessed/AA.dict.tar', 'rb') as file:
        aa = pickle.load(file)
    index = inverted_index.load()

    filtering.unigram_bigram("asd", index, n=3)
