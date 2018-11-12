import pickle

import constants
from data_processing.inverted_index import Index
from dataset.Dataset import Dataset
from retrieval.filtering import unigram_bigram_filter

training_set = Dataset(constants.TRAINING_SET)

with open(constants.INDEX_FILE, 'rb') as index_file:
    index: Index = pickle.load(index_file)

# indexed with the number of gold articles in the filtered set
found_articles = {2: 0, 1: 0, 0: 0}

for question in training_set:

    gold_article_ids = set([index.title2id[title] for title in question.gold_articles])
    filtered_articles = unigram_bigram_filter(question.question, index)

    number_of_articles_found = len(gold_article_ids.intersection(filtered_articles))

    found_articles[number_of_articles_found] += 1

fully_accurate = found_articles[2] / sum(found_articles.values())
somewhat_accurate = (found_articles[1] + found_articles[2]) / sum(found_articles.values())

print(f'Fully accurate: {round(fully_accurate, 4)}')
print(f'Somewhat accurate: {round(somewhat_accurate, 4)}')
