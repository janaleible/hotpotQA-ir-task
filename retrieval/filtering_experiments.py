import numpy
from tqdm import tqdm

import constants
from dataset.Dataset import Dataset
from retrieval.filtering import unigram_bigram_filter
from retrieval.index import Index


def accuracy(true_positives: int, all: int) -> float:
    return true_positives / all


training_set = Dataset(constants.TRAINING_SET)
index: Index = Index()

# indices for results array
COMPARISON = 0; BRIDGE = 1
HARD = 0; MEDIUM = 1; EASY = 2

# shape: (question type [comparison, bridge], level [hard, medium, easy], number of found gold articles [0, 1, 2])
found_articles = numpy.zeros((2, 3, 3))

for question in tqdm(training_set, unit='questions'):

    gold_article_ids = set()
    [[gold_article_ids.add(index.external2internal(id)) for id in index.title2wid[title]] for title in question.gold_articles]

    filtered_articles = unigram_bigram_filter(question.question, index)

    number_of_articles_found = len(gold_article_ids.intersection(filtered_articles))

    question_type = COMPARISON if question.type == 'comparison' else BRIDGE
    question_level = HARD if question.level == 'hard' else MEDIUM if question.level == 'medium' else EASY

    found_articles[question_type][question_level][number_of_articles_found] += 1

fully_accurate = accuracy(numpy.sum(found_articles[:, :, 2]), numpy.sum(found_articles[:, :, :]))
somewhat_accurate = accuracy(numpy.sum(found_articles[:, :, 1:]), numpy.sum(found_articles[:, :, :]))

accurate_hard_questions = accuracy(numpy.sum(found_articles[:, HARD, 2]), numpy.sum(found_articles[:, HARD, :]))
accurate_medium_questions = accuracy(numpy.sum(found_articles[:, MEDIUM, 2]), numpy.sum(found_articles[:, MEDIUM, :]))
accurate_easy_questions = accuracy(numpy.sum(found_articles[:, EASY, 2]), numpy.sum(found_articles[:, EASY, :]))

accurate_comparison_questions = accuracy(numpy.sum(found_articles[COMPARISON, :, 2]), numpy.sum(found_articles[COMPARISON, :, :]))
accurate_bridge_questions = accuracy(numpy.sum(found_articles[BRIDGE, :, 2]), numpy.sum(found_articles[BRIDGE, :, :]))

print(f'Fully accurate: {round(fully_accurate, 4)}')
print(f'somehwat accurate: {round(somewhat_accurate, 4)}')
print('-'*10)
print(f'easy question accuracy: {round(accurate_easy_questions, 4)}')
print(f'medium question accuracy: {round(accurate_medium_questions, 4)}')
print(f'hard question accuracy: {round(accurate_hard_questions, 4)}')
print('-'*10)
print(f'comparison question accuracy: {round(accurate_comparison_questions, 4)}')
print(f'bridge question accuracy: {round(accurate_bridge_questions, 4)}')

