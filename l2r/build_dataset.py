import json
import os
import pickle
import random
from tqdm import tqdm

from dataset.dataset import Dataset, Question
import main_constants as c
from services.index import Index


def get_document(title: str, index: Index):

    return index.get_document_by_title(title)


def sample_from_tfidf(question: Question, index: Index):
    # TODO: load once
    with open(c.TRECEVAL_RESULTS_TRAIN, 'r') as evaluation_file:
        tfidf_results = json.load(evaluation_file)
    distractor_documents = []
    for title in sorted(tfidf_results[question.id], key=tfidf_results[question.id].get, reverse=True):
        if title not in question.gold_articles:
            distractor_documents.append(get_document(title, index))
            if len(distractor_documents) == 2: break
    return distractor_documents


def sample_random(question: Question, index: Index):
    titles = list(index.title2wid.keys())
    distractor_titles = [titles[random.randint(0, len(titles) - 1)], titles[random.randint(0, len(titles) - 1)]]

    return [get_document(title, index) for title in distractor_titles]


def build_l2r_dataset():

    index = Index()
    dataset = Dataset.from_file(c.TRAINING_SET)

    examples = list()
    sample_function = sample_random

    for question in tqdm(dataset):

        gold_documents = [get_document(title, index) for title in question.gold_articles]

        distractor_documents = sample_function(question, index)

        examples.extend(zip(
            [question.question] * (len(gold_documents) + len(distractor_documents)),
            gold_documents + distractor_documents,
            [1] * len(gold_documents) + [0] * len(distractor_documents)
        ))

    random.shuffle(examples)

    testset_cutoff = int(0.1 * len(examples))

    os.makedirs(c.L2R_DATA_DIR, exist_ok=True)
    with open(c.L2R_TRAINING_SET, 'wb') as file:
        pickle.dump(examples[testset_cutoff:], file)

    with open(c.L2R_TEST_SET, 'wb') as file:
        pickle.dump(examples[:testset_cutoff], file)
