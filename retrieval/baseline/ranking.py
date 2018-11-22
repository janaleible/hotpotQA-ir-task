import collections as cl
import json
import logging
import os
from datetime import datetime

import nltk
import pyndri
import pytrec_eval

from dataset.dataset import Dataset
import main_constants as c
from retrieval.evaluate import Run, Evaluator
from retrieval.index import Index
from services import parallel

INDEX: Index


def parallel_tfidf(dataset: Dataset):
    full_run = Run()

    batches = list(enumerate(parallel.chunk(c.CHUNK_SIZE, dataset.questions)))
    for partial_run in parallel.execute(tfidf, batches):
        full_run.update(partial_run)

    return full_run


def tfidf(dataset: Dataset):
    if isinstance(dataset, tuple):
        batch_no, dataset = dataset
    else:
        batch_no = 0

    tfidf_query_env = pyndri.TFIDFQueryEnvironment(INDEX.index, k1=1.2, b=0.75)
    lm_env = pyndri.QueryEnvironment(INDEX.index, rules=('method:dirichlet,mu:5000',))
    prf_env = pyndri.PRFQueryEnvironment(lm_env, fb_docs=10, fb_terms=10)
    run = Run()

    try:
        for question in dataset:
            if question is None:
                continue

            # UNIGRAM
            # query = ' '.join(INDEX.tokenize(question.question))
            # results = tfidf_query_env.query(query, results_requested=5000)

            # BIGRAM
            # results = cl.defaultdict(float)
            # for bigram in nltk.bigrams(INDEX.tokenize(question.question)):
            #     bigram = ' '.join(bigram)
            #     result = tfidf_query_env.query(f'#1({bigram})', results_requested=5000)
            #     for (_id, score) in result:
            #         results[_id] += score
            # results = sorted(list(zip(results.keys(), results.values())), key=lambda x: x[1], reverse=True)

            # LANGUAGE MODEL + PSEUDO-RELEVANCE
            query = ' '.join(INDEX.tokenize(question.question))
            results = prf_env.query(query, results_requested=5000)

            # get at most 5000 results
            results = results[:5000]

            ranking = {INDEX.wid2title[INDEX.internal2external(_id)]: float(score)
                       for (_id, _), score in zip(results, reversed(range(1, len(results) + 1)))}

            run.add_question(question.id, ranking)
    except KeyboardInterrupt:
        print(run)

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\t[Finished processing batch {batch_no}.]')
    return run


def evaluate(dataset: Dataset, results_path: str, reference_path: str, evaluation_path: str, tfidf_fn=tfidf):
    evaluator = Evaluator(reference_path, measures=pytrec_eval.supported_measures)
    os.makedirs(c.TRECEVAL_RESULTS_DIR)
    os.makedirs(c.TRECEVAL_EVALUATION_DIR)

    run = tfidf_fn(dataset)

    with open(results_path, 'w') as run_file:
        json.dump(run, run_file, indent=True)

    evaluation = evaluator.evaluate(run)

    with open(evaluation_path, 'w') as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=True)

    maps = []
    ndcgs = []

    maps1000 = []
    recalls1000 = []
    ndcgs1000 = []

    maps10 = []
    recalls10 = []
    ndcgs10 = []

    for qid, question in evaluation.items():

        maps.append(question['map'])
        ndcgs.append(question['ndcg'])

        maps1000.append(question['map_cut_1000'])
        recalls1000.append(question['recall_1000'])
        ndcgs1000.append(question['ndcg_cut_1000'])

        maps10.append(question['map_cut_10'])
        recalls10.append(question['recall_10'])
        ndcgs10.append(question['ndcg_cut_10'])

    recalls = []
    for qid, ranking in run.items():
        gold = dataset.find_by_id(qid).gold_articles
        list_ranking = sorted(ranking, key=ranking.get, reverse=True)
        recalls.append(len(set(gold).intersection(list_ranking)) / 2)

    print(f'Average map: {round(sum(maps) / len(maps), 4)}')
    print(f'Average ndcg: {round(sum(ndcgs) / len(ndcgs), 4)}')

    print(f'Average recall@5000: {round(sum(recalls) / len(recalls), 4)}')

    print(f'Average map@1000: {round(sum(maps1000) / len(maps1000), 4)}')
    print(f'Average recall@1000: {round(sum(recalls1000) / len(recalls1000), 4)}')
    print(f'Average ndcg@1000: {round(sum(ndcgs1000) / len(ndcgs1000), 4)}')

    print(f'Average map@10: {round(sum(maps10) / len(maps10), 4)}')
    print(f'Average recall@10: {round(sum(recalls10) / len(recalls10), 4)}')
    print(f'Average ndcg@10: {round(sum(ndcgs10) / len(ndcgs10), 4)}')

    return run, evaluation


def main():
    global INDEX
    INDEX = Index()
    training_set = Dataset.from_file(c.TRAINING_SET)
    evaluate(training_set, c.TRECEVAL_RESULTS_TRAIN, c.TRECEVAL_REFERENCE_TRAIN, c.TRECEVAL_EVALUATION_TRAIN,
             parallel_tfidf)
