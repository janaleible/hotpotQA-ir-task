import json
import os

import pyndri
from tqdm import tqdm

from dataset.dataset import Dataset
import main_constants as c
from retrieval.evaluate import Run, Evaluator
from retrieval.index import Index
from retrieval.tokenizer import Tokenizer
from services import parallel


def parallel_tfidf(dataset: Dataset):

    full_run = Run()

    batches = list(parallel.chunk(c.CHUNK_SIZE, dataset.questions))
    for partial_run in parallel.execute(tfidf, batches):
        full_run.update(partial_run)

    return full_run


def tfidf(dataset: Dataset):

    index = Index()
    tfidf_query_env = pyndri.TFIDFQueryEnvironment(index.index)
    tokenizer = Tokenizer()
    run = Run()

    try:
        for question in tqdm(dataset, unit='questions'):

            tokenized = ' '.join(tokenizer.tokenize(question.question))
            result = tfidf_query_env.query(tokenized)

            ranking = {index.wid2title[index.internal2external(_id)]: float(score)
                       for (_id, _), score in zip(result[:10], reversed(range(1, 11)))}

            run.add_question(question.id, ranking)
    except KeyboardInterrupt:
        print(run)

    return run


def evaluate(dataset: Dataset, results_path: str, reference_path: str, evaluation_path: str, tfidf_fn=tfidf):

    run = tfidf_fn(dataset)

    with open(results_path, 'w') as run_file:
        json.dump(run, run_file, indent=True)

    evaluator = Evaluator(reference_path)
    evaluation = evaluator.evaluate(run)

    with open(evaluation_path, 'w') as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=True)

    maps = []
    hits_at_two = []
    hits_at_ten = []

    for qid, question in evaluation.items():
        maps.append(question['map'])

    for qid, ranking in run.items():

        gold = dataset.find_by_id(qid).gold_articles
        list_ranking = sorted(ranking, key=ranking.get, reverse=True)

        hits_at_two.append(0.5 * len(set(gold).intersection(list_ranking[:1])))
        hits_at_ten.append(0.5 * len(set(gold).intersection(list_ranking)))

    print(f'Mean mAP: {round(sum(maps) / len(maps), 4)}')
    print(f'hits@2: {round(sum(hits_at_two) / len(hits_at_two), 4)}')
    print(f'hits@10: {round(sum(hits_at_ten) / len(hits_at_ten), 4)}')

    return run, evaluation


def main():

    training_set = Dataset.from_file(c.TRAINING_SET)
    evaluate(training_set, c.TRECEVAL_RESULTS_TRAIN, c.TRECEVAL_REFERENCE_TRAIN, c.TRECEVAL_EVALUATION_TRAIN, parallel_tfidf)
