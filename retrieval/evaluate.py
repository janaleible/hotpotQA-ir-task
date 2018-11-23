import pickle
import sqlite3
from tqdm import tqdm
from main_constants import *
from typing import Dict, Tuple
import pytrec_eval
import collections as cl
import json
import os

from services import helpers, sql
from services.index import Index

INDEX: Index


class Run(dict):
    def add_question(self, _id, ranking):
        self[_id] = ranking

    def write_to_file(self, path):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self, file)


class Evaluator:
    _evaluator: pytrec_eval.RelevanceEvaluator

    def __init__(self, reference_path: str, measures: set = frozenset({'map', 'ndcg'})) -> None:
        with open(reference_path, 'r') as reference_file:
            reference = json.load(reference_file)
            self._evaluator = pytrec_eval.RelevanceEvaluator(reference, set(measures))

    def evaluate(self, run: Run, eval_path: str, eval_agg_path: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        trec_eval = self._evaluator.evaluate(run)
        trec_eval_agg = self.aggregate(trec_eval)

        with open(eval_path, 'w') as file:
            json.dump(trec_eval, file, indent=True)
        with open(eval_agg_path, 'w') as file:
            json.dump(trec_eval_agg, file, indent=True)

        return trec_eval, trec_eval_agg

    def aggregate(self, trec_eval: Dict[str, Dict[str, float]])-> Dict[str, float]:
        eval_aggr = cl.defaultdict(float)
        for trec_eval_item in trec_eval.values():
            for measure, value, in trec_eval_item.items():
                eval_aggr[measure] += value / len(trec_eval)

        return eval_aggr


def process(command: str):
    global INDEX
    INDEX = Index()
    model_type, model_name = command.split('@')
    dataset_id = helpers.training_set_id()
    if model_type == 'filter':
        dir_path = os.path.join(FILTERS_DIR, f'{model_name}.{dataset_id}')
    elif model_type == 'rank':
        dir_path = os.path.join(RANKERS_DIR, f'{model_name}.{dataset_id}')
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    reference_path = os.path.join(dir_path, 'reference.json')
    reference_exists = os.path.isfile(reference_path)
    if not reference_exists:
        _create_trec_eval_reference(dir_path)

    run_path = os.path.join(dir_path, 'retrievals.json')
    run_exists = os.path.isfile(run_path)
    if not run_exists:
        run = _create_trec_run(dir_path)
    else:
        with open(run_path, 'r') as file:
            run = json.load(file)

    trec_eval_path = os.path.join(dir_path, 'trec_eval.json')
    trec_eval_agg_path = os.path.join(dir_path, 'trec_eval_agg.json')
    evaluator = Evaluator(reference_path, measures=pytrec_eval.supported_measures)
    trec_eval, trec_eval_agg = evaluator.evaluate(run, trec_eval_path, trec_eval_agg_path)

    return trec_eval, trec_eval_agg


def _create_trec_eval_reference(dir_path: str):
    reference_path = os.path.join(dir_path, 'reference.json')
    reference = {}
    with sqlite3.connect(os.path.join(dir_path, 'retrievals.sqlite')) as db:
        cursor = db.cursor()
        cursor.execute(sql.get_count())
        (total,) = cursor.fetchone()
        with tqdm(total=total) as pbar:
            cursor.execute(sql.get_reference())
            for (question_id, question_ref) in cursor:
                question_ref = pickle.loads(question_ref)
                reference[question_id] = {title: 1 for title in question_ref}
                pbar.update(1)
    with open(reference_path, 'w') as file:
        json.dump(reference, file, indent=True)

    return


def _create_trec_run(dir_path: str):
    run_path = os.path.join(dir_path, 'run.json')
    run = Run()
    with sqlite3.connect(os.path.join(dir_path, 'retrievals.sqlite')) as db:
        cursor = db.cursor()
        cursor.execute(sql.get_count())
        (total,) = cursor.fetchone()
        with tqdm(total=total) as pbar:
            cursor.execute(sql.get_retrievals())
            for (question_id, question_ranking) in cursor:
                question_ranking = pickle.loads(question_ranking)
                question_ranking = {INDEX.wid2title[INDEX.internal2external(_id)]: float(score)
                                    for (_id, score) in question_ranking}
                run.add_question(question_id, question_ranking)
                pbar.update(1)
    run.write_to_file(run_path)

    return run
