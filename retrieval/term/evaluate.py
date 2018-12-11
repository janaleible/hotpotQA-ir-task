import pickle
import sqlite3
from tqdm import tqdm
from main_constants import *
from typing import Dict, Tuple
import pytrec_eval
import json
import os
from services import helpers, sql
from services.evaluation import Evaluator, Run

_WID2TITLE: Dict[int, str]
_INT2WID: Dict[int, int]


def process(command: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    helpers.log('Loading int2wid and wid2title mappings.')
    global _WID2TITLE, _INT2WID
    with open(WID2TITLE, 'rb') as file:
        _WID2TITLE = pickle.load(file)
    with open(INT2WID, 'rb') as file:
        _INT2WID = pickle.load(file)

    model_type, model_name = command.split('@')
    dataset_id = helpers.training_set_id()
    if model_type == 'term':
        dir_path = os.path.join(TERM_RETRIEVALS_DIR, f'{model_name}.{dataset_id}')
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


def _create_trec_eval_reference(dir_path: str) -> None:
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


def _create_trec_run(dir_path: str) -> Run:
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
                question_ranking = {_WID2TITLE[_INT2WID[_id]]: float(score)
                                    for (_id, score) in question_ranking}
                run.add_ranking(question_id, question_ranking)
                pbar.update(1)
    run.write_to_file(run_path)

    return run
