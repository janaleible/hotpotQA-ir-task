import argparse
import json
import os
from typing import Dict
import pytrec_eval

import main_constants as c
from dataset.dataset import Dataset


class Run(dict):

    def add_question(self, id, ranking):
        self[id] = ranking

    def write_to_file(self, path):

        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self, file)


class Evaluator:

    _evaluator: pytrec_eval.RelevanceEvaluator

    def __init__(self, reference_path: str, measures: set = frozenset({'map', 'ndcg'})) -> None:

        Evaluator.create_reference_files(exists_ok=True)

        with open(reference_path, 'r') as reference_file:
            reference = json.load(reference_file)
            self._evaluator = pytrec_eval.RelevanceEvaluator(reference, set(measures))

    def evaluate(self, run: Run) -> Dict[str, Dict[str, float]]:

        return self._evaluator.evaluate(run)

    @staticmethod
    def create_reference_files(exists_ok=False):

        os.makedirs(c.TRECEVAL_REFERENCE_DIR, exist_ok=True)

        for data_file, reference_file in zip([c.TRAINING_SET, c.DEV_FULLWIKI_SET],
                                             [c.TRECEVAL_REFERENCE_TRAIN, c.TRECEVAL_REFERENCE_DEV]):
            if not os.path.isfile(reference_file) or not exists_ok:
                data_set = Dataset.from_file(data_file)
                with open(reference_file, 'w') as file:
                    json.dump(data_set.get_treceval_reference(), file, indent=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate a ranking with trec_eval')
    parser.add_argument('reference_file', type=str,
                        help='path to the reference file (produce using the dataset script)')
    parser.add_argument('result_file', type=str, help='path to the result file')

    args, _ = parser.parse_known_args()

    evaluator = Evaluator(c.TRECEVAL_REFERENCE_TRAIN)

    with open(c.TRECEVAL_RESULTS_TRAIN, 'r') as results_file:
        results = json.load(results_file)

    evaluation = evaluator.evaluate(results)

    os.makedirs(c.TRECEVAL_EVALUATION_DIR, exist_ok=True)
    with open(c.TRECEVAL_EVALUATION_TRAIN, 'w') as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=True)
