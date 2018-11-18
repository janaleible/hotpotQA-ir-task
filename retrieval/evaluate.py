import argparse
import json
import os
from typing import Dict
import pytrec_eval
import constants


class Evaluator:

    _evaluator: pytrec_eval.RelevanceEvaluator

    def __init__(self, reference_path: str, measures: set = frozenset({'map', 'ndcg'})) -> None:

        with open(reference_path, 'r') as reference_file:
            reference = json.load(reference_file)
            self._evaluator = pytrec_eval.RelevanceEvaluator(reference, set(measures))

    def evaluate(self, result: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

        return self._evaluator.evaluate(result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate a ranking with trec_eval')
    parser.add_argument('reference_file', type=str,
                        help='path to the reference file (produce using the dataset script)')
    parser.add_argument('result_file', type=str, help='path to the result file')

    args, _ = parser.parse_known_args()

    evaluator = Evaluator(constants.TRECEVAL_REFERENCE_TRAIN)

    with open(constants.TRECEVAL_RESULTS_TRAIN, 'r') as results_file:
        results = json.load(results_file)

    evaluation = evaluator.evaluate(results)

    os.makedirs(constants.TRECEVAL_EVALUATION_DIR, exist_ok=True)
    with open(constants.TRECEVAL_EVALUATION_TRAIN, 'w') as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=True)
