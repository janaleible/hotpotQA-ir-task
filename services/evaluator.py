import json
import collections as cl
from typing import Dict

import pytrec_eval

from services.evaluation import EVALUATION
from services.run import Run


class Evaluator:
    _evaluator: pytrec_eval.RelevanceEvaluator
    reference: Dict[str, Dict[str, int]]

    def __init__(self, reference_path: str, measures: set = frozenset({'map', 'ndcg'})) -> None:
        with open(reference_path, 'r') as reference_file:
            self.reference = json.load(reference_file)
            self._evaluator = pytrec_eval.RelevanceEvaluator(self.reference, set(measures))

    def evaluate(self, run: Run, eval_path: str = None, eval_agg_path: str = None, save: bool = True) -> EVALUATION:
        trec_eval = self._evaluator.evaluate(run)
        trec_eval_agg = self.aggregate(trec_eval)

        if save:
            with open(eval_path, 'w') as file:
                json.dump(trec_eval, file, indent=True)
            with open(eval_agg_path, 'w') as file:
                json.dump(trec_eval_agg, file, indent=True)

        return trec_eval, trec_eval_agg

    def aggregate(self, trec_eval: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        eval_aggr = cl.defaultdict(float)
        for trec_eval_item in trec_eval.values():
            for measure, value, in trec_eval_item.items():
                eval_aggr[measure] += value / len(trec_eval)

        return eval_aggr
