import collections as cl
from typing import Dict, Tuple
import pytrec_eval
import json

EVALUATION = Tuple[Dict[str, Dict[str, float]], Dict[str, float]]


class Run(dict):
    def add_ranking(self, _id: str, ranking: Dict[str, float]) -> None:
        self[_id] = ranking

    def update_ranking(self, question_id: str, document_title: str, score: float) -> None:
        if not self.get(question_id, {}):
            self[question_id] = {}

        assert self[question_id].get(document_title, -1) == -1, f'Ranking already ' \
            f'exists {question_id}.{document_title}={self[question_id][document_title]}'

        self[question_id][document_title] = score

    def update_rankings(self, rankings: Dict[str, Dict[str, float]]) -> None:
        for _id in rankings.keys():
            if self.get(_id, {}):
                self[_id].update(rankings[_id])
            else:
                self[_id] = rankings[_id]

    def write_to_file(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self, file)


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
