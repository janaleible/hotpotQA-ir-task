import json
import sqlite3

import pytrec_eval
from tqdm import tqdm

import main_constants as ct
from services.evaluator import Evaluator
from services.run import Run


def run_eval(_set: str):
    if _set == 'train':
        candidate_db = ct.TRAIN_CANDIDATES_DB
        ref = ct.TRAIN_TREC_REFERENCE
    elif _set == 'dev':
        candidate_db = ct.DEV_CANDIDATES_DB
        ref = ct.DEV_TREC_REFERENCE
    elif _set == 'test':
        candidate_db = ct.TEST_CANDIDATES_DB
        ref = ct.TEST_TREC_REFERENCE
    else:
        raise ValueError(f'Unknown set {_set}.')

    connection = sqlite3.connect(candidate_db)
    cursor = connection.cursor()

    question_ids = cursor.execute("SELECT DISTINCT question_id FROM candidates").fetchall()
    run = Run()

    for question_id_tuple in tqdm(question_ids):
        question_id = question_id_tuple[0]
        raw_ranking = cursor.execute(
            "select doc_title, tfidf from candidates where question_id = ? order by cast(tfidf as number) desc",
            [question_id]).fetchall()
        ranking = {json.loads(doc_title): float(pseudo_rank) for ((doc_title, tfidf), pseudo_rank) in
                   zip(raw_ranking, reversed(range(len(raw_ranking))))}
        run.add_ranking(json.loads(question_id), ranking)
    cursor.close()
    connection.close()

    evaluator = Evaluator(ref, pytrec_eval.supported_measures)
    (trec_eval, trec_eval_agg) = evaluator.evaluate(run, save=False)

    er_10 = 0
    for stats in trec_eval.values():
        er_10 += stats['recall_10'] == 1.0
    er_10 /= len(trec_eval)

    print(f'ndcg@10:\t\t{trec_eval_agg["ndcg_cut_10"]:.4f}')
    print(f'map@10:\t\t{trec_eval_agg["map_cut_10"]:.4f}')
    print(f'er@10:\t\t{er_10:.4f}')
    print(f'recall@10:\t\t{trec_eval_agg["recall_10"]:.4f}')
    print(f'recall@100:\t\t{trec_eval_agg["recall_100"]:.4f}')
    print(f'recall@1000:\t\t{trec_eval_agg["recall_1000"]:.4f}')


if __name__ == '__main__':
    run_eval('dev')
