import json
import sqlite3

import pytrec_eval
from tqdm import tqdm

import main_constants as ct
from services.evaluator import Evaluator
from services.run import Run


def run_eval():
    connection = sqlite3.connect(ct.TEST_CANDIDATES_DB)
    cursor = connection.cursor()

    question_ids = cursor.execute("SELECT DISTINCT question_id FROM candidates").fetchall()
    run = Run()

    for question_id_tuple in tqdm(question_ids):

        question_id = question_id_tuple[0]

        raw_ranking = cursor.execute("select doc_title, tfidf from candidates where question_id = ? order by cast(tfidf as number) desc", [question_id]).fetchall()

        ranking = {json.loads(doc_title): float(pseudo_rank) for ((doc_title, tfidf), pseudo_rank) in zip(raw_ranking, reversed(range(len(raw_ranking))))}

        run.add_ranking(json.loads(question_id), ranking)

    evaluator = Evaluator(ct.TEST_TREC_REFERENCE, pytrec_eval.supported_measures)

    (_, evaluation) = evaluator.evaluate(run, save=False)

    print('ndcg10', evaluation['ndcg_cut_10'])
    print('map10', evaluation['map_cut_10'])
    print('recall10', evaluation['recall_10'])
    print('recall100', evaluation['recall_100'])
    print('recall1000', evaluation['recall_1000'])


if __name__ == '__main__':
    run_eval()
