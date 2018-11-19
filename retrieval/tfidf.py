import json
import pyndri
from tqdm import tqdm

from dataset.dataset import Dataset
import main_constants as c
from retrieval.evaluate import Run, Evaluator
from retrieval.index import Index
from retrieval.tokenizer import Tokenizer


def run_tfidf(dataset: Dataset):

    index = Index()
    tfidf_query_env = pyndri.TFIDFQueryEnvironment(index.index)
    tokenizer = Tokenizer()
    run = Run()

    for question in tqdm(dataset, unit='questions'):

        tokenized = ' '.join(tokenizer.tokenize(question.question))
        result = tfidf_query_env.query(tokenized)

        ranking = {index.wid2title[index.internal2external(_id)]: float(score)
                   for (_id, _), score in zip(result[:10], reversed(range(1, 11)))}

        run.add_question(question.id, ranking)

    return run


if __name__ == '__main__':

    dataset = Dataset(c.TRAINING_SET)
    evaluator = Evaluator(c.TRECEVAL_REFERENCE_TRAIN)

    # run = run_tfidf(dataset)

    # with open(c.TRECEVAL_RESULTS_TRAIN, 'w') as run_file:
    #     json.dump(run, run_file, indent=True)
    #

    with open(c.TRECEVAL_RESULTS_TRAIN) as run_file:
        run = json.load(run_file)

    with open(c.TRECEVAL_REFERENCE_TRAIN, 'r') as reference_file:
        reference = json.load(reference_file)

    evaluation = evaluator.evaluate(run)

    with open(c.TRECEVAL_EVALUATION_TRAIN, 'w') as evaluation_file:
        json.dump(evaluation, evaluation_file, indent=True)

    maps = []
    ndgcs = []
    hits_at_two = []
    hits_at_ten = []

    for qid, question in evaluation.items():
        maps.append(question['map'])
        ndgcs.append(question['ndcg'])

    for qid, ranking in run.items():

        gold = dataset.find_by_id(qid).gold_articles
        list_ranking = sorted(ranking, key=ranking.get, reverse=True)

        hits_at_two.append(0.5 * len(set(gold).intersection(list_ranking[:1])))
        hits_at_ten.append(0.5 * len(set(gold).intersection(list_ranking)))



    print(f'Mean mAP: {round(sum(maps) / len(maps), 4)}')
    print(f'hits@2: {round(sum(hits_at_two) / len(hits_at_two), 4)}')
    print(f'hits@10: {round(sum(hits_at_ten) / len(hits_at_ten), 4)}')
