import json

import pyndri
from tqdm import tqdm

from dataset.dataset import Dataset
import main_constants as c
from retrieval.evaluate import Run, Evaluator
from retrieval.index import Index
from retrieval.tokenizer import Tokenizer

dataset = Dataset(c.TRAINING_SET)
index = Index()
tokenizer = Tokenizer()

tfidf_query_env = pyndri.TFIDFQueryEnvironment(index.index)

run = Run()
evaluator = Evaluator(c.TRECEVAL_REFERENCE_TRAIN)

for question in tqdm(dataset, unit='questions'):

    tokenized = ' '.join(tokenizer.tokenize(question.question))
    result = tfidf_query_env.query(tokenized)

    ranking = {index.wid2title[index.internal2external(_id)]: float(score)
               for (_id, _), score in zip(result[:10], reversed(range(1, 11)))}

    run.add_question(question.id, ranking)

with open(c.TRECEVAL_RESULTS_TRAIN, 'w') as run_file:
    json.dump(run, run_file, indent=True)

evaluation = evaluator.evaluate(run)

with open(c.TRECEVAL_EVALUATION_TRAIN, 'w') as evaluation_file:
    json.dump(evaluation, evaluation_file, indent=True)
    