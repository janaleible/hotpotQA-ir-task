from services import helpers, parallel
import main_constants as const
from typing import Tuple
from glob import glob
import json
import os


def build() -> None:
    helpers.log('Building Trec Eval references.')

    question_set_paths = sorted(glob(os.path.join(const.HOTPOT_DIR, '*')))
    for data, reference in parallel.execute(_build_reference, question_set_paths):
        helpers.log(f'Created reference {reference} from {data}.')


def _build_reference(question_set_path: str) -> Tuple[str, str]:
    with open(question_set_path, 'r') as file:
        question_set = json.load(file)
    reference = {}
    for question in question_set:
        titles = (title for (title, _) in question['supporting_facts'])
        reference[question['_id']] = {title: 1 for title in titles}

    reference_path = question_set_path.replace('/hotpot/', '/trec_eval/').replace('.json', '_reference.json')
    with open(reference_path, 'w') as file:
        json.dump(reference, file)

    return question_set_path, reference_path
