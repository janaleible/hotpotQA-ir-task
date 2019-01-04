from services import helpers, parallel
import main_constants as ct
from typing import Tuple, List, Any
import json


def build() -> None:
    helpers.log('Building Trec Eval references.')
    with open(ct.TRAIN_HOTPOT_SET, 'r') as file:
        question_set = json.load(file)
        dev_question_set = question_set[ct.TRAIN_DEV_SPLIT:]
        train_question_set = question_set[:ct.TRAIN_DEV_SPLIT]
    with open(ct.DEV_HOTPOT_SET, 'r') as file:
        test_question_set = json.load(file)
    iterator = [('train', train_question_set), ('dev', dev_question_set), ('test', test_question_set)]
    for _set, reference in parallel.execute(_build_reference, iterator):
        helpers.log(f'Created reference {reference} for {_set} set.')


def _build_reference(question_set: List[Any]) -> Tuple[str, str]:
    _set, question_set = question_set
    reference = {}
    for question in question_set:
        titles = (title for (title, _) in question['supporting_facts'])
        reference[question['_id']] = {title: 1 for title in titles}
    if _set == 'train':
        reference_path = ct.TRAIN_TREC_REFERENCE
    elif _set == 'dev':
        reference_path = ct.DEV_TREC_REFERENCE
    elif _set == 'test':
        reference_path = ct.TEST_TREC_REFERENCE
    else:
        raise ValueError('Unknown trec reference.')

    with open(reference_path, 'w') as file:
        json.dump(reference, file, indent=True)

    return _set, reference_path
