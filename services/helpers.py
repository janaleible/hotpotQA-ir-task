from main_constants import TRAIN_HOTPOT_SET
from datetime import datetime
import logging
import os

logging.basicConfig(level='INFO')


def log(*args) -> None:
    """Misc logging formatting."""
    terms = [f'[{datetime.now()}]', f'[{os.getpid()}]']
    terms.extend(args)
    logging.info('\t'.join(terms))


def training_set_id():
    return TRAIN_HOTPOT_SET.split('/')[-1].split('.json')[0].split('_')[-1]
