import os
import subprocess
from datetime import datetime

from main_constants import NO_INDEXES
from services import parallel
import logging

logging.basicConfig(level='INFO')


def build():
    start = datetime.now()
    for _ in parallel.execute(_build, range(NO_INDEXES)):
        pass
    logging.info(f'Built {NO_INDEXES} indexes in {datetime.now() - start}')
    return


def _build(index_no: int):
    logging.info(f'Started building index {index_no}')
    os.makedirs(f'./data/index/indri_{index_no}')
    subprocess.call(f'IndriBuildIndex build_indri_index.xml -index=./data/index/indri_{index_no}/', shell=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    logging.info(f'Finished building index {index_no}')

    return index_no
