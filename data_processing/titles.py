from main_constants import RAW_DATA_DIR, INDEX_DIR, WID2TITLE, TITLE2WID
from datetime import datetime
from services import parallel, helpers
from typing import Dict
from glob import glob
import logging
import pickle
import json
import bz2
import os

logging.basicConfig(level='INFO')


def build():
    assert os.path.exists(RAW_DATA_DIR), f'Cannot find raw data in {os.path.abspath(RAW_DATA_DIR)}'
    os.makedirs(os.path.abspath(INDEX_DIR), exist_ok=True)

    folder_paths = sorted(glob(os.path.join(RAW_DATA_DIR, '*')))
    title2wid = {}
    wid2title = {}
    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tBuilding title maps.')
    for group_title2wid in parallel.execute(_build_group_title_map, folder_paths):
        for (title, wid) in group_title2wid.items():
            if title2wid.get(title, None) is None:
                title2wid[title] = wid
                wid2title[wid] = title
            else:
                # Hack for taking care of the double title that points to a proper article and to a disambiguation
                # article. Assumes the only article of interest is the one that is not a disambiguation one.
                helpers.log(f'Title {title} has the WID {title2wid.get(title)}. Current WID: {wid}.')
                if wid == 2209045:
                    helpers.log(f'Replacing WID {title2wid.get(title)} with WID {wid}.')
                    title2wid[title] = wid
                    wid2title[wid] = title

    with open(WID2TITLE, 'wb') as file:
        pickle.dump(wid2title, file)
    with open(TITLE2WID, 'wb') as file:
        pickle.dump(title2wid, file)
    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tBuilt title maps.')

    return


def _build_group_title_map(folder_path: str) -> Dict[str, int]:
    title2wid: Dict[str, int] = {}

    file_paths = sorted(glob(os.path.join(folder_path, '*.bz2')))
    for file_path in file_paths:
        with bz2.BZ2File(file_path) as file:
            for line in file:
                doc = json.loads(line)
                doc_wid, doc_title = int(doc['id']), doc['title']

                if title2wid.get(doc_title, None) is None:
                    title2wid[doc_title] = doc_wid
                else:
                    # Hack for taking care of the double title that points to a proper article and to a disambiguation
                    # article. Assumes the only article of interest is the one that is not a disambiguation one.
                    helpers.log(f'Title {doc_title} has the WID {title2wid.get(doc_title)}. Current WID: {doc_wid}.')
                    if doc_wid == 2209045:
                        helpers.log(f'Replacing WID {title2wid.get(doc_title)} with WID {doc_wid}.')
                        title2wid[doc_title] = doc_wid

    logging.info(f'[{datetime.now()}]\t[{os.getpid()}]\tBuilt title maps for folder {folder_path.split("/")[-1]}.')
    return title2wid
