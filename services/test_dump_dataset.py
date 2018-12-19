import main_constants as ct
from services import run
import pickle
import json


def hotpot():
    with open('models/max_pool_llr_pw/runs/dev.full.pickle', 'rb') as file:
        dev_run: run.Run = pickle.load(file)
    with open('models/max_pool_llr_pw/runs/dev.full.hotpot.json', 'w') as file:
        json.dump(dev_run.to_json(ct.DEV_FEATURES_DB, ct.TRAIN_HOTPOT_SET), file)


if __name__ == '__main__':
    hotpot()
