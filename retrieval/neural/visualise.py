import csv

from retrieval.neural.configs import Config
import matplotlib.pyplot as plt
import main_constants as constants

def plot(config: Config, show=True):

    epochs = []
    loss = []
    training_acc = []
    dev_acc = []
    map_10 = []
    ndcg_10 = []
    recall_10 = []

    with open(constants.L2R_TRAIN_PROGRESS.format(config.name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            epochs.append(int(row[0]))
            loss.append(float(row[1]))
            training_acc.append(float(row[2]))
            dev_acc.append(float(row[3]))
            map_10.append(float(row[4]))
            ndcg_10.append(float(row[5]))
            recall_10.append(float(row[6]))

    plt.plot(epochs, loss, color='tab:orange', label='average loss')
    plt.plot(epochs, training_acc, color='tab:blue', label='training set accuracy')
    plt.plot(epochs, dev_acc, color='tab:green', label='test set accuracy')
    plt.xlabel('Epochs')
    plt.xticks(range(1,len(epochs)))
    plt.legend(loc='upper left')

    plt.savefig(constants.L2R_LEARNING_PROGRESS_PLOT.format(config.name))

    if show: plt.show()
