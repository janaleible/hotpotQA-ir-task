import csv

from retrieval.neural.configs import Config, models
import matplotlib.pyplot as plt
import main_constants as constants


def plot(config: Config, show=True):

    epochs = []
    loss = []
    training_acc = []
    dev_acc = []

    with open(constants.L2R_TRAIN_PROGRESS.format(config.name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            epochs.append(int(row[0]))
            loss.append(float(row[1]))
            training_acc.append(float(row[2]))
            dev_acc.append(float(row[9]))

    loss[0] = None # fix artificial -1 for loss before first epoch

    plt.figure(figsize=(8, 4))

    plt.plot(epochs, loss, color='tab:orange', label='average loss')
    plt.plot(epochs, training_acc, color='tab:blue', label='training set accuracy')
    plt.plot(epochs, dev_acc, color='tab:green', label='dev set accuracy')
    plt.xlabel('Epochs')
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
    plt.legend(loc='best')

    plt.savefig(constants.L2R_LEARNING_PROGRESS_PLOT.format(config.name))

    if show: plt.show()


if __name__ == '__main__':

    plot(models['max_pool_llr+features_pw'])