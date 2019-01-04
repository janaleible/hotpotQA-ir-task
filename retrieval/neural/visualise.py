import csv

from matplotlib.lines import Line2D

from retrieval.neural.configs import Config, models
import matplotlib.pyplot as plt
import main_constants as constants


def plot(config: Config, show=True, output_path=None):

    epochs = []
    loss = []
    training_acc = []
    dev_acc = []

    dev_map_10 = []
    train_map_10 = []

    dev_recall_10  = []
    train_recall_10 = []

    dev_ndcg_10 = []
    train_ndcg_10 = []

    dev_recall_100 = []
    train_recall_100 = []

    train_offset = 2
    dev_offset = 13

    epochs_offset = 0
    loss_offset = 1

    accuracy = 0
    map_cut_10 = 1
    ndcg_cut_10 = 2
    recall_10 = 3
    map_cut_100 = 4
    ndcg_cut_100 = 5
    recall_100 = 6
    map_cut_1000 = 7
    ndcg_cut_1000 = 8
    recall_1000 = 9
    P_5 = 10

    with open(constants.L2R_TRAIN_PROGRESS.format(config.name), 'r') as f:
        reader = csv.reader(f)
        for row in reader:

            if not len(row) == 24: return # catch and ignore old format files

            epochs.append(int(row[epochs_offset]))
            loss.append(float(row[loss_offset]))

            training_acc.append(float(row[train_offset + accuracy]))
            dev_acc.append(float(row[dev_offset + accuracy]))

            dev_map_10.append(float(row[dev_offset + map_cut_10]))
            train_map_10.append(float(row[train_offset + map_cut_10]))

            dev_recall_10.append(float(row[dev_offset + recall_10]))
            train_recall_10.append(float(row[train_offset + recall_10]))

            dev_recall_100.append(float(row[dev_offset + recall_100]))
            train_recall_100.append(float(row[train_offset + recall_100]))

            dev_ndcg_10.append(float(row[dev_offset + ndcg_cut_10]))
            train_ndcg_10.append(float(row[train_offset + ndcg_cut_10]))

    loss[0] = None # fix artificial -1 for loss before first epoch

    plt.figure(figsize=(8, 4))

    loss_color = 'tab:red'
    recall_10_color = 'tab:orange'
    map_10_color = 'tab:green'
    ndcg_10_color = 'tab:blue'

    train_linestyle = '-'
    dev_linestyle = '--'

    plt.plot(epochs, loss, color=loss_color, label='average loss', linestyle=train_linestyle)

    plt.plot(epochs, train_recall_10, color=recall_10_color, linestyle=train_linestyle )
    plt.plot(epochs, dev_recall_10, color=recall_10_color, linestyle=dev_linestyle, label='Recall@10')

    plt.plot(epochs, train_map_10, color=map_10_color, linestyle=train_linestyle)
    plt.plot(epochs, dev_map_10, color=map_10_color, linestyle=dev_linestyle, label='mAP@10')

    plt.plot(epochs, train_ndcg_10, color=ndcg_10_color, linestyle=train_linestyle)
    plt.plot(epochs, dev_ndcg_10, color=ndcg_10_color, linestyle=dev_linestyle, label='NDCG@10')

    plt.xlabel('Epochs')

    legend_keys = [
        Line2D([0], [0], color='tab:gray', linewidth=1, linestyle=train_linestyle),
        Line2D([0], [0], color='tab:gray', linewidth=1, linestyle=dev_linestyle),
        Line2D([0], [0], color=loss_color, linewidth=1, linestyle='-'),
        Line2D([0], [0], color=recall_10_color, linewidth=1, linestyle='-'),
        Line2D([0], [0], color=map_10_color, linewidth=1, linestyle='-'),
        Line2D([0], [0], color=ndcg_10_color, linewidth=1, linestyle='-'),
    ]
    legend_labels = ['train', 'dev', 'loss', 'recall@10', 'mAP@10', 'NDCG@10']
    plt.legend(legend_keys, legend_labels, loc='upper right', fontsize='small')


    if show: plt.show()
    else: plt.savefig(output_path.format(config.name))


if __name__ == '__main__':

    plot(models['max_pool_llr_embeddings_pw'])
