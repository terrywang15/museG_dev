import os
import pandas as pd
import matplotlib.pyplot as plt


def make_graphs(log_path, folderpath):
    """

    :param log_path:
    :param folderpath:
    :return:
    """

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    log_smooth = pd.read_csv(log_path)
    log_smooth = log_smooth.groupby("epoch").mean().drop("batch", axis=1)
    y_ul = log_smooth.g_loss.std() * 6

    # Make graph for all losses
    fig, ax = plt.subplots()
    ax.plot(log_smooth.d_loss_real, label="d_loss_real")
    ax.plot(log_smooth.d_loss_fake, label="d_loss_fake")
    ax.plot(log_smooth.g_loss, label="g_loss")
    ax.legend(loc=2)
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, len(log_smooth)))
    ax.set_ylim(0)
    ax.set_title("All Losses")
    fig.savefig(folderpath + "/all_losses.png")

    fig, ax = plt.subplots()
    ax.plot(log_smooth.d_loss_real, label="d_loss_real")
    ax.legend(loc=2)
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, len(log_smooth)))
    ax.set_ylim(0)
    ax.set_title("Discriminator Real Sample Losses")
    fig.savefig(folderpath + "/real_losses.png")

    fig, ax = plt.subplots()
    ax.plot(log_smooth.d_loss_fake, label="d_loss_fake")
    ax.legend(loc=2)
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, len(log_smooth)))
    ax.set_ylim(0)
    ax.set_title("Discriminator Fake Sample Losses")
    fig.savefig(folderpath + "/fake_losses.png")

    fig, ax = plt.subplots()
    ax.plot(log_smooth.g_loss, label="g_loss")
    ax.legend(loc=2)
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, len(log_smooth)))
    ax.set_ylim(0)
    ax.set_title("GAN (generator) Losses")
    fig.savefig(folderpath + "/gan_losses.png")


logPath = "2020-03-05 00-38/loss_history.csv"
folderPath = "2020-03-05 00-38/loss_graphs"

make_graphs(logPath, folderPath)
