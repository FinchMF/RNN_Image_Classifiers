############################
# MODEL PERFORMANCE GRAPHS #
############################

import numpy as np

def graph_model_loss(gru_loss, lstm_loss):
    loss_gru = np.loadtxt(gru_loss).astype(np.float)
    loss_lstm = np.loadtxt(lstm_loss).astype(np.float)
    model_losses = np.vstack((loss_gru, loss_lstm))

    return model_losses 


def graph_model_acc(gru_acc, lstm_acc):
    acc_gru = np.loadtxt(gru_acc).astype(np.float)
    acc_lstm = np.loadtxt(lstm_acc).astype(np.float)
    model_acc = np.vstack((acc_gru, acc_lstm))

    return model_acc