import argparse
import torch
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run GCN-UOD.')
    parser.add_argument('--model', default='GCN-OUD', help='Model name.')
    parser.add_argument('--data', default='SpamBase', help='Datafile name.')
    parser.add_argument('--k', type=int, default=10, help='Num. of Generators.')
    parser.add_argument('--lr_E', type=float, default=1e-4, help='Learning rate for net E.')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Learning rate for net G.')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs.')
    parser.add_argument('--alpha', type=float, default=1, help='Weight parameter of rec. loss.')
    parser.add_argument('--drop_one', action='store_true', help='If dropping the target value 1.')

    return parser.parse_args()


def lossE(rec_data, conf, org_data, hp_alpha, device):
    b = torch.bernoulli(torch.Tensor(conf.size()).uniform_(0, 1)).to(device)
    conf = conf * b + (1 - b)
    adjusted_rec_data = conf * rec_data + (1 - conf) * org_data
    rec_loss = torch.mean((adjusted_rec_data - org_data) ** 2)
    con_loss = torch.mean(-torch.log(conf))
    loss = rec_loss + hp_alpha * con_loss

    return loss, rec_loss, con_loss


def load_data(datafile_name):
    data_path = "./data/{}.csv".format(datafile_name, names=False)
    data = pd.read_csv(data_path, header=None)
    Y = data.pop(0).values
    X = data.values
    del data

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)

    min_value, _ = torch.min(X, dim=0)
    max_value, _ = torch.max(X, dim=0)
    X = (X - min_value + 1e-8) / (max_value - min_value + 1e-8)

    return X, Y


def add_noise(data, noise_factor=0.1):
    noise = torch.randn_like(data) * noise_factor
    return data + noise


def plot_history(history, save_path, model_name):
    save_path = save_path + '/history.png'

    epochs = np.arange(1, len(history['errE']) + 1)
    plt.plot(epochs, history['errE'], label='Estimator Loss')
    if model_name == 'ACE':
        plt.plot(epochs, history['rec_err'], label='Reconstruction Loss')
        plt.plot(epochs, history['con_err'], label='Confidence Loss')
    plt.plot(epochs, history['auc'], label='AUC')
    if model_name == 'GCN-UOD':
        plt.plot(epochs, history['errG'], label='Generator Loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def get_save_path(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    subfolders = [f.name for f in os.scandir(base_path) if f.is_dir()]

    exp_folders = [f for f in subfolders if re.match(r'^exp\d+$', f)]

    if len(subfolders) == 0:
        new_folder = 'exp'
    elif len(subfolders) == 1 and subfolders[0] == 'exp':
        new_folder = 'exp1'
    else:
        max_number = max(int(re.search(r'\d+', f).group()) for f in exp_folders)
        new_folder = f'exp{max_number + 1}'

    save_path = base_path + '/' + new_folder
    os.makedirs(save_path)

    return save_path


def save_results(history, outlier_score, y, model_info):
    base_path = './results'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    save_path = get_save_path(base_path)

    plot_history(history, save_path, model_info['model_name'])  # plot and save history
    pd.DataFrame(history).to_csv(save_path + '/history.csv', index=False)  # save history as csv
    torch.save(
        model_info['model'].state_dict(), save_path + f"/{model_info['model_name']}_{model_info['data_name']}.pt"
    )

    with open(save_path + '/time.txt', 'w') as f:
        f.write(f"{model_info['time']} s")

    with open(save_path + '/auc.txt', 'w') as f:
        f.write(f"{history['auc'][-1]:.4f}")
    print("Results can be found in '" + save_path + "'.")
