from pathlib import Path
import torch
import random

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding
from .ie_augmentation.iels_allen_cahn import *


def time_series_to_pairs(U):
    w, h = U.size(2), U.size(3)
    U_cat = torch.empty((0, 2, w, h))
    for j in range(U.size(1) - 1):
        U_cat = torch.cat((U_cat, U[:, [j, j + 1], :, :]), dim=0)

    return U_cat


def load_allen_cahn_pt(
        data_path,
        n_train,
        n_test,
        batch_size,
        test_batch_sizes,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        data_augmentation=False,
        pde_params_epsilon=0.05
):
    """Load the dataset"""
    data = torch.load(
        data_path
    )
    torch.manual_seed(0)
    random.seed(0)

    data = data[:, 0:33, :, :]

    number_of_pairs = data.size(1) - 1

    s0 = n_train // number_of_pairs
    data_train = time_series_to_pairs(data[0:s0, 0:])
    data_train = torch.cat((data_train, time_series_to_pairs(data[s0:s0 + 1, 0:n_train % number_of_pairs + 1])))
    data_test = time_series_to_pairs(data[-(n_test // number_of_pairs) - 1: -1, 0:])
    data_test = torch.cat((data_test, time_series_to_pairs(data[-1:, 0:n_test % number_of_pairs + 1])))


    x_train = data_train[:, 0:1, :, :]
    x_test = data_test[:, 0:1, :, :]

    y_train = data_train[:, 1:2, :, :]
    y_test = data_test[:, 1:2, :, :]

    if data_augmentation is True:
        y_random = y_train[torch.randperm(y_train.size(0), generator=torch.Generator().manual_seed(1))]
        y_random2 = y_train[torch.randperm(y_train.size(0), generator=torch.Generator().manual_seed(2))]
        y_com = 0.8*y_train + 0.1*y_random + 0.1*y_random2
        x_com = order3(y_com, pde_params_epsilon, dt=0.5)
        x_train = torch.cat((x_train, x_com))
        y_train = torch.cat((y_train, y_com))

    del data

    test_batch_size = test_batch_sizes

    train_db = TensorDataset(
        x_train,
        y_train,
        transform_x=PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
        transform_x=PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, test_loader
