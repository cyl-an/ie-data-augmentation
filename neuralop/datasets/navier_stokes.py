from pathlib import Path
import torch
import random

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding
from .ie_augmentation.iels_ns import *


def time_series_to_pairs(U):
    w, h = U.size(2), U.size(3)
    U_cat = torch.empty((0, 2, w, h))
    for j in range(U.size(1) - 1):
        U_cat = torch.cat((U_cat, U[:, [j, j + 1], :, :]), dim=0)

    return U_cat


def load_navier_stokes_pt(
        data_path,
        n_train,
        n_test,
        batch_size,
        test_batch_sizes,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        data_augmentation=False,
        pde_params_nu=0.001
):
    """Load the dataset"""
    data = torch.load(
        data_path
    )
    data = data.float()
    torch.manual_seed(0)
    random.seed(0)

    data1 = data[:, 10:, :, :]
    number_of_pairs = data1.size(1) - 1
    s0 = n_train // number_of_pairs
    data_train = time_series_to_pairs(data1[0:s0])
    data_train = torch.cat((data_train, time_series_to_pairs(data1[s0:s0 + 1, 0:n_train % number_of_pairs + 1])))

    data2 = data[:, 10:, :, :]
    number_of_pairs = data2.size(1) - 1
    data_test = time_series_to_pairs(data2[-(n_test // number_of_pairs) - 1: -1, 0:])
    data_test = torch.cat((data_test, time_series_to_pairs(data2[-1:, 0:n_test % number_of_pairs + 1])))


    x_train = data_train[:, 0:1, :, :]
    x_test = data_test[:, 0:1, :, :]

    y_train = data_train[:, 1:2, :, :]
    y_test = data_test[:, 1:2, :, :]

    if data_augmentation is True:
        y_random = y_train[torch.randperm(y_train.size(0), generator=torch.Generator().manual_seed(1))]
        y_random2 = y_train[torch.randperm(y_train.size(0), generator=torch.Generator().manual_seed(2))]
        if pde_params_nu == 1e-3:
            y_com = (y + 2).float()
        if pde_params_nu == 1e-4:
            y_com = (0.2*y + 5).float()
        x_com = order3_ns(y_com, dt_back=0.5).float()
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
