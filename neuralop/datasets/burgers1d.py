from pathlib import Path
from re import X
import torch
import random
import numpy as np

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding
from .ie_augmentation.iels_burgers import *


def time_series_to_pairs(U, steps=1):
    w, h = U.size(2), U.size(3)
    U_cat = torch.empty((0, 2, w, h))
    for j in range(U.size(1) - steps):
        U_cat = torch.cat((U_cat, U[:, [j, j + steps], :, :]), dim=0)

    return U_cat


def data_normalization(data, scale=0.5, mean=0):
    mean = torch.mean(data, -1, keepdim=True)
    noise = 0.1 - 0.2 * torch.rand(data.size(0), 1, 1, 1, device=data.device)
    normalized_data = data - mean
    return normalized_data + noise


def load_burgers1d_pt(
        data_path,
        n_train,
        n_test,
        batch_size,
        test_batch_sizes,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        data_augmentation=False,
        pde_params_nu=0.1
):
    """Load the dataset"""
    data_np = np.load(
        data_path
    )
    data_tensor = torch.from_numpy(data_np).float()
    data = data_tensor.squeeze(0).unsqueeze(2)

    torch.manual_seed(0)
    random.seed(0)

    steps = 1
    number_of_pairs = data.size(1) - steps

    s0 = n_train // number_of_pairs
    data_train = time_series_to_pairs(data[0:s0, 0:], steps=steps)
    data_train = torch.cat(
        (data_train, time_series_to_pairs(data[s0:s0 + 1, 0:n_train % number_of_pairs + steps], steps=steps)))
    data_test = time_series_to_pairs(data[-(n_test // number_of_pairs) - 1: -1, 0:], steps=steps)
    data_test = torch.cat((data_test, time_series_to_pairs(data[-1:, 0:n_test % number_of_pairs + steps], steps=steps)))

    x_train = data_train[:, 0:1, :, :]
    x_test = data_test[:, 0:1, :, :]

    y_train = data_train[:, 1:2, :, :]
    y_test = data_test[:, 1:2, :, :]

    data_for_ie = time_series_to_pairs(data[0:1 * s0, 20:], steps=steps)
    y = data_for_ie[:, 1:2, :, :]
    if data_augmentation is True:
        y_random = y[torch.randperm(y.size(0), generator=torch.Generator().manual_seed(1))]
        y_random2 = y[torch.randperm(y.size(0), generator=torch.Generator().manual_seed(2))]
        for i in range(2):
            y_com = ((0.8-0.2*i)*y + 0.1*(i+1)*y_random + 0.1*(i+1)*y_random2)
            # y_com = smooth(y_com, dt=1e-3)
            # y_normalized = data_normalization(y_com)
            y_normalized = y_com
            x_back = order3_burgers1d(y_normalized, pde_params_nu, dt=0.05)
            x_train = torch.cat((x_train, x_back))
            y_train = torch.cat((y_train, y_normalized))

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
