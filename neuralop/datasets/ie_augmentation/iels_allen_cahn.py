import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# phase-field
def pde(x, epsilon):
    n_classes = x.size(1)
    h = x.size(2)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    x_padded = F.pad(x, (1, 1, 1, 1), 'circular')
    return epsilon ** 2 * h ** 2 * F.conv2d(x_padded, laplacian_kernel, groups=n_classes) \
        + (x - x ** 3)


def d_pde(u, u_t, epsilon):
    n_classes = u.size(1)
    h = u.size(2)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    u_t_padded = F.pad(u_t, (1, 1, 1, 1), 'circular')
    return epsilon ** 2 * h ** 2 * F.conv2d(u_t_padded, laplacian_kernel, groups=n_classes) \
        + (1 - 3 * u ** 2) * u_t


def dd_pde(u, u_t, u_tt, epsilon):
    n_classes = u.size(1)
    h = u.size(2)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    u_tt_padded = F.pad(u_tt, (1, 1, 1, 1), 'circular')
    return epsilon ** 2 * h ** 2 * F.conv2d(u_tt_padded, laplacian_kernel, groups=n_classes) \
        + (u_tt - 6 * u * u_t ** 2 - 3 * u ** 2 * u_tt)


def order1(u, epsilon, dt=0.5):
    u_t = pde(u, epsilon)
    u_tt = d_pde(u, u_t, epsilon)
    out = u - dt * u_t
    return out


def order2(u, epsilon, dt=0.5):
    u_t = pde(u, epsilon)
    u_tt = d_pde(u, u_t, epsilon)
    out = u - dt * u_t + 0.5 * dt ** 2 * u_tt
    return out


def order3(u, epsilon, dt=0.5):
    u_t = pde(u, epsilon)
    u_tt = d_pde(u, u_t, epsilon)
    u_ttt = dd_pde(u, u_t, u_tt, epsilon)
    out = u - dt * u_t + 0.5 * dt ** 2 * u_tt - 1 / 6 * dt ** 3 * u_ttt
    return out
