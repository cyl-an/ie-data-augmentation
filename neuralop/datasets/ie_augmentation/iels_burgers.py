import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# burgers1d
# nu_burgers = 0.1
def pde_burgers1d(u, nu_burgers):
    n_classes = u.size(1)
    h = u.size(3)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    kernel_x = torch.tensor([[[[0., 0., 0.],
                               [-0.5, 0., 0.5],
                               [0., 0., 0.]]]]).to(device=device)
    kernel_y = torch.tensor([[[[0., 0.5, 0.],
                               [0., 0., 0.],
                               [0., -0.5, 0.]]]]).to(device=device)

    u_padded = F.pad(u, (1, 1, 1, 1), 'circular')
    return nu_burgers / torch.pi * h ** 2 * F.conv2d(u_padded, laplacian_kernel, groups=n_classes) \
        - h * u * (F.conv2d(u_padded, kernel_x, groups=n_classes))


def d_pde_burgers1d(u, ut, nu_burgers):
    n_classes = u.size(1)
    h = u.size(3)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    kernel_x = torch.tensor([[[[0., 0., 0.],
                               [-0.5, 0., 0.5],
                               [0., 0., 0.]]]]).to(device=device)
    kernel_y = torch.tensor([[[[0., 0.5, 0.],
                               [0., 0., 0.],
                               [0., -0.5, 0.]]]]).to(device=device)

    ut_padded = F.pad(ut, (1, 1, 1, 1), 'circular')
    u_padded = F.pad(u, (1, 1, 1, 1), 'circular')
    return nu_burgers / torch.pi * h ** 2 * F.conv2d(ut_padded, laplacian_kernel, groups=n_classes) \
        - h * ut * (F.conv2d(u_padded, kernel_x, groups=n_classes) + F.conv2d(u_padded, kernel_y, groups=n_classes)) \
        - h * u * (F.conv2d(ut_padded, kernel_x, groups=n_classes) + F.conv2d(ut_padded, kernel_y, groups=n_classes))


def dd_pde_burgers1d(u, ut, utt, nu_burgers):
    n_classes = u.size(1)
    h = u.size(3)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32).to(device=device).repeat(n_classes, 1, 1, 1)
    kernel_x = torch.tensor([[[[0., 0., 0.],
                               [-0.5, 0., 0.5],
                               [0., 0., 0.]]]]).to(device=device)
    kernel_y = torch.tensor([[[[0., 0.5, 0.],
                               [0., 0., 0.],
                               [0., -0.5, 0.]]]]).to(device=device)

    utt_padded = F.pad(utt, (1, 1, 1, 1), 'circular')
    ut_padded = F.pad(ut, (1, 1, 1, 1), 'circular')
    u_padded = F.pad(u, (1, 1, 1, 1), 'circular')
    return nu_burgers / torch.pi * h ** 2 * F.conv2d(utt_padded, laplacian_kernel, groups=n_classes) \
        - h * utt * (F.conv2d(u_padded, kernel_x, groups=n_classes) + F.conv2d(u_padded, kernel_y, groups=n_classes)) \
        - 2 * h * ut * (
                F.conv2d(ut_padded, kernel_x, groups=n_classes) + F.conv2d(ut_padded, kernel_y, groups=n_classes)) \
        - h * u * (F.conv2d(utt_padded, kernel_x, groups=n_classes) + F.conv2d(utt_padded, kernel_y, groups=n_classes))


def order2_burgers1d(u, nu_burgers, dt=0.01):
    u_t = pde_burgers1d(u, nu_burgers)
    u_tt = d_pde_burgers1d(u, u_t, nu_burgers)
    out = u - dt * u_t + 0.5 * dt ** 2 * u_tt
    return out


def order3_burgers1d(u, nu_burgers, dt=0.01):
    u_t = pde_burgers1d(u, nu_burgers)
    u_tt = d_pde_burgers1d(u, u_t, nu_burgers)
    u_ttt = dd_pde_burgers1d(u, u_t, u_tt, nu_burgers)
    out = u - dt * u_t + 0.5 * dt ** 2 * u_tt - 1 / 6 * dt ** 3 * u_ttt
    return out
