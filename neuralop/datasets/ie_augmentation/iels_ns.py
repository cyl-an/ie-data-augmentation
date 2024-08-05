import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Navier-Stokes
resolution = 128
Nx = resolution
Ny = resolution
Lx = Ly = 1
x, dx = np.linspace(0, Lx, resolution, endpoint=False, retstep=True)
y, dy = np.linspace(0, Ly, resolution, endpoint=False, retstep=True)
X, Y = np.meshgrid(x, y)
X, Y = torch.from_numpy(X), torch.from_numpy(Y)
force = 0.1 * (torch.sin(2 * torch.pi * (X + Y)) + torch.cos(2 * torch.pi * (X + Y)))

k_max = math.floor(resolution / 2.0)
# Wavenumbers in y-direction
k_y = 2 * math.pi * torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                               torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(resolution, 1)
# Wavenumbers in x-direction
k_x = k_y.transpose(0, 1)
# Negative Laplacian in Fourier space
lap = k_x ** 2 + k_y ** 2
lap[0, 0] = 1.0
dealias = (torch.logical_and(torch.abs(k_y) <= 2 * math.pi * (2.0/3.0)*k_max,
            torch.abs(k_x) <= 2 * math.pi * (2.0/3.0)*k_max).float()).unsqueeze(0).unsqueeze(0)


def pde_ns(psi_h, omega_h, force_h, nu):
    psi_x = torch.fft.ifft2(1j * k_x * psi_h).real
    psi_y = torch.fft.ifft2(1j * k_y * psi_h).real
    omega_x = torch.fft.ifft2(1j * k_x * omega_h).real
    omega_y = torch.fft.ifft2(1j * k_y * omega_h).real
    lap_omega_h = lap * omega_h
    return dealias *torch.fft.fft2(psi_x * omega_y - psi_y * omega_x) - nu * lap_omega_h + force_h


def d_pde_ns(psi_h, psi_t_h, omega_h, omega_t_h, nu):
    psi_x = torch.fft.ifft2(1j * k_x * psi_h).real
    psi_y = torch.fft.ifft2(1j * k_y * psi_h).real
    psi_tx = torch.fft.ifft2(1j * k_x * psi_t_h).real
    psi_ty = torch.fft.ifft2(1j * k_y * psi_t_h).real
    omega_x = torch.fft.ifft2(1j * k_x * omega_h).real
    omega_y = torch.fft.ifft2(1j * k_y * omega_h).real
    omega_tx = torch.fft.ifft2(1j * k_x * omega_t_h).real
    omega_ty = torch.fft.ifft2(1j * k_y * omega_t_h).real
    lap_omega_t_h = lap * omega_t_h
    return torch.fft.fft2(
        psi_tx * omega_y + psi_x * omega_ty - psi_ty * omega_x - psi_y * omega_tx) - nu * lap_omega_t_h


def dd_pde_ns(psi_h, psi_t_h, psi_tt_h, omega_h, omega_t_h, omega_tt_h, nu):
    psi_x = torch.fft.ifft2(1j * k_x * psi_h).real
    psi_y = torch.fft.ifft2(1j * k_y * psi_h).real
    psi_tx = torch.fft.ifft2(1j * k_x * psi_t_h).real
    psi_ty = torch.fft.ifft2(1j * k_y * psi_t_h).real
    psi_ttx = torch.fft.ifft2(1j * k_x * psi_tt_h).real
    psi_tty = torch.fft.ifft2(1j * k_y * psi_tt_h).real
    omega_x = torch.fft.ifft2(1j * k_x * omega_h).real
    omega_y = torch.fft.ifft2(1j * k_y * omega_h).real
    omega_tx = torch.fft.ifft2(1j * k_x * omega_t_h).real
    omega_ty = torch.fft.ifft2(1j * k_y * omega_t_h).real
    omega_ttx = torch.fft.ifft2(1j * k_x * omega_tt_h).real
    omega_tty = torch.fft.ifft2(1j * k_y * omega_tt_h).real
    lap_omega_tt_h = lap * omega_tt_h
    return torch.fft.fft2(psi_ttx * omega_y + 2 * psi_tx * omega_ty + psi_x * omega_tty - psi_tty * omega_x - \
                          2 * psi_ty * omega_tx - psi_y * omega_ttx) - nu * lap_omega_tt_h


def order1_ns(omega, nu, dt_back):
    omega_h = torch.fft.fft2(omega)
    psi_h = omega_h / lap
    force_h = torch.fft.fft2(force)
    omega_t_h = pde_ns(psi_h, omega_h, force_h, nu)
    omega_h = omega_h - dt_back * omega_t_h
    omega = torch.fft.ifft2(omega_h)
    return omega.real


def order2_ns(omega, nu, dt_back):
    omega_h = torch.fft.fft2(omega)
    psi_h = omega_h / lap
    force_h = torch.fft.fft2(force)
    omega_t_h = pde_ns(psi_h, omega_h, force_h, nu)
    psi_t_h = omega_t_h / lap
    omega_tt_h = d_pde_ns(psi_h, psi_t_h, omega_h, omega_t_h, nu)
    omega_h = omega_h - dt_back * omega_t_h + 1 / 2 * dt_back ** 2 * omega_tt_h
    omega = torch.fft.ifft2(omega_h)
    return omega.real


def order3_ns(omega, nu, dt_back):
    omega_h = torch.fft.fft2(omega)
    psi_h = omega_h / lap
    force_h = torch.fft.fft2(force)
    omega_t_h = pde_ns(psi_h, omega_h, force_h, nu)
    psi_t_h = omega_t_h / lap
    omega_tt_h = d_pde_ns(psi_h, psi_t_h, omega_h, omega_t_h, nu)
    psi_tt_h = omega_tt_h / lap
    omega_ttt_h = dd_pde_ns(psi_h, psi_t_h, psi_tt_h, omega_h, omega_t_h, omega_tt_h, nu)
    omega_h = omega_h - dt_back * omega_t_h + 1 / 2 * dt_back ** 2 * omega_tt_h - 1 / 6 * dt_back ** 3 * omega_ttt_h
    omega = torch.fft.ifft2(omega_h)
    return omega.real
