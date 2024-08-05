import sys
import yaml
import torch.nn.functional as F

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import matplotlib.pyplot as plt

from neuralop.models import TFNO2d
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_navier_stokes_pt
from neuralop.training import setup
from neuralop.training.callbacks import MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import count_params

# read configure.yaml
with open('configure/navier_stokes_1e-4.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset_params = config['dataset']
model_params = config['model']
pde_params = config['pde']

dataset_params['data_path'] = dataset_params['data_path'].format(pde_params['nu'])
model_params['save_path'] = model_params['save_path'].format(pde_params['nu'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the dataset
train_loader, test_loaders = load_navier_stokes_pt(
    data_path=dataset_params['data_path'],
    n_train=dataset_params['n_train'],
    n_test=dataset_params['n_test'],
    batch_size=dataset_params['batch_size'],
    test_batch_sizes=dataset_params['test_batch_sizes'],
    positional_encoding=dataset_params['positional_encoding'],
    data_augmentation=dataset_params['data_augmentation'],
    pde_params_nu=pde_params['nu']
)

model = TFNO2d(
    n_modes_height=model_params['n_modes_height'],
    n_modes_width=model_params['n_modes_width'],
    hidden_channels=model_params['hidden_channels'],
    out_channels=model_params['out_channels'],
    projection_channels=model_params['projection_channels'],
    norm=model_params['norm'],
    skip=model_params['skip'],
    use_mlp=model_params['use_mlp'],
    factorization=model_params['factorization'],
    rank=model_params['rank']
)

model = model.to(device)

n_params = count_params(model)
model = model.to(device)

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=2.5e-3,
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=60, gamma=0.5
)

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

trainer = Trainer(model=model, n_epochs=501,
                  device=device,
                  wandb_log=False,
                  log_test_interval=100,
                  log_output=True,
                  use_distributed=False,
                  verbose=True,
                  callbacks=[
                      MGPatchingCallback(levels=0,
                                         padding_fraction=0,
                                         stitching=False),
                      SimpleWandBLoggerCallback()
                  ]
                  )

trainer.train(
    train_loader=train_loader,
    test_loaders={128: test_loaders},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses
)

torch.save(model.state_dict(), model_params['save_path'])
