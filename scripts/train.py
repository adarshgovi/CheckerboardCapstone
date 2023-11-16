import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from absl import app, flags
from common_utils import logging
from common_utils.random import RNG, set_random_seed
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt

import sys
# print(sys.path)
# sys.path.append("")
import diffusion_models
from diffusion_models import data
import wandb

logging.support_unobserve()

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")
flags.DEFINE_list("tags", [], "Tags to add to the run.")
flags.DEFINE_string("wandb_name", "checkerboard_diffusion", "wandb name.")
flags.mark_flags_as_required(["config"])


class CheckerMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2),
        )
        self.double()

    def forward(self, x):
        return self.net(x)


# define MSE loss
criterion = nn.MSELoss()


# create an instance of VAE


# optimizer choice

def train(config):
    if config.seed is not None:
        set_random_seed(config.seed)

    # Load the dataset
    train_set, test_set = data.get_datasets(config.data)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.training.batch_size, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.training.batch_size, shuffle=False
    )
    inf_train_loader = itertools.cycle(train_loader)

    # Assuming you have loaded and processed the Checkerboard dataset
    x_coordinates = train_set.data[:, 0]
    y_coordinates = train_set.data[:, 1]

    t_total = 500
    betas = np.linspace(1e-4, 0.3, t_total)
    alpha_t = 1.0 - torch.tensor(betas)
    alpha_bar_ts = torch.cumprod(alpha_t, 0)

    model = CheckerMLP()

    for p in model.parameters():
        torch.nn.init.normal_(p, 0, 0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)

    for iteration in range(config.training.n_iters):
        print(iteration)
        x0 = next(inf_train_loader)  # random point from data set
        timestep = torch.randint(0, t_total - 1, (x0.shape[0], 1))  # random timestep
        x_t, epsilon = forward_pass(x0, timestep, alpha_bar_ts)  # perform forward pass (add noise)
        print("x_t shape")
        print(x_t.shape)
        print("timestep shape")
        print(timestep.shape)
        x_t = torch.cat([x_t, timestep], dim=1)  # include timestep in input to model
        optimizer.zero_grad()  # reset gradients
        eps_theta = model(x_t)  # run through model
        loss = torch.nn.functional.mse_loss(eps_theta, epsilon, reduction='none')  # compute loss
        mean_loss = torch.mean(loss)  # take mean of loss of batch
        wandb.log({"loss": mean_loss})
        mean_loss.backward()  # back prop
        optimizer.step()

    model.eval()
    x0 = next(iter(val_loader))
    x_val = inference_pass(x0, t_total, alpha_t, alpha_bar_ts, model)
    plt.scatter(x_val[:, 0].detach().numpy(), x_val[:, 1].detach().numpy())

    plt.show()


def forward_pass(x0, timestep, alpha_bar_ts):
    epsilon = torch.randn_like(x0)  # random noise
    a_bar_t = alpha_bar_ts[timestep]  # random noise multiplier, determines "how much" based on timestep
    x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * epsilon
    return x_t, epsilon


def inference_pass(x0, timerange, alpha, alpha_bar, model):
    with torch.no_grad():
        x_t = torch.randn_like(x0)
        for timestep in reversed(range(timerange)):
            model.zero_grad()
            print(timestep)
            a_t = alpha[timestep]
            a_bar_t = alpha_bar[timestep]
            timestep_tensor = torch.ones((x0.shape[0], )) * float(timestep)
            print(x_t.shape)
            print(timestep_tensor.shape)
            eps = model(torch.cat([x_t, timestep_tensor.unsqueeze(1)], dim=1))
            z = torch.randn_like(x0)
            x_t = (1 / torch.sqrt(a_t)) * (x_t - (((1 - a_t) / (torch.sqrt(1 - a_bar_t))) * eps)) + torch.sqrt((1 - a_t)) * z * (
                        timestep != 0)

    return x_t


def main(argv):
    print(FLAGS.config.to_dict())
    os.environ['WANDB_ENTITY'] = 'adarshg'
    os.environ['WANDB_PROJECT'] = 'CHECKERBOARD_DIFFUSION'
    logging.init(config=FLAGS.config.to_dict(), tags=FLAGS.tags, project=FLAGS.wandb_name)
    train(FLAGS.config)
    wandb.log({})


if __name__ == "__main__":
    app.run(main)
