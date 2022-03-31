from typing import Tuple

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_2d_decision_boundary(net: nn.Module, title: str=None, rng: Tuple[int, int] = (-2, 2),
                              pos_points: str = 'synthetic_data/rectangles/positive.json',
                              neg_points: str = 'synthetic_data/rectangles/negative.json',
                              test_points: str = 'synthetic_data/rectangles/test.json',
                              ):
    """
    Plots the decision boundary of a 2D neural network.
    net : R^2 -> R^2
    xx_points: json file of list of 2d points
    """

    xrange = np.arange(rng[0], rng[1], 0.01)
    yrange = np.arange(rng[0], rng[1], 0.01)
    x, y = np.meshgrid(xrange, yrange)
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    xy = torch.stack((x.reshape(-1), y.reshape(-1))).T
    with torch.no_grad():
        z = net(xy).detach().argmax(1).numpy().reshape(x.shape)
    plt.contourf(x, y, z, levels=[0,1/2,1], cmap=plt.cm.RdYlBu)

    # plot positive points as pluses
    with open(pos_points) as f:
        pos_points = json.load(f)
    pos_points = np.array(pos_points)
    # plot with a small point size
    plt.scatter(pos_points[:, 0], pos_points[:, 1], s=1, c='red')

    # plot negative points as minuses
    with open(neg_points) as f:
        neg_points = json.load(f)
    neg_points = np.array(neg_points)
    plt.scatter(neg_points[:, 0], neg_points[:, 1], color='blue', s=1)

    # plot test points as o's
    with open(test_points) as f:
        test_points = json.load(f)
    test_points = np.array(test_points)
    plt.scatter(test_points[:, 0], test_points[:, 1], color='gray', s=1)
    plt.title(title)
    plt.show()
