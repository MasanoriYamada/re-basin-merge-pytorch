import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.misc import set_seed


class WeightLandscape1d(object):
    def __init__(self):
        self.model = None
        self.data_loader = None

    def get_landscape(self, model, loss_fn, data_loader, distance=1, steps=50, seed=None):
        self.model = model
        self.loss_fn = loss_fn
        assert self.loss_fn.reduction in ['mean', 'sum']
        self.device = next(model.parameters()).device
        self.loss_fn_each = copy.deepcopy(self.loss_fn)
        self.loss_fn_each.reduction = 'none'
        self.data_loader = data_loader

        # model params
        original_state_dict = copy.deepcopy(self.model.state_dict())
        start_point_w = {key: self.model.state_dict()[key] for key in self.model.state_dict() if ('weigh' in key)}
        self.start_point_w = copy.deepcopy(start_point_w)  # deepcopy

        # normalize
        if seed is not None:
            set_seed(seed)  # fix seed
        direction = rand_u_like(self.start_point_w)
        direction = filter_normalize(direction, self.start_point_w)

        # alpha grid
        if steps == 1:
            alphas = np.linspace(0, 1, 1)
        else:
            alphas = np.linspace(-distance / 2, distance / 2, steps)

        loss_landscape = []
        for alpha in tqdm(alphas):
            new_point_w = {}
            for key in original_state_dict:
                if key in direction.keys():  # only weight
                    new_point_w[key] = original_state_dict[key] + alpha * direction[key]
                    # for i in range(original_state_dict[key].shape[0]):
                    #    print('scale', key, original_state_dict[key][i].norm(),  direction[key][i].norm())
                    # print('scale', key, original_state_dict[key].norm(), direction[key].norm())
                else:
                    new_point_w[key] = original_state_dict[key]

            self.model.load_state_dict(new_point_w)
            local_loss = []
            for batch_idx, (x, target) in enumerate(self.data_loader):
                if type(self.loss_fn) == torch.nn.modules.loss.BCELoss:  # for BCELoss only accept float tensor
                    target = target.float()
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                local_loss.append(self.loss_fn_each(output, target).detach())
            loss_landscape.append(torch.cat(local_loss))
        loss_landscape = torch.stack(loss_landscape)
        self.model.load_state_dict(original_state_dict)  # recover model
        return loss_landscape.cpu().numpy()


def filter_normalize(direction, weight, order=2):
    scaled_direction = {}
    for key in weight:
        # if 'conv' in key:
        if (torch.float == weight[key].dtype):
            if weight[key].dim() == 1:
                pass  # ignore bn/weight bn/bias fc/bias
            else:
                scaled_direction[key] = torch.zeros_like(weight[key])
                for i in range(weight[key].shape[0]):
                    scaled_direction[key][i] = direction[key][i] * (weight[key][i].norm(order) / (
                            direction[key][i].norm(order) + 1e-10))  # conv/weight fc/weight
                # scaled_direction[key] = direction[key] * (weight[key].norm(order) / direction[key].norm(
                #    order) + 1e-10)  # conv/weight fc/weight
        elif torch.int64 == weight[key].dtype:
            pass
    return scaled_direction


def rand_u_like(example_vector):
    new_vector = {}
    for key in example_vector:
        if torch.float == example_vector[key].dtype:
            new_vector[key] = torch.randn(size=example_vector[key].size(), dtype=example_vector[key].dtype).to(
                example_vector[key].device)
        elif torch.int64 == example_vector[key].dtype:
            pass
    return new_vector


def save_matplotlib1d(save_name, data, distance, steps):
    import matplotlib.pyplot as plt
    xs = np.linspace(-distance / 2, distance / 2, steps)
    fig = plt.figure()
    plt.plot(xs, data)
    plt.savefig(
        save_name,
        dpi=300)
