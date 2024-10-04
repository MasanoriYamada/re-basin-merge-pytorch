import os
import random
import numpy as np
import torch
import copy
from distutils.util import strtobool
from collections import OrderedDict


def flatten_params(model):
    return model.state_dict()


def params_to_1dvec_except_bn_i(params):
    ret = []
    for key in params:
        if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key) and ('identity' not in key):
            ret.append(params[key].reshape(-1))
    return torch.cat(ret)


def params_to_1dvec(params):
    ret = []
    for key in params:
        ret.append(params[key].reshape(-1))
    return torch.cat(ret)


def lerp(lam, t1, t2):
    t3 = copy.deepcopy(t1)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def lerp_tesnsor(lam, t1, t2):
    t3 = copy.deepcopy(t2)
    t3 = (1 - lam) * t1 + lam * t2
    return t3


def directbool(x):
    return bool(strtobool(x))


def none_or_str(x):
    if x == 'None':
        return None
    return x

def freeze(x):
    ret = copy.deepcopy(x)
    for key in x:
        ret[key] = x[key].detach()
    return ret


def clone(x):
    ret = OrderedDict()
    for key in x:
        ret[key] = x[key].clone().detach()
    return ret


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def dict_list_to_list_dict(dict_list):
    return [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]


def nan_detect(loss):
    if type(loss) == float:
        if loss != loss:
            print('Warning nan detect!')
            raise StopIteration
    else:
        if (loss != loss).any():
            print('Warning nan detect!')
            raise StopIteration


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('EarlyStop!')
                raise StopIteration
        else:
            self.best_score = score
            self.counter = 0
