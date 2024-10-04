import numpy as np
import torch
from collections import OrderedDict


def fisher_connection(params_a, params_b, params_a_grad, params_b_grad):
    # diagonal app
    eps = 1e-16  # float sq
    param_names = list(params_a.keys())
    c = OrderedDict()
    for key in param_names:
        org_shape = params_a[key].shape
        a = params_a[key].reshape(-1)
        b = params_b[key].reshape(-1)
        a_grad = params_a_grad[key].reshape(-1)
        b_grad = params_b_grad[key].reshape(-1)
        hess_a = a_grad ** 2
        hess_b = b_grad ** 2
        hess_a = torch.clip(hess_a, min=eps)
        hess_b = torch.clip(hess_b, min=eps)
        # match_weight = np.dot(np.linalg.inv(2 * (hess_a + hess_b)), (np.dot(F + F.T, a) + np.dot(G + G.T, b)).T)
        # all app
        # inv_hessian = 1/(hess_a + hess_b)
        # match_weight = inv_hessian * (torch.einsum('i,i->i', hess_a, a) + torch.einsum('i,i->i', hess_b, b))
        coef_a = hess_a / (hess_a + hess_b)
        coef_b = hess_b / (hess_a + hess_b)
        match_weight = coef_a * a + coef_b * b

        # inv app
        # tmp_a = torch.einsum('i,->i', a_grad, torch.einsum('i,i->', a_grad, a))
        # tmp_b = torch.einsum('i,->i', b_grad, torch.einsum('i,i->', b_grad, b))
        # match_weight = inv_hessian * (tmp_a + tmp_b)

        c[key] = match_weight.reshape(org_shape)
    return c


def linear_connection(params_a, params_b):
    # diagonal app
    param_names = list(params_a.keys())
    c = OrderedDict()
    for key in param_names:
        org_shape = params_a[key].shape
        a = params_a[key].reshape(-1)
        b = params_b[key].reshape(-1)
        match_weight = 0.5 * (a + b)
        c[key] = match_weight.reshape(org_shape)
    return c