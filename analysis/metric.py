import copy
import torch
from torch.nn import NLLLoss
import torch.optim as optim
from utils.misc import lerp, params_to_1dvec_except_bn_i
from utils.training import train, test
from analysis.loss_landscape import WeightLandscape1d, save_matplotlib1d


def get_l2(params_a, params_b):
    params_a = params_to_1dvec_except_bn_i(params_a)
    params_b = params_to_1dvec_except_bn_i(params_b)
    return (params_a - params_b).norm(p='fro').detach().item() / len(params_a)


def get_flatness(params_a, params_b, grad_b):
    params_a = params_to_1dvec_except_bn_i(params_a)
    params_b = params_to_1dvec_except_bn_i(params_b)
    grad_b = params_to_1dvec_except_bn_i(grad_b)
    flatness = torch.einsum('i,i->', grad_b, params_a - params_b)  # dl/dpb * (a - pb)
    return flatness


def get_barrier(model_a, model_b, data_loader, device):
    params_ab = lerp(0.5, model_a.state_dict(), model_b.state_dict())
    model = copy.deepcopy(model_a)
    model.load_state_dict(params_ab)
    loss, acc = test(model, device, data_loader)
    return loss


def get_grad(model, data_loader, args, device):
    opt = optim.SGD(model.parameters(), lr=0.0, momentum=0.0, weight_decay=0.0)
    train_loss, train_acc, grad_dict = train(args, model, device, data_loader, opt, 0, print_flg=False)
    return grad_dict


def calc_weight_landscape(model, data_loader, distance, steps, save_name, seed):
    model.eval()
    wl = WeightLandscape1d()
    loss_fn = NLLLoss(reduction='sum')
    loss_data_1d = wl.get_landscape(model, loss_fn, data_loader, distance=distance,
                                    steps=steps, seed=seed)
    print('calc landascape1d')
    loss_data_1d = loss_data_1d.mean(axis=1)
    save_matplotlib1d(save_name, loss_data_1d, distance, steps)
    return loss_data_1d
