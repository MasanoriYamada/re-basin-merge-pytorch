import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from torch.nn.utils import _stateless


def new_fim(module: nn.Module, _input: torch.Tensor,
            parameters: dict[str, nn.Parameter] = None
            ) -> list[torch.Tensor]:
    # https://github.com/pytorch/pytorch/issues/49171
    if parameters is None:
        parameters = dict(module.named_parameters())
    with torch.no_grad():
        _output = module(_input)  # (N, C)
        prob = F.softmax(_output, dim=1).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
    keys, values = zip(*parameters.items())

    def func(*params: torch.Tensor):
        _output = _stateless.functional_call(module, {n: p for n, p in zip(keys, params)}, _input)
        return F.log_softmax(_output, dim=1)  # (N, C)

    jacobian_list: tuple[torch.Tensor] = torch.autograd.functional.jacobian(func, values)

    fim_list: list[torch.Tensor] = []
    for jacobian in tqdm(jacobian_list):  # Todo: parallel
        jacobian = jacobian.flatten(start_dim=2)  # (N, C, D)
        fim = prob * jacobian.unsqueeze(-1) * jacobian.unsqueeze(-2)  # (N, C, D, D)
        fim_list.append(fim.sum(1).mean(0))  # (D, D)
    return fim_list


if __name__ == "__main__":
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    inp = torch.rand(10, 2)
    print('start')
    fim = new_fim(model, inp)
    for key, w_fim in zip(model.state_dict(), fim):
        print(key, w_fim)
    print('end')