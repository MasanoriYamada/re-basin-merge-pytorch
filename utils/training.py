import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from utils.misc import lerp_tesnsor, dict_mean, nan_detect


def train(args, model, device, train_loader, optimizer, epoch, softmax=False, print_flg=True):
    model.train()
    scaler = GradScaler(enabled=args.amp)
    correct = 0
    train_loss = 0
    grad_dicts = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.autocast(enabled=args.amp, device_type='cuda', dtype=torch.float16):
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            nan_detect(loss)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            scaler.scale(loss).backward()
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            scaler.step(optimizer)
            scaler.update()

        grad_dict = {}
        i = 0
        for key in model.module.state_dict():
            if 'identity' in key:
                grad_dict[key] = torch.zeros_like(model.module.state_dict()[key])
                i = i + 1
            elif ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key) \
                    and ('identity' not in key):
                grad_dict[key] = list(model.parameters())[i].grad
                i = i + 1
            else:
                grad_dict[key] = torch.zeros_like(model.module.state_dict()[key])
        assert i == len(list(model.parameters()))  # check skip grad param and param consistency
        grad_dicts.append(grad_dict)
        if batch_idx % args.log_interval == 0:
            if print_flg:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    if print_flg:
        print('Train Accuracy: ({:.0f}%) '.format(acc))
    return train_loss, acc, dict_mean(grad_dicts)


def train_sam(args, model, device, train_loader, optimizer, epoch, softmax=False, print_flg=True):
    model.train()
    correct = 0
    train_loss = 0
    grad_dicts = []


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        if softmax:
            output = F.log_softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        loss = F.nll_loss(output, target)
        nan_detect(loss)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        output = model(data)
        if softmax:
            output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        nan_detect(loss)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.nll_loss(output, target, reduction='sum').item()

        grad_dict = {}
        i = 0
        for key in model.module.state_dict():
            if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                grad_dict[key] = list(model.parameters())[i].grad
                i = i + 1
            else:
                grad_dict[key] = torch.zeros_like(model.module.state_dict()[key])
        assert i == len(list(model.parameters()))  # check skip grad param and param consistency
        grad_dicts.append(grad_dict)
        if batch_idx % args.log_interval == 0:
            if print_flg:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    if print_flg:
        print('Train Accuracy: ({:.0f}%) '.format(acc))
    return train_loss, acc, dict_mean(grad_dicts)


def train_label_smooth(args, model, device, train_loader, optimizer, epoch, softmax=False, print_flg=True):
    import torch.nn.functional as F
    import torch.nn as nn
    def linear_combination(x, y, epsilon):
        return (1 - epsilon) * x + epsilon * y

    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, epsilon=0.1, reduction='mean', softmax=False):
            super().__init__()
            self.epsilon = epsilon
            self.reduction = reduction
            self.softmax = softmax

        def forward(self, preds, target):
            n = preds.size()[-1]
            if self.softmax:
                preds = F.log_softmax(preds, dim=1)
            loss = reduce_loss(-preds.sum(dim=-1), self.reduction)
            nll = F.nll_loss(preds, target, reduction=self.reduction)
            return linear_combination(nll, loss / n, self.epsilon)
    model.train()
    scaler = GradScaler(enabled=args.amp)
    criterion = LabelSmoothingCrossEntropy(reduction='mean')
    criterion_sum = LabelSmoothingCrossEntropy(reduction='sum')
    correct = 0
    train_loss = 0
    grad_dicts = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.autocast(enabled=args.amp, device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)
            nan_detect(loss)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            scaler.scale(loss).backward()
            train_loss += criterion_sum(output, target).item()
            scaler.step(optimizer)
            scaler.update()

        grad_dict = {}
        i = 0
        for key in model.module.state_dict():
            if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                grad_dict[key] = list(model.parameters())[i].grad
                i = i + 1
            else:
                grad_dict[key] = torch.zeros_like(model.module.state_dict()[key])
        assert i == len(list(model.parameters()))  # check skip grad param and param consistency
        grad_dicts.append(grad_dict)
        if batch_idx % args.log_interval == 0:
            if print_flg:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    if print_flg:
        print('Train Accuracy: ({:.0f}%) '.format(acc))
    return train_loss, acc, dict_mean(grad_dicts)


def test(model, device, test_loader, softmax=False, print_flg=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(), torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if print_flg:
        print('\nAverage loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
            test_loss, acc))
    return test_loss, acc


def double_test(model, device, test_loader_a, test_loader_b, softmax=False, print_flg=True):
    loss_a, acc_a = test(model, device, test_loader_a, softmax, print_flg)
    loss_b, acc_b = test(model, device, test_loader_b, softmax, print_flg)
    return (loss_a + loss_b)/2, (acc_a + acc_b)/2


def test_ensembling(model_a, model_b, lam, device, test_loader, softmax=False, print_flg=True):
    model_a.eval()
    model_b.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(), torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_a = model_a(data)
            output_b = model_b(data)
            output = lerp_tesnsor(lam, output_a, output_b)
            if softmax:
                output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if print_flg:
        print('\nAverage loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
            test_loss, acc))
    return test_loss, acc


def double_test_ensembling(model_a, model_b, lam, device, test_loader_a, test_loader_b, softmax=False, print_flg=True):
    loss_a, acc_a = test_ensembling(model_a, model_b, lam, device, test_loader_a, softmax, print_flg)
    loss_b, acc_b = test_ensembling(model_a, model_b, lam, device, test_loader_b, softmax, print_flg)
    return (loss_a + loss_b)/2, (acc_a + acc_b)/2


def get_lr_schedule(lr_schedule, n_epoch, lr_max=None, lr_drop_epoch=None, lr_one_drop=None):
    if lr_schedule == 'superconverge':
            lr_schedule_func = lambda t: np.interp([t], [0, n_epoch * 2 // 5, n_epoch], [0, lr_max, 0])[0]

    elif lr_schedule == 'piecewise':
        def lr_schedule_func(t):
            if t / n_epoch < 0.5:
                return lr_max
            elif t / n_epoch < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.

    elif lr_schedule == 'linear':
        lr_schedule_func = lambda t: np.interp([t], [0, n_epoch // 3, n_epoch * 2 // 3, n_epoch], [lr_max, lr_max, lr_max / 10, lr_max / 100])[0]
    elif lr_schedule == 'onedrop':
        def lr_schedule_func(t):
            if t < lr_drop_epoch:
                return lr_max
            else:
                return lr_one_drop
    elif lr_schedule == 'multipledecay':
        def lr_schedule_func(t):
            return lr_max - (t//(n_epoch//10))*(lr_max/10)
    elif lr_schedule == 'cosine':
        def lr_schedule_func(t):
            return lr_max * 0.5 * (1 + np.cos(t / n_epoch * np.pi))
    elif lr_schedule == 'cyclic':
        lr_schedule_func = lambda t: np.interp([t], [0, 0.4 * n_epoch, n_epoch], [0, lr_max, 0])[0]
    return lr_schedule_func
