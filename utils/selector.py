import numpy as np
import math
import torch
import torch.optim as optim
import torchopt
from utils.sam import SAM
from timm.scheduler import CosineLRScheduler
from torchvision import transforms
from collections import OrderedDict
from matching.weight_matching import mlp_permutation_spec, vgg11_permutation_spec, resnet20_permutation_spec, resnet20_i_permutation_spec


def cls_test_transforms(size, mean, std):
    upper_size = int(math.pow(2, math.ceil(math.log2(size))))
    return transforms.Compose([
        transforms.Resize(upper_size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def cls_train_transforms(size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_trans(trans_type, data_name):
    if trans_type is None:
        transform = None
    elif trans_type == 'color_32':
        if data_name == 'mnist' or data_name == 'usps':
            transform = transforms.Compose([transforms.Resize([32, 32]),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor()])
    elif trans_type == 'coreset':
        if data_name == 'usps_mnist':
            transform = transforms.Compose([transforms.Resize([32, 32]),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor()])
        elif data_name == 'mnist_rotated_mnist_90':
            transform = transforms.Compose([transforms.ToTensor()])
        elif data_name == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
    elif trans_type == 'data_cond':
        transform = transforms.Compose([])

    else:
        transform = transforms.Compose([])
    return transform


def get_data(name, batch_size, length, trans_type):
    print(f'dataset: {name}')
    print(f'transform: {trans_type}')
    if name == 'mnist':
        from datas.mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'usps':
        from datas.usps import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'fmnist':
        from datas.fmnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'cifar10':
        from datas.cifar10 import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'cifar10_split_a':
        from datas.cifar10_split_a import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'cifar10_split_b':
        from datas.cifar10_split_b import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(batch_size, transform, length)
        test_loader = get_test_loader(batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_0':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(0, batch_size, transform, length)
        test_loader = get_test_loader(0, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_15':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(15, batch_size, transform, length)
        test_loader = get_test_loader(15, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_30':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(30, batch_size, transform, length)
        test_loader = get_test_loader(30, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_45':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(45, batch_size, transform, length)
        test_loader = get_test_loader(45, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_60':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(60, batch_size, transform, length)
        test_loader = get_test_loader(60, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_75':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(75, batch_size, transform, length)
        test_loader = get_test_loader(75, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'rotated_mnist_90':
        from datas.rotated_mnist import get_train_loader, get_test_loader
        transform = get_trans(trans_type, name)
        train_loader = get_train_loader(90, batch_size, transform, length)
        test_loader = get_test_loader(90, batch_size, transform, length)
        return train_loader, test_loader

    elif name == 'mnist_fmnist':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.fmnist import get_train_data as fmnist_train
        from datas.fmnist import get_test_data as fmnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'fmnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        fm_dataset = fmnist_train(transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, fm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        fm_dataset = fmnist_test(transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, fm_dataset)
        return train_loader, test_loader

    elif name == 'usps_mnist':
        from datas.usps import get_train_data as usps_train
        from datas.usps import get_test_data as usps_test
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'usps')
        transform_b = get_trans(trans_type, 'mnist')
        # train
        us_dataset = usps_train(transform_a, length)
        m_dataset = mnist_train(transform_b, length)
        train_loader = merge_train(batch_size, us_dataset, m_dataset)
        # test
        us_dataset = usps_test(transform_a, length)
        m_dataset = mnist_test(transform_b, length)
        test_loader = merge_test(batch_size, us_dataset, m_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_15':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(15, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(15, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_30':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(30, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(30, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_45':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(45, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(45, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_60':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(60, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(60, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_75':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(75, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(75, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    elif name == 'mnist_rotated_mnist_90':
        from datas.mnist import get_train_data as mnist_train
        from datas.mnist import get_test_data as mnist_test
        from datas.rotated_mnist import get_train_data as rotated_mnist_train
        from datas.rotated_mnist import get_test_data as rotated_mnist_test
        from datas.double import get_train_loader as merge_train
        from datas.double import get_test_loader as merge_test
        transform_a = get_trans(trans_type, 'mnist')
        transform_b = get_trans(trans_type, 'rotated_mnist')
        # train
        m_dataset = mnist_train(transform_a, length)
        rm_dataset = rotated_mnist_train(90, transform_b, length)
        train_loader = merge_train(batch_size, m_dataset, rm_dataset)
        # test
        m_dataset = mnist_test(transform_a, length)
        rm_dataset = rotated_mnist_test(90, transform_b, length)
        test_loader = merge_test(batch_size, m_dataset, rm_dataset)
        return train_loader, test_loader

    else:
        print('ERROR: args.data is nothing data:{}'.format(name))


def get_input_shape(data_name, trans_type):
    if trans_type == 'color_32':
        input_shape = 3 * 32 * 32
    else:
        if data_name in ['mnist_fmnist', 'usps_mnist', 'mnist', 'fmnist', 'usps', 'mnist_fmnist', 'rotated_mnist_0', 'rotated_mnist_15',
                         'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60', 'rotated_mnist_75',
                         'rotated_mnist_90', 'mnist_rotated_mnist_15', 'mnist_rotated_mnist_30', 'mnist_rotated_mnist_45',
                         'mnist_rotated_mnist_60', 'mnist_rotated_mnist_75', 'mnist_rotated_mnist_90']:
            input_shape = 28 * 28
        elif data_name in ['cifar10', 'cifar10_split_a', 'cifar10_split_b']:
            input_shape = 3 * 32 * 32
    return input_shape


def get_model(name, num_classes, data_name, trans_type, width_multiplier=1, bias=True, clip=False, classes=None, device='cuda'):
    input_shape = get_input_shape(data_name, trans_type)
    if name == 'mlp':
        from models.mlp import MLP
        model = MLP(input=input_shape, num_classes=num_classes, bias=bias)
    elif name == 'mlp2':
        from models.mlp import MLP2
        model = MLP2(input=input_shape, num_classes=num_classes, bias=bias)
    elif name == 'vgg11':
        from models.vgg import VGG
        model = VGG(vgg_name='VGG11', num_classes=num_classes, w=width_multiplier, input_shape=input_shape)
    elif name == 'vgg19':
        from models.vgg import VGG
        model = VGG(vgg_name='VGG19', num_classes=num_classes, w=width_multiplier, input_shape=input_shape)
    elif name == 'resnet20':
        from models.resnet import resnet20
        model = resnet20(w=width_multiplier, num_classes=num_classes, clip=clip, classes=classes, device=device)
    return model


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return


def load_model(model, load_model_path):
    if load_model_path:
        print('===> Load model.., {}'.format(load_model_path))
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        data = torch.load(open(load_model_path, 'rb'), map_location=map_location)
        checkpoint = data['model']
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        checkpoint = fix_model_state_dict(checkpoint)  # for data parallel
        try:
            model.load_state_dict(checkpoint)
        except:
            model.net.load_state_dict(checkpoint)
    return model


def get_opt(model, name, lr, weight_decay):
    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif name == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=lr, momentum=0.9)
    return optimizer


def get_torchopt(name, lr, weight_decay):
    if name == 'sgd':
        optimizer = torchopt.sgd(lr=lr, momentum=0.9, weight_decay=weight_decay, moment_requires_grad=True)
    elif name == 'adam':
        optimizer = torchopt.adam(lr=lr, moment_requires_grad=True)
    return optimizer


def get_scheduler(name, opt, total_epoch, warmup_epoch):
    if name is None:
        return None
    if name == 'warmup_cosine_decay':
        scheduler = CosineLRScheduler(opt, t_initial=total_epoch, lr_min=1e-6, warmup_t=warmup_epoch,
                                      warmup_lr_init=1e-6, warmup_prefix=True)
    elif name == 'multi_step_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[int(total_epoch * 0.5), int(total_epoch * 0.75)],
                                                         gamma=0.1)
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, float(total_epoch)
        )
    elif name == 'linear_up_down':
        lr_schedule = np.interp(np.arange(1 + total_epoch), [0, warmup_epoch, total_epoch], [0, 1, 0])
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    else:
        print('ERROR: args.scheduler is nothing scheduler:{}'.format(name))

    return scheduler


def get_torchopt_scheduler(name, lr, total_epoch, warmup_epoch):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(1, 1))], lr=lr)  # dummy
    if name is None:
        return None
    if name == 'warmup_cosine_decay':
        scheduler = CosineLRScheduler(opt, t_initial=total_epoch, lr_min=1e-6, warmup_t=warmup_epoch,
                                      warmup_lr_init=1e-6, warmup_prefix=True)
    elif name == 'multi_step_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[int(total_epoch * 0.5), int(total_epoch * 0.75)],
                                                         gamma=0.1)
    elif name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, float(total_epoch)
        )
    elif name == 'linear_up_down':
        lr_schedule = np.interp(np.arange(1 + total_epoch), [0, warmup_epoch, total_epoch], [0, 1, 0])
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    else:
        print('ERROR: args.scheduler is nothing scheduler:{}'.format(name))

    return scheduler


def get_mpermutation_spec(name, num_classes, width_multiplier, bias, add_i=False):
    if name == 'mlp':
        ps = mlp_permutation_spec(3, bias)
    elif name == 'vgg11':
        # model = get_model(name, num_classes, data_name, trans_type, width_multiplier, bias)
        ps = vgg11_permutation_spec()
    elif name == 'resnet20':
        if add_i:
            ps = resnet20_i_permutation_spec()
        else:
            ps = resnet20_permutation_spec()
    return ps
