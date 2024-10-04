import copy
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import configargparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchopt
import wandb
from torch.nn.utils._stateless import functional_call
# from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datas.load import Loader as TableLoader
from utils.logger import Logger
from utils.misc import directbool
from utils.misc import flatten_params, lerp, freeze, clone
from utils.misc import set_seed, nan_detect
from utils.plot import plot_interp_acc, plot_interp_loss
from utils.repair import reset_bn_stats
from utils.selector import get_data, get_model, get_opt, get_torchopt, get_mpermutation_spec, get_scheduler, get_torchopt_scheduler, get_trans
from utils.training import train, test, test_ensembling, double_test, double_test_ensembling
from matching.weight_matching import weight_matching, apply_permutation, StoreBN
from analysis.metric import get_grad, get_flatness, get_l2, calc_weight_landscape


def get_args():
    # basic
    parser = configargparse.ArgumentParser(default_config_files=['./conf/rotated_mnist/ste_mlp_rotated_mnist_0_90.yaml'])
    parser.add('-c', '--conf', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--dir', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_wandb', type=str, default='offline', choices=['online', 'offline'])
    parser.add_argument('--log_up_model', type=directbool, default=False, choices=[True, False], help='True: updaload model to wandb, False:save local with refefence')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--log_interval_ste', type=int, default=200)
    # data
    parser.add_argument("--data", type=str, default='mnist',
                        choices=['mnist_fmnist', 'usps_mnist', 'mnist', 'fmnist',
                                 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b', 'mnist_rotated_mnist_15',
                                 'mnist_rotated_mnist_30', 'mnist_rotated_mnist_45', 'mnist_rotated_mnist_60', 'mnist_rotated_mnist_75', 'mnist_rotated_mnist_90'])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--length", type=int, default=None, help='data length')
    parser.add_argument("--trans_type", default=None, choices=[None, 'color_32'])
    # model
    parser.add_argument("--model", type=str, default='mlp', choices=['mlp', 'vgg', 'resnet20'])
    parser.add_argument("--width_multiplier", type=int, default=1)
    parser.add_argument("--bias", type=directbool, default=True, choices=[True, False])
    parser.add_argument("--model_a", type=str, required=True,
                        choices=['mnist', 'fmnist', 'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b'])
    parser.add_argument("--model_b", type=str, required=True,
                        choices=['mnist', 'fmnist', 'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b'])
    parser.add_argument("--a_v", type=int, required=True, help='model_a_version on wandb')
    parser.add_argument("--b_v", type=int, required=True, help='model_b_version on wandb')
    parser.add_argument("--a_epoch", type=str, default='100')
    parser.add_argument("--b_epoch", type=str, default='100')
    parser.add_argument('--amp', type=directbool, default=False, help='amp mode')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'clip'])
    parser.add_argument("--clip_head_dim", type=int, default=512)
    # ste2
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument("--opt", type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lr_schedule", default=None,
                        choices=[None, 'warmup_cosine_decay', 'multi_step_lr', 'cosine'])
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument('--repair', type=directbool, default=False)
    # data cond
    parser.add_argument("--train_data_type", type=str, default='coreset', choices=['full', 'coreset', 'data_cond'])
    parser.add_argument("--cond_a", type=str, help='path of cond data a')
    parser.add_argument("--cond_b", type=str, help='path of cond data b')
    parser.add_argument("--epoch_dc", type=int, default=50)
    parser.add_argument("--batch_size_dc", type=int, default=512)
    parser.add_argument("--opt_dc", type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument("--lr_dc", type=float, default=0.001)
    parser.add_argument("--lr_schedule_dc", default=None,
                        choices=[None, 'warmup_cosine_decay', 'multi_step_lr', 'cosine'])
    parser.add_argument("--warmup_epoch_dc", type=int, default=0)
    parser.add_argument("--ipc", type=int, default=10, help='number of data per class')
    # analysis
    parser.add_argument("--landscape", type=directbool, default=False, choices=[True, False])
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = Logger(args)
    with wandb.init(
            project="re-basin-torch",
            anonymous="allow",
            name=f'ste_{args.data}_{args.model}_{args.train_data_type}_{args.ipc}_{args.model_a}:{args.a_v}_{args.model_b}:{args.b_v}_{args.seed}',
            tags=[f"{args.data}", f"{args.model}", 'ste', f"{args.loss}"],
            job_type="matching",
            mode=args.log_wandb,
            group=args.conf,
            sync_tensorboard=True,
    ) as wandb_run:
        log_dir = os.path.join(logger.exp_dir, wandb_run.name, "_".join(wandb_run.tags))
        writer = SummaryWriter(log_dir=log_dir)
        config = wandb.config
        for key in args.__dict__:
            config[key] = args.__dict__[key]
            print(f'{key}: {config[key]}')
        config.device = device

        # load models
        if config.loss == 'clip':
            clip = True
            num_head = config.clip_head_dim
            train_loader, _ = get_data(config.data, config.batch_size, config.length, config.trans_type)
            classes = train_loader.dataset.classes
        else:
            clip = False
            num_head = config.num_classes
            classes = None
        model_name = f'{config.model_a}-{config.model}-weights:v{config.a_v}'
        file_name = f'params_{config.a_epoch}.pt'
        target_path = os.path.join('artifacts', model_name, file_name)
        if not os.path.exists(target_path):
            target_path = Path(wandb_run.use_artifact(model_name).get_path(file_name).download())
        checkpoint_a = torch.load(target_path, map_location=device)
        model_a = get_model(config.model, num_head, config.data, config.trans_type, config.width_multiplier, config.bias, clip,
                  classes, device)
        model_a.load_state_dict(checkpoint_a)
        model_a = model_a.to(device)

        model_name = f'{config.model_b}-{config.model}-weights:v{config.b_v}'
        file_name = f'params_{config.b_epoch}.pt'
        target_path = os.path.join('artifacts', model_name, file_name)
        if not os.path.exists(target_path):
            target_path = Path(wandb_run.use_artifact(model_name).get_path(file_name).download())
        checkpoint_b = torch.load(target_path, map_location=device)
        model_b = get_model(config.model, num_head, config.data, config.trans_type, config.width_multiplier, config.bias, clip,
                  classes, device)
        model_b.load_state_dict(checkpoint_b)
        model_b = model_b.to(device)

        print('model A')
        print(model_a)
        print('model B')
        print(model_b)

        # load data (mnist + fmnist)
        if args.train_data_type in ['coreset', 'data_cond']:
            target_paths = [config.cond_a, config.cond_b] if config.cond_b != 'None' else [config.cond_a]
            trans = get_trans(args.train_data_type, config.data)
            loader = TableLoader(target_paths, trans)
            train_loader = loader.get_train_loader(config.batch_size)
            _, test_loader = get_data(config.data, config.batch_size, config.length, config.trans_type)
        else:
            train_loader, test_loader = get_data(config.data, config.batch_size, config.length, config.trans_type)
        train_loader_a, test_loader_a = get_data(config.model_a, config.batch_size, config.length, config.trans_type)
        train_loader_b, test_loader_b = get_data(config.model_b, config.batch_size, config.length, config.trans_type)
        permutation_spec = get_mpermutation_spec(config.model, config.num_classes, config.width_multiplier, config.bias)
        # Save final_permutation as an Artifact
        artifact = wandb.Artifact(f'{config.data}_{config.model}_ste',
                                  type="permutation",
                                  metadata={
                                      "dataset": f'{config.data}',
                                      "model": f'{config.model}',
                                      "analysis": 'ste'
                                  })
        os.makedirs(f'{logger.exp_dir}/model', exist_ok=True)

        # Best permutation found so far...
        best_perm = None
        best_perm_loss = 999
        best_test_acc = 0
        perm = None
        train_state = model_a.state_dict()
        # init_model_a = deepcopy(model_a)
        model_target = deepcopy(model_a)
        model_b_p = deepcopy(model_b)
        scheduler = get_torchopt_scheduler(config.lr_schedule, config.lr, config.epoch, config.warmup_epoch)

        # init flatness
        # b_train_loader, _ = get_data(config.model_b, config.batch_size, config.length,
        #                              config.trans_type)  # model_b data
        # model_dummy = deepcopy(model_b)
        # updated_grad_b = get_grad(torch.nn.DataParallel(model_dummy), b_train_loader, args, device)
        # l2 = get_l2(flatten_params(init_model_a), flatten_params(model_b))
        # sharpness = get_flatness(flatten_params(init_model_a), flatten_params(model_b), updated_grad_b)
        # writer.add_scalar('l2', l2, 0)
        # writer.add_scalar('sharpness', sharpness, 0)

        def get_custom_schedule(scheduler, steps_per_epoch):
            if scheduler is not None:
                def custom_schedule(step: int) -> float:
                    epoch = step // steps_per_epoch
                    try:
                        return scheduler.get_epoch_values(epoch)[0]
                    except:
                        return scheduler.optimizer.param_groups[0]['lr']
                return custom_schedule
            else:
                return config.lr

        steps_per_epoch = len(train_loader)
        custom_schedule = get_custom_schedule(scheduler, steps_per_epoch)
        optimizer = get_torchopt(config.opt, custom_schedule, config.weight_decay)
        for key in train_state:
            train_state[key] = train_state[key].float()
        opt_state = optimizer.init(train_state)
        try:
            for epoch in tqdm(range(1, args.epoch + 1)):
                correct = 0.
                for i, (x, t) in enumerate(tqdm(train_loader, leave=False)):
                    x = x.to(device)
                    t = t.to(device)
                    global_step = i + epoch * len(train_loader)
                    if global_step % args.log_interval_ste == 0:
                        train_state_copy = clone(train_state)
                    # projection by weight matching
                    perm, _ = weight_matching(permutation_spec,
                                              train_state, flatten_params(model_b),
                                              max_iter=100, init_perm=perm, print_flg=False)
                    projected_params = apply_permutation(permutation_spec, perm, flatten_params(model_b))
                    if global_step % args.log_interval_ste == 0:
                        projected_params_copy = clone(projected_params)
                    # ste
                    ste_params = {}
                    for key in train_state:
                        train_state[key] = train_state[key].detach()  # leaf
                        train_state[key].requires_grad = True
                        train_state[key].grad = None  # optimizer.zero_grad()
                    # straight-through-estimator https://github.com/samuela/git-re-basin/blob/main/src/mnist_mlp_ste2.py#L178
                    for key in projected_params:
                        ste_params[key] = projected_params[key].detach() + (
                                    train_state[key] - train_state[key].detach())

                    midpoint_params = lerp(0.5, freeze(flatten_params(model_a)), ste_params)
                    if global_step % args.log_interval_ste == 0:
                        midpoint_params_copy = clone(midpoint_params)
                    model_target.train()

                    # repair https://arxiv.org/abs/2211.08403
                    if config.repair:
                        model_target.load_state_dict(midpoint_params)  # copy bn state to midpoint_params for bn.
                        reset_bn_stats(model_target, train_loader, device, epochs=1)
                        # avoid batch norm param because can not backprop
                        store_bn = StoreBN()
                        midpoint_params = store_bn.remove_bn(midpoint_params)
                    # stateless function to get grad of train_state via midpoint_params
                    output = functional_call(model_target, midpoint_params, x)
                    loss = F.nll_loss(output, t)
                    nan_detect(loss)
                    loss.backward()
                    # optimize
                    grads = OrderedDict()
                    for key in train_state:
                        if train_state[key].grad is None:
                            grads[key] = torch.zeros_like(train_state[key])
                        else:
                            grads[key] = train_state[key].grad
                    for key in train_state:
                        train_state[key] = train_state[key].detach()  # avoid opt_sate chain
                    updates, opt_state = optimizer.update(grads, opt_state, params=train_state, inplace=False)
                    train_state = torchopt.apply_updates(train_state, updates, inplace=False)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(t.view_as(pred)).sum().item()
                    writer.add_scalar('loss/train', loss.item(), global_step)
                    writer.add_scalar('loss/train_best_perm', best_perm_loss, global_step)
                    if scheduler is not None:
                        writer.add_scalar('lr/ste', custom_schedule(global_step), global_step)
                    else:
                        writer.add_scalar('lr/ste', custom_schedule, global_step)
                    if loss < best_perm_loss:
                        best_perm_loss = loss.item()
                        best_perm = perm
                    if global_step % len(train_loader.dataset) == 0:  # for epoch
                        acc = correct / len(train_loader.dataset)
                        # print(f'train acc: {acc}')
                        writer.add_scalar('acc/train', acc, global_step)
                    if global_step % args.log_interval_ste == 0:
                        model_target.load_state_dict(midpoint_params_copy)
                        model_target.to(device)
                        if config.repair:
                            reset_bn_stats(model_target, train_loader, device, epochs=1)
                        test_loss, test_acc = double_test(model_target, device, test_loader_a, test_loader_b, softmax=False, print_flg=False)
                        # test_loss, test_acc = test(model_target, device, test_loader, softmax=False, print_flg=False)
                        if best_test_acc < test_acc:
                            best_test_acc = test_acc

                        # calc flatness
                        # model_dummy.load_state_dict(projected_params)
                        # updated_grad_b = get_grad(torch.nn.DataParallel(model_dummy), b_train_loader, args, device)
                        # l2 = get_l2(flatten_params(init_model_a), projected_params_copy)
                        # sharpness = get_flatness(flatten_params(init_model_a), projected_params_copy, updated_grad_b)
                        print(f'best test acc: {best_test_acc}')
                        writer.add_scalar('acc/test', test_acc, global_step)
                        writer.add_scalar('acc/test_best', best_test_acc, global_step)
                        writer.add_scalar('loss/test', test_loss, global_step)
                        # writer.add_scalar('l2', l2, epoch)
                        # writer.add_scalar('sharpness', sharpness, epoch)

                if config.epoch == epoch:  # save last & best
                    filename = f"{logger.exp_dir}/model/midpoint_{epoch}.pt"
                    torch.save(midpoint_params_copy, filename)
                    if config.log_up_model:
                        artifact.add_file(filename)
                    else:
                        artifact.add_reference(uri='file://' + os.path.abspath(filename))

                    filename = f"{logger.exp_dir}/model/target_{epoch}.pt"
                    torch.save(train_state_copy, filename)
                    if config.log_up_model:
                        artifact.add_file(filename)
                    else:
                        artifact.add_reference(uri='file://' + os.path.abspath(filename))

                    filename = f"{logger.exp_dir}/model/projected_{epoch}.pt"
                    torch.save(projected_params_copy, filename)
                    if config.log_up_model:
                        artifact.add_file(filename)
                    else:
                        artifact.add_reference(uri='file://' + os.path.abspath(filename))
                if scheduler is not None:
                    scheduler.step(epoch)
        except Exception:
            import traceback
            traceback.print_exc()

        filename = f"{logger.exp_dir}/model/best_perm.pt"
        torch.save(best_perm, filename)
        if config.log_up_model:
            artifact.add_file(filename)
        else:
            artifact.add_reference(uri='file://' + os.path.abspath(filename))
        best_params = apply_permutation(permutation_spec, best_perm, flatten_params(model_b))
        filename = f"{logger.exp_dir}/model/params_best.pt"
        torch.save(best_params, filename)
        if config.log_up_model:
            artifact.add_file(filename)
        else:
            artifact.add_reference(uri='file://' + os.path.abspath(filename))

        lambdas = torch.linspace(0, 1, steps=25)

        test_acc_interp_clever = []
        test_acc_interp_naive = []
        test_acc_interp_ensemble = []
        train_acc_interp_clever = []
        train_acc_interp_naive = []
        train_acc_interp_ensemble = []

        test_loss_interp_clever = []
        test_loss_interp_naive = []
        test_loss_interp_ensemble = []
        train_loss_interp_clever = []
        train_loss_interp_naive = []
        train_loss_interp_ensemble = []

        # smart
        model_a_dict = copy.deepcopy(model_a.state_dict())
        model_b_p_dict = best_params
        for lam in tqdm(lambdas):
            naive_p = lerp(lam, model_a_dict, model_b_p_dict)
            model_b_p.load_state_dict(naive_p)
            if config.repair:  # repair https://arxiv.org/abs/2211.08403
                reset_bn_stats(model_b_p, train_loader, device, epochs=1)
            test_loss, acc = double_test(model_b_p.to(device), device, test_loader_a, test_loader_b)
            # test_loss, acc = test(model_b_p.to(device), device, test_loader)
            test_acc_interp_clever.append(acc)
            test_loss_interp_clever.append(test_loss)
            train_loss, acc = double_test(model_b_p.to(device), device, train_loader_a, train_loader_b)
            # train_loss, acc = test(model_b_p.to(device), device, train_loader)
            train_acc_interp_clever.append(acc)
            train_loss_interp_clever.append(train_loss)

        # ensemble
        model_b.load_state_dict(checkpoint_b)
        for lam in tqdm(lambdas):
            test_loss, acc = double_test_ensembling(model_a.to(device), model_b.to(device), lam, device, test_loader_a,
                                                    test_loader_b)
            # test_loss, acc = test_ensembling(model_a.to(device), model_b.to(device), lam, device, test_loader)
            test_acc_interp_ensemble.append(acc)
            test_loss_interp_ensemble.append(test_loss)
            train_loss, acc = double_test_ensembling(model_a.to(device), model_b.to(device), lam, device,
                                                     train_loader_a, train_loader_b)
            # train_loss, acc = test_ensembling(model_a.to(device), model_b.to(device), lam, device, train_loader)
            train_acc_interp_ensemble.append(acc)
            train_loss_interp_ensemble.append(train_loss)

        # naive
        model_b.load_state_dict(checkpoint_b)
        model_a_dict = copy.deepcopy(model_a.state_dict())
        model_b_dict = copy.deepcopy(model_b.state_dict())
        for lam in tqdm(lambdas):
            naive_p = lerp(lam, model_a_dict, model_b_dict)
            model_b.load_state_dict(naive_p)
            test_loss, acc = double_test(model_b.to(device), device, test_loader_a, test_loader_b)
            # test_loss, acc = test(model_b.to(device), device, test_loader)
            test_acc_interp_naive.append(acc)
            test_loss_interp_naive.append(test_loss)
            train_loss, acc = double_test(model_b.to(device), device, train_loader_a, train_loader_b)
            # train_loss, acc = test(model_b.to(device), device, train_loader)
            train_acc_interp_naive.append(acc)
            train_loss_interp_naive.append(train_loss)

        os.makedirs(f'{logger.exp_dir}/data', exist_ok=True)

        # data concentrate
        train_loss_data_cond, train_acc_data_cond = None, None
        test_loss_data_cond, test_acc_data_cond = None, None
        test_acc_best_data_cond = None
        train_loss_best_data_cond = None
        optimizer_dc = None
        if args.train_data_type in ['coreset', 'data_cond']:
            test_acc_best_data_cond = 0
            train_loss_best_data_cond = 999
            target_paths = [config.cond_a, config.cond_b] if config.cond_b != 'None' else [config.cond_a]
            trans = get_trans(args.train_data_type, config.data)
            loader = TableLoader(target_paths, trans)
            train_loader = loader.get_train_loader(args.batch_size)
            model = get_model(config.model, config.num_classes, config.data, config.trans_type, config.width_multiplier,
                              config.bias)
            model = torch.nn.DataParallel(model)
            optimizer_dc = get_opt(model, config.opt_dc, config.lr_dc, 0)
            scheduler_dc = get_scheduler(config.lr_schedule_dc, optimizer_dc, config.epoch_dc, config.warmup_epoch_dc)
            for epoch in range(args.epoch_dc):
                train_loss_data_cond, train_acc_data_cond, _ = train(args, model.to(device), device, train_loader,
                                                                     optimizer_dc, epoch)
                test_loss_data_cond, test_acc_data_cond = double_test(model.to(device), device, test_loader_a, test_loader_b)
                # test_loss_data_cond, test_acc_data_cond = test(model.to(device), device, test_loader)

                if test_acc_best_data_cond < test_acc_data_cond:
                    test_acc_best_data_cond = test_acc_data_cond
                if train_loss_data_cond < train_loss_best_data_cond:
                    train_loss_best_data_cond = train_loss_data_cond

                writer.add_scalar('acc/train_cond', train_acc_data_cond, epoch)
                writer.add_scalar('acc/test_cond', test_acc_data_cond, epoch)
                writer.add_scalar('acc/test_cond_best', test_acc_best_data_cond, epoch)
                writer.add_scalar('loss/train_cond', train_loss_data_cond, epoch)
                writer.add_scalar('loss/test_cond', test_loss_data_cond, epoch)
                writer.add_scalar('loss/train_cond_best', train_loss_best_data_cond, epoch)
                writer.add_scalar('lr/cond', optimizer_dc.param_groups[0]['lr'], epoch)
                if scheduler_dc is not None:
                    scheduler_dc.step(epoch)

        fig_acc = plot_interp_acc(lambdas,
                                  train_acc_interp_naive, test_acc_interp_naive,
                                  train_acc_interp_clever, test_acc_interp_clever,
                                  train_acc_interp_ensemble, test_acc_interp_ensemble,
                                  test_acc_best_data_cond)
        plt.savefig(
            f"{logger.exp_dir}/data/{config.data}_{config.model}_ste_{config.model_a}_v{config.a_v}_{config.model_b}_v{config.b_v}_{config.seed}_acc.png",
            dpi=300)

        fig_loss = plot_interp_loss(lambdas,
                                    train_loss_interp_naive, test_loss_interp_naive,
                                    train_loss_interp_clever, test_loss_interp_clever,
                                    train_loss_interp_ensemble, test_loss_interp_ensemble,
                                    train_loss_best_data_cond)
        plt.savefig(
            f"{logger.exp_dir}/data/{config.data}_{config.model}_ste_{config.model_a}_v{config.a_v}_{config.model_b}_v{config.b_v}_{config.seed}_loss.png",
            dpi=300)

        dat = {'lambdas': lambdas,
               'train_acc_interp_naive': train_acc_interp_naive,
               'test_acc_interp_naive': test_acc_interp_naive,
               'train_acc_interp_clever': train_acc_interp_clever,
               'test_acc_interp_clever': test_acc_interp_clever,
               'train_acc_interp_ensemble': train_acc_interp_ensemble,
               'test_acc_interp_ensemble': test_acc_interp_ensemble,
               'train_loss_interp_naive': train_loss_interp_naive,
               'test_loss_interp_naive': test_loss_interp_naive,
               'train_loss_interp_clever': train_loss_interp_clever,
               'test_loss_interp_clever': test_loss_interp_clever,
               'train_loss_interp_ensemble': train_loss_interp_ensemble,
               'test_loss_interp_ensemble': test_loss_interp_ensemble,
               }
        df = pd.DataFrame(dat)
        df.to_csv(f'{logger.exp_dir}/data/summary.csv')
        wandb_run.log({
            "log_table": wandb.Table(dataframe=df),
            "interp_acc_fig": wandb.Image(fig_acc),
            "interp_loss_fig": wandb.Image(fig_loss)
        })
        wandb_run.log_artifact(artifact)


if __name__ == "__main__":
    main()
