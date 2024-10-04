import copy
import os
from pathlib import Path

import configargparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.logger import Logger
from utils.plot import plot_interp_acc, plot_interp_loss
from utils.selector import get_data, get_model, get_mpermutation_spec
from utils.training import test, test_ensembling, double_test, double_test_ensembling
from utils.misc import directbool, set_seed, flatten_params, lerp, dict_list_to_list_dict
from matching.weight_matching import weight_matching, flat_weight_matching_v2, apply_permutation, add_identity_to_net
from utils.repair import reset_bn_stats
from utils.connection import fisher_connection, linear_connection
from analysis.metric import get_grad, get_flatness, get_l2, calc_weight_landscape, get_barrier


def get_args():
    # basic
    parser = configargparse.ArgumentParser(default_config_files=['./conf/weight_match_mlp_mnist.yaml'])
    parser.add('-c', '--conf', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--dir', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_wandb', type=str, default='offline', choices=['online', 'offline'])
    parser.add_argument('--log_up_model', type=directbool, default=False, choices=[True, False], help='True: updaload model to wandb, False:save local with refefence')
    parser.add_argument('--log_interval', type=int, default=1000)
    # data
    parser.add_argument("--data", type=str, default='mnist',
                        choices=['mnist_fmnist', 'usps_mnist', 'mnist', 'fmnist', 'usps',
                                 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                 'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90',
                                 'mnist_rotated_mnist_15',
                                 'mnist_rotated_mnist_30', 'mnist_rotated_mnist_45', 'mnist_rotated_mnist_60',
                                 'mnist_rotated_mnist_75', 'mnist_rotated_mnist_90',
                                 ])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--length", type=int, default=None, help='data length')
    parser.add_argument("--trans_type", default=None, choices=[None, 'color_32'])
    # model
    parser.add_argument("--model", type=str, default='mlp', choices=['mlp', 'vgg11', 'resnet20'])
    parser.add_argument("--width_multiplier", type=int, default=1)
    parser.add_argument("--bias", type=directbool, default=True, choices=[True, False])
    parser.add_argument("--add_i", type=directbool, default=False, choices=[True, False])
    parser.add_argument("--model_a", type=str, required=True,
                        choices=['mnist', 'fmnist', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                 'rotated_mnist_15', 'rotated_mnist_30',
                                 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90',
                                 ])
    parser.add_argument("--model_b", type=str, required=True,
                        choices=['mnist', 'fmnist', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                 'rotated_mnist_15', 'rotated_mnist_30',
                                 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90',
                                 ])
    parser.add_argument("--a_v", type=int, required=True, help='model_a_version on wandb')
    parser.add_argument("--b_v", type=int, required=True, help='model_b_version on wandb')
    parser.add_argument("--a_epoch", type=str, default='100')
    parser.add_argument("--b_epoch", type=str, default='100')
    parser.add_argument('--amp', type=directbool, default=False, help='amp mode')
    parser.add_argument('--grad_batch', type=int, default=100000, help='batch for estimating gradient')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'clip'])
    parser.add_argument("--clip_head_dim", type=int, default=512)
    # matching
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lam', type=float, default=1, help='1. is min |w_a - w_b|')
    parser.add_argument('--connection', default=None, choices=[None, 'linear', 'fisher'])
    parser.add_argument('--repair', type=directbool, default=False)
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
            name=f'weight_match_{args.data}_{args.model}_{args.model_a}:{args.a_v}_{args.model_b}:{args.b_v}_{args.seed}',
            tags=[f"{args.data}", f"{args.model}", 'weight_match', f"{args.loss}"],
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
        os.makedirs(f'{logger.exp_dir}/data')

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
        # model_a = get_model(config.model, config.num_classes, config.data, config.trans_type, config.width_multiplier,
        #                     config.bias)
        model_a = get_model(config.model, num_head, config.data, config.trans_type, config.width_multiplier, config.bias, clip,
                  classes, device)
        model_a.load_state_dict(checkpoint_a, strict=True)
        model_a = model_a.to(device)

        model_name = f'{config.model_b}-{config.model}-weights:v{config.b_v}'
        file_name = f'params_{config.b_epoch}.pt'
        target_path = os.path.join('artifacts', model_name, file_name)
        if not os.path.exists(target_path):
            target_path = Path(wandb_run.use_artifact(model_name).get_path(file_name).download())
        checkpoint_b = torch.load(target_path, map_location=device)
        # model_b = get_model(config.model, config.num_classes, config.data, config.trans_type, config.width_multiplier,
        #                     config.bias)
        model_b = get_model(config.model, num_head, config.data, config.trans_type, config.width_multiplier, config.bias, clip,
                  classes, device)
        model_b.load_state_dict(checkpoint_b, strict=True)
        model_b = model_b.to(device)

        if config.add_i:
            model_a = add_identity_to_net(model_a)
            model_b = add_identity_to_net(model_b)
            checkpoint_a = model_a.state_dict()
            checkpoint_b = model_b.state_dict()

        print('model A')
        print(model_a)
        print('model B')
        print(model_b)

        # calc grad
        train_loader_a, test_loader_a = get_data(config.model_a, config.grad_batch, config.length,
                                                 config.trans_type)  # model_b data
        grads_a = get_grad(torch.nn.DataParallel(model_a), train_loader_a, args, device)
        train_loader_b, test_loader_b = get_data(config.model_b, config.grad_batch, config.length,
                                                 config.trans_type)  # model_b data
        grads_b = get_grad(torch.nn.DataParallel(model_b), train_loader_b, args, device)

        permutation_spec = get_mpermutation_spec(config.model, config.num_classes, config.width_multiplier, config.bias, config.add_i)
        final_permutation, metrics = flat_weight_matching_v2(permutation_spec, flatten_params(model_a),
                                                          flatten_params(model_b),
                                                          grads_a, grads_b, lam=config.lam, max_iter=5000,
                                                          init_perm=None, print_flg=True)

        updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))
        metrics = dict_list_to_list_dict(metrics)
        for met in metrics:
            wandb.log(met, step=met['step'])
        # check init model analysis
        grad_b = get_grad(torch.nn.DataParallel(model_b), train_loader_b, args, device)
        init_l2 = get_l2(flatten_params(model_a), flatten_params(model_b))
        init_flatness = get_flatness(flatten_params(model_a), flatten_params(model_b), grad_b)
        # distance = 1
        # steps = 21
        # fig_init_loss_landscape = f'{logger.exp_dir}/data/init_w_landscape1d_{distance}.png'
        # init_loss_landscape = calc_weight_landscape(model_b, train_loader_b, distance, steps, fig_init_loss_landscape,
        #                                             config.seed)

        # check final model analysis
        # model a
        final_flatness_a = get_flatness(updated_params, flatten_params(model_a), grads_a)

        # barrier
        model_b.load_state_dict(updated_params)
        barrier_a = get_barrier(model_a, model_b, train_loader_a, device)
        barrier_b = get_barrier(model_a, model_b, train_loader_b, device)

        # model b
        model_b.load_state_dict(updated_params)
        updated_grad_b = get_grad(torch.nn.DataParallel(model_b), train_loader_b, args, device)
        final_l2 = get_l2(flatten_params(model_a), updated_params)
        final_flatness_b = get_flatness(flatten_params(model_a), updated_params, updated_grad_b)

        # local train acc
        train_loss_a_on_a, train_acc_a_on_a = test(model_a.to(device), device, train_loader_a, print_flg=False)
        train_loss_b_on_a, train_acc_b_on_a = test(model_b.to(device), device, train_loader_a, print_flg=False)
        train_loss_a_on_b, train_acc_a_on_b = test(model_a.to(device), device, train_loader_b, print_flg=False)
        train_loss_b_on_b, train_acc_b_on_b = test(model_b.to(device), device, train_loader_b, print_flg=False)

        # local test acc
        test_loss_a_on_a, test_acc_a_on_a = test(model_a.to(device), device, test_loader_a, print_flg=False)
        test_loss_b_on_a, test_acc_b_on_a = test(model_b.to(device), device, test_loader_a, print_flg=False)
        test_loss_a_on_b, test_acc_a_on_b = test(model_a.to(device), device, test_loader_b, print_flg=False)
        test_loss_b_on_b, test_acc_b_on_b = test(model_b.to(device), device, test_loader_b, print_flg=False)

        # loss landscape
        # fig_final_loss_landscape = f'{logger.exp_dir}/data/final_w_landscape1d_{distance}.png'
        # model_b.load_state_dict(updated_params)
        # final_loss_landscape = calc_weight_landscape(model_b, train_loader_b, distance, steps, fig_final_loss_landscape,
        #                                              config.seed)
        # loss_landscape_df = pd.DataFrame({'init': init_loss_landscape, 'final': final_loss_landscape})
        train_flipped_loss = 0.5 * (train_loss_a_on_b + train_loss_b_on_a - train_loss_a_on_a - train_loss_b_on_b)
        test_flipped_loss = 0.5 * (test_loss_a_on_b + test_loss_b_on_a - test_loss_a_on_a - test_loss_b_on_b)
        wandb.log({'direct_l2': init_l2, 'direct_flatness_b': init_flatness}, step=0)
        wandb.log({'direct_l2': final_l2, 'direct_flatness_a': final_flatness_a, 'direct_flatness_b': final_flatness_b,
                   'barrier_a': barrier_a, 'barrier_b': barrier_b, 'train_flipped_loss': train_flipped_loss, 'test_flipped_loss': test_flipped_loss,
                   'train_loss_a_on_a': train_loss_a_on_a, 'train_acc_a_on_a': train_acc_a_on_a,
                   'train_loss_a_on_b': train_loss_a_on_b, 'train_acc_a_on_b': train_acc_a_on_b,
                   'train_loss_b_on_a': train_loss_b_on_a, 'train_acc_b_on_a': train_acc_b_on_a,
                   'train_loss_b_on_b': train_loss_b_on_b, 'train_acc_b_on_b': train_acc_b_on_b,
                   'test_loss_a_on_a': test_loss_a_on_a, 'test_acc_a_on_a': test_acc_a_on_a,
                   'test_loss_a_on_b': test_loss_a_on_b, 'test_acc_a_on_b': test_acc_a_on_b,
                   'test_loss_b_on_a': test_loss_b_on_a, 'test_acc_b_on_a': test_acc_b_on_a,
                   'test_loss_b_on_b': test_loss_b_on_b, 'test_acc_b_on_b': test_acc_b_on_b
                   }, step=1)

        # test against mnist and fmnist
        train_loader, test_loader = get_data(config.data, config.batch_size, config.length, config.trans_type)
        train_loader_a, test_loader_a = get_data(config.model_a, config.batch_size, config.length, config.trans_type)
        train_loader_b, test_loader_b = get_data(config.model_b, config.batch_size, config.length, config.trans_type)
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

        if config.connection == 'linear':
            params_a = copy.deepcopy(model_a.state_dict())
            linear_param = linear_connection(params_a, updated_params)
            model_b.load_state_dict(linear_param)
            if config.repair:
                reset_bn_stats(model_b, train_loader, device, epochs=1)
            test_loss_connect, test_acc_connect = double_test(model_b.to(device), device, test_loader_a, test_loader_b,
                                                              print_flg=False)
            train_loss_connect, train_acc_connect = double_test(model_b.to(device), device, train_loader_a,
                                                                train_loader_b, print_flg=False)
            print(f'linear train acc {train_acc_connect}')
            print(f'linear test acc {test_acc_connect}')


        elif config.connection == 'fisher':
            # load grad
            model_name = f'{config.model_a}-{config.model}-weights:v{config.a_v}'
            file_name = f'grads_{config.a_epoch}.pt'
            target_path = Path(wandb_run.use_artifact(model_name).get_path(file_name).download())
            grads_a = torch.load(target_path, map_location=device)

            model_name = f'{config.model_b}-{config.model}-weights:v{config.b_v}'
            file_name = f'grads_{config.a_epoch}.pt'
            target_path = Path(wandb_run.use_artifact(model_name).get_path(file_name).download())
            grads_b = torch.load(target_path, map_location=device)
            params_a = copy.deepcopy(model_a.state_dict())
            fisher_param = fisher_connection(params_a, updated_params, grads_a, grads_b)
            model_b.load_state_dict(fisher_param)
            test_loss_connect, test_acc_connect = double_test(model_b.to(device), device, test_loader_a, test_loader_b,
                                                              print_flg=False)
            train_loss_connect, train_acc_connect = double_test(model_b.to(device), device, train_loader_a,
                                                                train_loader_b, print_flg=False)
            print(f'fisher train acc {train_acc_connect}')
            print(f'fisher test acc {test_acc_connect}')

        else:
            test_loss_connect, test_acc_connect = None, None
            train_loss_connect, train_acc_connect = None, None

        # ensemble
        print('ensemble')
        model_b.load_state_dict(checkpoint_b, strict=True)
        for lam in tqdm(lambdas):
            test_loss, acc = double_test_ensembling(model_a.to(device), model_b.to(device), lam, device, test_loader_a,
                                                    test_loader_b)
            test_acc_interp_ensemble.append(acc)
            test_loss_interp_ensemble.append(test_loss)
            train_loss, acc = double_test_ensembling(model_a.to(device), model_b.to(device), lam, device,
                                                     train_loader_a, train_loader_b)
            train_acc_interp_ensemble.append(acc)
            train_loss_interp_ensemble.append(train_loss)

        # naive
        print('naive')
        model_b.load_state_dict(checkpoint_b, strict=True)
        model_a_dict = copy.deepcopy(model_a.state_dict())
        model_b_dict = copy.deepcopy(model_b.state_dict())
        for lam in tqdm(lambdas):
            naive_p = lerp(lam, model_a_dict, model_b_dict)
            model_b.load_state_dict(naive_p)
            test_loss, acc = double_test(model_b.to(device), device, test_loader_a, test_loader_b)
            test_acc_interp_naive.append(acc)
            test_loss_interp_naive.append(test_loss)
            train_loss, acc = double_test(model_b.to(device), device, train_loader_a, train_loader_b)
            train_acc_interp_naive.append(acc)
            train_loss_interp_naive.append(train_loss)

        # smart
        print('smart')
        model_b.load_state_dict(updated_params)
        model_b.to(device)
        model_a.to(device)
        model_a_dict = copy.deepcopy(model_a.state_dict())
        model_b_dict = copy.deepcopy(model_b.state_dict())
        for lam in tqdm(lambdas):
            naive_p = lerp(lam, model_a_dict, model_b_dict)
            model_b.load_state_dict(naive_p)
            if config.repair:  # repair https://arxiv.org/abs/2211.08403
                reset_bn_stats(model_b, train_loader, device, epochs=1)
            test_loss, acc = double_test(model_b.to(device), device, test_loader_a, test_loader_b)
            test_acc_interp_clever.append(acc)
            test_loss_interp_clever.append(test_loss)
            train_loss, acc = double_test(model_b.to(device), device, train_loader_a, train_loader_b)
            train_acc_interp_clever.append(acc)
            train_loss_interp_clever.append(train_loss)

        wandb.log({
            "best/train_loss_interp_ensemble": min(train_loss_interp_ensemble),
            "best/test_loss_interp_ensemble": min(test_loss_interp_ensemble),
            "best/train_acc_interp_ensemble": max(train_acc_interp_ensemble),
            "best/test_acc_interp_ensemble": max(test_acc_interp_ensemble),
            "best/train_loss_interp_naive": min(train_loss_interp_naive),
            "best/test_loss_interp_naive": min(test_loss_interp_naive),
            "best/train_acc_interp_naive": max(train_acc_interp_naive),
            "best/test_acc_interp_naive": max(test_acc_interp_naive),
            "best/train_loss_interp_clever": min(train_loss_interp_clever),
            "best/test_loss_interp_clever": min(test_loss_interp_clever),
            "best/train_acc_interp_clever": max(train_acc_interp_clever),
            "best/test_acc_interp_clever": max(test_acc_interp_clever),
        }, step=0)

        df = pd.DataFrame({
            "lambda": [l.item() for l in lambdas],
            "train_loss_interp_ensemble": train_loss_interp_ensemble,
            "test_loss_interp_ensemble": test_loss_interp_ensemble,
            "train_acc_interp_ensemble": train_acc_interp_ensemble,
            "test_acc_interp_ensemble": test_acc_interp_ensemble,
            "train_loss_interp_naive": train_loss_interp_naive,
            "test_loss_interp_naive": test_loss_interp_naive,
            "train_acc_interp_naive": train_acc_interp_naive,
            "test_acc_interp_naive": test_acc_interp_naive,
            "train_loss_interp_clever": train_loss_interp_clever,
            "test_loss_interp_clever": test_loss_interp_clever,
            "train_acc_interp_clever": train_acc_interp_clever,
            "test_acc_interp_clever": test_acc_interp_clever,
        })
        fig_acc = plot_interp_acc(lambdas,
                                  train_acc_interp_naive, test_acc_interp_naive,
                                  train_acc_interp_clever, test_acc_interp_clever,
                                  train_acc_interp_ensemble, test_acc_interp_ensemble,
                                  test_acc_connect=test_acc_connect)
        plt.savefig(
            f"{logger.exp_dir}/data/{config.data}_{config.model}_weight_matching_{config.model_a}_v{config.a_v}_{config.model_b}_v{config.b_v}_{config.seed}_acc.png",
            dpi=300)
        fig_loss = plot_interp_loss(lambdas,
                                    train_loss_interp_naive, test_loss_interp_naive,
                                    train_loss_interp_clever, test_loss_interp_clever,
                                    train_loss_interp_ensemble, test_loss_interp_ensemble,
                                    test_loss_connect=test_loss_connect)
        plt.savefig(
            f"{logger.exp_dir}/data/{config.data}_{config.model}_weight_matching_{config.model_a}_v{config.a_v}_{config.model_b}_v{config.b_v}_{config.seed}_loss.png",
            dpi=300)
        wandb.log({
            "log_table": wandb.Table(dataframe=df),
            # "losslandscape_table": wandb.Table(dataframe=loss_landscape_df),
            "interp_acc_fig": wandb.Image(fig_acc),
            "interp_loss_fig": wandb.Image(fig_loss),
            # "init_loss_landscape": wandb.Image(fig_init_loss_landscape),
            # "final_loss_landscape": wandb.Image(fig_final_loss_landscape),
        })
        # Save final_permutation as an Artifact
        artifact = wandb.Artifact(f'{config.data}_{config.model}_weight_matching',
                                  type="permutation",
                                  metadata={
                                      "dataset": f'{config.data}',
                                      "model": f'{config.model}',
                                      "analysis": 'weight-matching'
                                  })
        filename = f'{logger.exp_dir}/data/best_perm.pt'
        torch.save(final_permutation, filename)
        if config.log_up_model:
            artifact.add_file(filename)
        else:
            artifact.add_reference(uri='file://' + os.path.abspath(filename))

        filename = f'{logger.exp_dir}/data/best_permed_weight.pt'
        torch.save(updated_params, filename)
        if config.log_up_model:
            artifact.add_file(filename)
        else:
            artifact.add_reference(uri='file://' + os.path.abspath(filename))
        wandb_run.log_artifact(artifact)


if __name__ == "__main__":
    main()
