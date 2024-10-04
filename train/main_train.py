import os

import configargparse
import wandb
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from utils.training import train, test, train_sam, double_test, train_label_smooth
from utils.misc import directbool, set_seed, none_or_str
from utils.selector import get_data, get_model, get_opt, get_scheduler
from utils.logger import Logger
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''


def get_args():
    # basic
    parser = configargparse.ArgumentParser(default_config_files=['./conf/train_mlp_mnist.yaml'])
    parser.add('-c', '--conf', required=True, is_config_file=True, help='config file path')
    parser.add_argument('--dir', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_wandb', type=str, default='offline', choices=['online', 'offline'])
    parser.add_argument('--log_up_model', type=directbool, default=False, choices=[True, False], help='True: updaload model to wandb, False:save local with refefence')
    parser.add_argument('--log_interval', type=int, default=10)
    # data
    parser.add_argument("--data", type=str, default='mnist', choices=['mnist', 'fmnist', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                                                      'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                                                      'rotated_mnist_75', 'rotated_mnist_90',
                                                                      'mnist_fmnist', 'usps_mnist',
                                                                      'mnist_rotated_mnist_15',
                                                                      'mnist_rotated_mnist_30',
                                                                      'mnist_rotated_mnist_45',
                                                                      'mnist_rotated_mnist_60',
                                                                      'mnist_rotated_mnist_75',
                                                                      'mnist_rotated_mnist_90'
                                                                      ])
    parser.add_argument("--eval_data_a", type=str, default=None,
                        choices=['mnist', 'fmnist', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                 'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90'])
    parser.add_argument("--eval_data_b", type=str, default=None,
                        choices=['mnist', 'fmnist', 'usps', 'cifar10', 'cifar10_split_a', 'cifar10_split_b',
                                 'rotated_mnist_15', 'rotated_mnist_30', 'rotated_mnist_45', 'rotated_mnist_60',
                                 'rotated_mnist_75', 'rotated_mnist_90'])

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--length", type=int, default=None, help='data length')
    parser.add_argument("--trans_type", type=none_or_str, default=None, choices=[None, 'color_32'])
    # model
    parser.add_argument("--model", type=str, default='mlp', choices=['mlp', 'vgg11', 'resnet20'])
    parser.add_argument("--width_multiplier", type=int, default=1)
    parser.add_argument("--bias", type=directbool, default=True, choices=[True, False])
    parser.add_argument('--amp', type=directbool, default=False, help='amp mode')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'clip'])
    parser.add_argument("--clip_head_dim", type=int, default=512)
    # train
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    # opt
    parser.add_argument("--opt", type=str, default='adam', choices=['sgd', 'adam', 'sam'])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_schedule", type=none_or_str, default=None, choices=[None, 'warmup_cosine_decay', 'multi_step_lr', 'cosine', 'linear_up_down'])
    parser.add_argument("--warmup_epoch", type=int, default=0)
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
            name=f'train_{args.data}_{args.model}_{args.opt}_{args.lr}_{args.seed}',
            tags=[f"{args.data}", f"{args.model}", "train", f"{args.loss}"],
            job_type="train",
            mode=args.log_wandb,
            sync_tensorboard=True,
    ) as wandb_run:
        log_dir = os.path.join(logger.exp_dir, wandb_run.name, "_".join(wandb_run.tags))
        writer = SummaryWriter(log_dir=log_dir)
        config = wandb.config
        for key in args.__dict__:
            config[key] = args.__dict__[key]
            print(f'{key}: {config[key]}')
        config.device = device
        artifact = wandb.Artifact(f"{config.data}-{config.model}-weights", type="model-weights")

        train_loader, test_loader = get_data(config.data, config.batch_size, config.length, config.trans_type)
        if config.loss == 'clip':
            clip = True
            num_head = config.clip_head_dim
        else:
            clip = False
            num_head = config.num_classes
        model = get_model(config.model, num_head, config.data, config.trans_type, config.width_multiplier, config.bias, clip, train_loader.dataset.classes, device)
        print(model)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        opt = get_opt(model, config.opt, config.lr, config.weight_decay)
        scheduler = get_scheduler(config.lr_schedule, opt, config.epoch, config.warmup_epoch)
        print(f'opt: {opt}')
        test_best_acc = 0
        test_best_loss = 999
        best_model = deepcopy(model)
        best_grad_dict = []
        try:
            for epoch in range(1, args.epoch + 1):
                test_loss = None
                test_acc = None
                if config.opt == 'sam':
                    train_loss, train_acc, grad_dict = train_sam(args, model, device, train_loader, opt, epoch, print_flg=True)
                else:
                    train_loss, train_acc, grad_dict = train(args, model, device, train_loader, opt, epoch, print_flg=True)
                    # train_loss, train_acc, grad_dict = train_label_smooth(args, model, device, train_loader, opt, epoch)
                if epoch % config.log_interval == 0:
                    if config.eval_data_a is None and config.eval_data_b is None:
                        test_loss, test_acc = test(model, device, test_loader, print_flg=True)
                    else:
                        _, test_loader_a = get_data(config.eval_data_a, config.batch_size, config.length, config.trans_type)
                        _, test_loader_b = get_data(config.eval_data_b, config.batch_size, config.length, config.trans_type)
                        test_loss, test_acc = double_test(model, device, test_loader_a, test_loader_b, print_flg=True)
                    if test_acc > test_best_acc:
                        test_best_acc = test_acc
                        best_model = deepcopy(model)
                        best_grad_dict = deepcopy(grad_dict)
                    if test_loss < test_best_loss:
                        test_best_loss = test_loss
                metric = {
                    "loss/train": train_loss,
                    "loss/test": test_loss,
                    "loss/test_best": test_best_loss,
                    "acc/train": train_acc,
                    "acc/test": test_acc,
                    "acc/test_best": test_best_acc,
                    "lr": opt.param_groups[0]['lr']
                }
                for key in metric:
                    if metric[key] is not None:
                        writer.add_scalar(key, metric[key], global_step=epoch)
                # No point saving the model at all if we're running in test mode.
                os.makedirs(f"{logger.exp_dir}/model", exist_ok=True)
                # if (epoch % 100 == 0 or epoch == config.epoch - 1):
                if config.epoch == epoch:  # save last & best
                    filename = f"{logger.exp_dir}/model/params_{epoch}.pt"
                    torch.save(model.module.state_dict(), filename)
                    if config.log_up_model:
                        artifact.add_file(filename)
                    else:
                        artifact.add_reference(uri='file://' + os.path.abspath(filename))
                    filename = f"{logger.exp_dir}/model/grads_{epoch}.pt"
                    torch.save(grad_dict, filename)
                    if config.log_up_model:
                        artifact.add_file(filename)
                    else:
                        artifact.add_reference(uri='file://' + os.path.abspath(filename))
                if scheduler is not None:
                    scheduler.step(epoch)
            filename = f"{logger.exp_dir}/model/params_best.pt"
            torch.save(best_model.module.state_dict(), filename)
            if config.log_up_model:
                artifact.add_file(filename)
            else:
                artifact.add_reference(uri='file://' + os.path.abspath(filename))
            filename = f"{logger.exp_dir}/model/grads_best.pt"
            torch.save(best_grad_dict, filename)
            if config.log_up_model:
                artifact.add_file(filename)
            else:
                artifact.add_reference(uri='file://' + os.path.abspath(filename))
            wandb_run.log_artifact(artifact)

        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print('torch ver', torch.__version__)
    print('torchvision ver', torchvision.__version__)
    main()
