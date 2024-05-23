import os

import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath = os.path.split(rootPath)[0]
sys.path.insert(0,rootPath)
import argparse

import torch

from progressbar import *
import graphnas.meta_trainer as trainer
import graphnas.utils.tensor_utils as utils


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()
    print(type(args.derive_meta_test))
    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)
    # controller
    parser.add_argument('--derive_meta_test', type=bool, default=False)
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="HGBn-ACM", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    # maml parameters
    parser.add_argument('--adapt_lr', type=float, default=3.5e-4)
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument('--adapt_steps', type=int, default=5)
    parser.add_argument('--adapt_meta_test_steps', type=int, default=100)

    # predictor
    parser.add_argument('--predict', type=bool, default=False)
    parser.add_argument('--T', type=int, default=25,
                        help="before use predictor, the epochs of controller train and update")
    parser.add_argument('--threshold_kl', type=float, default=0.000001,
                        help="standard controller use predictor")

    # The following lines are the config of same_standard
    parser.add_argument('--use_same_standard', type=bool, default=True,
                        help='use same standard for loss or not')
    parser.add_argument('--sample_batch_size', type=int, default=250,
                        help='the size of the subgraph target nodes number.')
    parser.add_argument('--graph_batch_size', type=int, default=32,
                        help='input batch size for parent tasks (default: 64)')
    parser.add_argument('--node_batch_size', type=int, default=1,
                        help='input batch size for parent tasks (default: 3)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading;default:4')
    # meta-learning settings
    parser.add_argument('--order', type=int, default=2, help='gradient order')
    parser.add_argument('--node_level', type=int, default=1, help='node-level adaptation')
    parser.add_argument('--graph_level', type=int, default=1, help='graph-level adaptation')
    parser.add_argument('--node_lr', type=float, default=0.001, help='learning rate for node-level adaptation')
    parser.add_argument('--node_update', type=int, default=1, help='update step for node-level adaptation')
    parser.add_argument('--graph_lr', type=float, default=0.001, help='learning rate for graph-level adaptation')
    parser.add_argument('--graph_update', type=int, default=1, help='update step for graph-level adaptation')
    parser.add_argument('--support_set_size', type=int, default=10, help='size of support set')
    parser.add_argument('--query_set_size', type=int, default=5, help='size of query set')


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    # args.max_epoch = 1
    # args.controller_max_step = 1
    # args.derive_num_sample = 1
    args.max_epoch = 10  # 10
    args.epochs = 100  # 100
    # args.controller_max_step = 1 #??
    args.derive_num_sample = 10  # 10
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)
    # args.derive_meta_test = True
    # args.use_same_standard = False
    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args = build_args()
    main(args)
