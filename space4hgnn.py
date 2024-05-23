import os
import time
import argparse
import numpy as np
from pandas import DataFrame
from space4hgnn.utils import read_config
from openhgnn.utils import set_random_seed, Logger
from openhgnn.trainerflow import build_flow
from pynvml import *
from openhgnn.tasks import build_task

#dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 10099 and 10098 instead.
def Space4HGNN(args):
    metric_list = []
    epoches = []
    start = time.time()

    flow = build_flow(args, args.task)
    temp = flow.train()
    metric = temp['metric']
    epoch = temp['epoch']
    print(metric)
    #'HGBl-amazon', 'HGBl-LastFM',  return valid
    # 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB' return test
    # metric['valid']['loss'] = metric['valid']['loss'].item()
    for _, value in metric.items():
        value['loss'] = value['loss'].item()
    print(metric)
    metric_list.append(metric)
    epoches.append(epoch)
    # print("metric_list",metric_list)#[{'valid': {'roc_auc': 0.9670163799743547, 'loss': tensor(0.5650, device='cuda:0')}}]
    out_dict = {}
    for metrics in metric_list:
        for mode, metric in metrics.items():
            for m, score in metric.items():
                if out_dict.get(f"{mode}_{m}", None) is None:
                    out_dict[f"{mode}_{m}"] = []
                out_dict[f"{mode}_{m}"].append(score)

    end = time.time()
    mean_dict = {k + 'mean': np.mean(v) for k, v in out_dict.items()}
    std_dict = {k + 'std': np.std(v) for k, v in out_dict.items()}
    para = sum(p.numel() for p in flow.model.parameters())
    result = {
        'key': [args.key],
        'value': [args.value],
        'dataset': [args.dataset],
        'model_family': [args.model_family],
        'gnn_type': [args.gnn_type],
        'times': [args.times],
        'hidden_dim': [args.hidden_dim],
        'layers_pre_mp': [args.layers_pre_mp],
        'layers_post_mp': [args.layers_post_mp],
        'layers_gnn': [args.layers_gnn],
        'stage_type': [args.stage_type],
        'activation': [args.activation],
        'has_bn': [args.has_bn],
        'has_l2norm': [args.has_l2norm],
        'mini_batch_flag': [args.mini_batch_flag],
        'macro_func': [args.macro_func],
        'dropout': [args.dropout],
        'lr': [args.lr],
        'num_heads': [args.num_heads],
        'weight_decay': [args.weight_decay],
        'patience': [args.patience],
        'max_epoch': [args.max_epoch],
        'feat': [args.feat],
        'optimizer': [args.optimizer],
        'loss_fn': [args.loss_fn],
        'parameter': [para],
        'epoch': [np.mean(epoches)],
        'time': [end - start],
    }
    result.update(mean_dict)
    result.update(std_dict)
    df = DataFrame(result)
    # print(df)
    path = 'space4hgnn/prediction/excel/{}/{}_{}'.format(args.predictfile, args.key, args.value)
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv('{}/{}_{}_{}_{}.csv'.format(path, args.model_family, args.gnn_type, args.times, args.dataset))


if __name__ == '__main__':
    space_parser = argparse.ArgumentParser()
    space_parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
    space_parser.add_argument('--subgraph_extraction', '-u', default='metapath', type=str,
                              help='subgraph_extraction of models')
    space_parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    space_parser.add_argument('--dataset', '-d', default='HGBl-LastFM', type=str, help='name of datasets')
    space_parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    space_parser.add_argument('--repeat', '-r', default='1', type=int, help='-1 means cpu')
    space_parser.add_argument('--gnn_type', '-a', default='sageconv', type=str, help='aggregation type')
    space_parser.add_argument('--times', '-s', default=1, type=int, help='which yaml file')
    # it seems that the times arg control the configure of the model
    space_parser.add_argument('--key', '-k', default='has_bn', type=str, help='attribute')
    space_parser.add_argument('--value', '-v', default='True', type=str, help='value')
    space_parser.add_argument('--configfile', '-c', default='test', type=str,
                              help='The file path to load the configuration.')
    # is seems that the configfile arg control the really path to the yaml file
    space_parser.add_argument('--predictfile', '-p', default='predict', type=str,
                              help='The file path to store predict files.')
    space_args = space_parser.parse_args()

    args = read_config(space_args)

    args.seed = 1
    set_random_seed(args.seed)
    path = './space4hgnn/prediction/txt/{}/{}_{}/{}_{}_{}'.format(args.predictfile, args.key, args.value,
                                                                  args.model_family, args.gnn_type, args.times)
    if not os.path.exists(path):
        os.makedirs(path)
    args.HGB_results_path = '{}/{}_{}.txt'.format(path, args.dataset[5:], str(1))
    print(args)
    args.use_same_standard = False
    # from here the argument of args are complete.
    # args.task = 'node_classification'
    # args.dataset = 'HGBl-DBLP'
    # args.dataset = 'HGBl-amazon'

    args.Ismetatrain = "d"
    args.logger = Logger(args)
    for i in range(3):
        print("this is",i)
        Space4HGNN(args=space_args)


