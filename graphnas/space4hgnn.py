import os
import time
import argparse
import numpy as np
from pandas import DataFrame
from openhgnn.utils import set_random_seed
from openhgnn.trainerflow import build_flow


def Space4HGNN(args):
    metric_list = []
    epoches = []
    start = time.time()
    # this logger function is also very important. I must write something,it can logger
    # the INFO
    if args.use_same_standard:
        args.repeat = 3
    else:
        args.repeat = 1
    for i in range(args.repeat):
        args.seed = i
        set_random_seed(int(time.time()))
        path = './space4hgnn/prediction/txt/{}/{}_{}/{}_{}_{}'.format(args.predictfile, args.key, args.value,
                                                                      args.model_family, args.gnn_type, args.times)
        if not os.path.exists(path):
            os.makedirs(path)
        args.HGB_results_path = '{}/{}_{}.txt'.format(path, args.dataset[5:], str(i + 1))
        # here I need to rewrite something.
        flow = build_flow(args, args.task)
        temp = flow.train()
        metric = temp['metric']
        epoch = temp['epoch']
        # this line is only for link prediction
        # print("metric",metric)
        # 'HGBl-amazon'{'valid': {'roc_auc': 0.9670163799743547, 'loss': tensor(0.5650, device='cuda:0')}}
        # 'HGBl-LastFM' {'valid': {'roc_auc': 0.6444686051853101, 'loss': tensor(0.7239, device='cuda:0')}}
        # 'HGBl-ACM' {'test': {'roc_auc': 0.461083582319853, 'loss': tensor(0.6934, device='cuda:0')}}
        # 'HGBl-DBLP' {'test': {'roc_auc': 0.49158433277465546, 'loss': tensor(0.7238, device='cuda:0')}}
        # 'HGBl-IMDB' {'test': {'roc_auc': 0.5508626407883968, 'loss': tensor(0.7232, device='cuda:0')}}
        # if args.task == "link_prediction":
        #     for _,value in metric.items():
        #         value['loss'] = value['loss'].item()
        metric_list.append(metric)
        epoches.append(epoch)
    # print(metric_list)
    out_dict = {}
    for metrics in metric_list:
        for mode, metric in metrics.items():  # test or valid
            for m, score in metric.items():
                if out_dict.get(f"{mode}_{m}", None) is None:
                    out_dict[f"{mode}_{m}"] = []
                if m == "loss":
                    out_dict[f"{mode}_{m}"].append(score.item())  # .item()
                else:
                    out_dict[f"{mode}_{m}"].append(score)

    end = time.time()
    max_dict = {k + 'max': np.max(v) for k, v in out_dict.items()}
    mean_dict = {k + 'mean': np.mean(v) for k, v in out_dict.items()}
    std_dict = {k + 'std': np.std(v) for k, v in out_dict.items()}
    para = sum(p.numel() for p in flow.model.parameters())
    result = {
        'key': [args.key],
        'value': [args.value],
        'dataset': [args.dataset],
        'model_family': [args.model_family],
        'gnn_type': [args.gnn_type],
        # 'times': [args.times],
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
    result.update(max_dict)
    result.update(mean_dict)
    result.update(std_dict)
    args.logger.info(str(result))
    df = DataFrame(result)
    # print(df)
    path = 'space4hgnn/prediction/excel/{}/{}_{}'.format(args.predictfile, args.key, args.value)
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv('{}/{}_{}_{}.csv'.format(path, args.model_family, args.gnn_type, args.dataset))
    if args.use_same_standard:
        if args.task == "node_classification":
            reward = float(max_dict['test_ndcgsmax'])
        elif args.task == "link_prediction":
            if 'test_ndcgsmax' in max_dict.keys():
                reward = float(max_dict['test_ndcgsmax'])
            else:
                reward = float(max_dict['valid_ndcgsmax'])
        # reward = mean_dict['valid_ndcgsmean']
        else:
            reward = round(float(max_dict['max']), 4)
    else:
        if args.task == "node_classification":
            reward = float(max_dict['valid_Macro_f1max'])
        elif args.task == "link_prediction":
            if 'test_roc_aucmax' in max_dict.keys():
                reward = float(max_dict['test_roc_aucmax'])
            else:
                reward = float(max_dict['valid_roc_aucmax'])
        # reward = mean_dict['valid_Macro_f1mean']
        else:
            reward = round(float(max_dict['max']), 4)
    return reward


if __name__ == '__main__':
    space_parser = argparse.ArgumentParser()
    space_parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
    space_parser.add_argument('--subgraph_extraction', '-u', default='metapath', type=str,
                              help='subgraph_extraction of models')
    space_parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    space_parser.add_argument('--dataset', '-d', default='HGBl-PubMed', type=str, help='name of datasets')
    space_parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    space_parser.add_argument('--repeat', '-r', default='3', type=int, help='-1 means cpu')
    space_parser.add_argument('--gnn_type', '-a', default='gcnconv', type=str, help='aggregation type')
    space_parser.add_argument('--times', '-s', default=1, type=int, help='which yaml file')
    # it seems that the times arg control the configure of the model
    space_parser.add_argument('--key', '-k', default='has_bn', type=str, help='attribute')
    space_parser.add_argument('--value', '-v', default='True', type=str, help='value')
    space_parser.add_argument('--configfile', '-c', default='config', type=str,
                              help='The file path to load the configuration.')
    # is seems that the configfile arg control the really path to the yaml file
    space_parser.add_argument('--predictfile', '-p', default='predict', type=str,
                              help='The file path to store predict files.')
    space_args = space_parser.parse_args()

    # space_args = read_config(space_args)

    Space4HGNN(args=space_args)
