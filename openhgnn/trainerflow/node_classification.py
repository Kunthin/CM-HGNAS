import time
from collections import defaultdict
import random
import dgl
import torch
from tqdm import tqdm
from ..utils.sampler import get_node_data_loader
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
import numpy as np


@register_flow("node_classification")
class NodeClassification(BaseFlow):
    r"""
    Node classification flow,
    The task is to classify the nodes of target nodes.
    Note: If the output dim is not equal the number of classes, we will modify the output dim with the number of classes.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        category: str
            The target node type to predict
        num_classes: int
            The number of classes for category node type

        """

        super(NodeClassification, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.pred_idx = getattr(self.task.dataset, 'pred_idx', None)

        self.labels = self.task.get_labels().to(self.device)
        self.num_nodes_dict = {ntype: self.hg.num_nodes(ntype) for ntype in self.hg.ntypes}
        self.to_homo_flag = getattr(self.model, 'to_homo_flag', False)

        if self.to_homo_flag:
            self.g = dgl.to_homogeneous(self.hg)

        # this place is for the data sample.
        if self.args.use_same_standard:
            from graphnas.same_standard.dill_with_graph import sample_subgraph
            if self.args.dataset_name == 'HGBn-DBLP':
                test_rate = 0.9
                num_nodes = self.hg.number_of_nodes('paper')
                n_test = int(num_nodes * test_rate)
                n_train = num_nodes - n_test
                train, test = torch.utils.data.random_split(range(num_nodes), [n_train, n_test])
                train_idx = torch.tensor(train.indices).long()
                self.test_idx = torch.tensor(test.indices).long()[:1000]
                random_int = torch.randperm(len(train_idx))
                self.valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                self.train_idx = train_idx[random_int[len(train_idx) // 5:]]
                self.category = 'paper'
            self.all_target_node_index, self.all_pos_node_index, self.all_neg_node_index, self.all_neibor_node_idx, self.remove_list = sample_subgraph(
                self.hg,
                self.args.sample_batch_size,
                self.args.dataset_name,
                self.train_idx,
                self.val_idx,
                self.test_idx,
                self.category)
        if self.args.mini_batch_flag:
            self.fanouts = [args.fanout] * self.args.num_layers
            sampler = dgl.dataloading.MultiLayerNeighborSampler(self.fanouts)
            use_uva = self.args.use_uva

            if self.to_homo_flag:
                loader_g = self.g
            else:
                loader_g = self.hg

            if self.train_idx is not None:
                if self.to_homo_flag:
                    loader_train_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                   {self.category: self.train_idx}).to(self.device)
                else:
                    loader_train_idx = {self.category: self.train_idx.to(self.device)}

                self.train_loader = dgl.dataloading.DataLoader(loader_g, loader_train_idx, sampler,
                                                               batch_size=self.args.batch_size, device=self.device,
                                                               shuffle=True, use_uva=use_uva)
            if self.train_idx is not None:
                if self.to_homo_flag:
                    loader_val_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict, {self.category: self.val_idx}).to(
                        self.device)
                else:
                    loader_val_idx = {self.category: self.val_idx.to(self.device)}
                self.val_loader = dgl.dataloading.DataLoader(loader_g, loader_val_idx, sampler,
                                                             batch_size=self.args.batch_size, device=self.device,
                                                             shuffle=True, use_uva=use_uva)
            if self.args.test_flag:
                if self.test_idx is not None:
                    if self.to_homo_flag:
                        loader_test_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                      {self.category: self.test_idx}).to(self.device)
                    else:
                        loader_test_idx = {self.category: self.test_idx.to(self.device)}
                    self.test_loader = dgl.dataloading.DataLoader(loader_g, loader_test_idx, sampler,
                                                                  batch_size=self.args.batch_size, device=self.device,
                                                                  shuffle=True, use_uva=use_uva)
            if self.args.prediction_flag:
                if self.pred_idx is not None:
                    if self.to_homo_flag:
                        loader_pred_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                      {self.category: self.pred_idx}).to(self.device)
                    else:
                        loader_pred_idx = {self.category: self.pred_idx.to(self.device)}
                    self.pred_loader = dgl.dataloading.DataLoader(loader_g, loader_pred_idx, sampler,
                                                                  batch_size=self.args.batch_size, device=self.device,
                                                                  shuffle=True, use_uva=use_uva)

    def preprocess(self):
        r"""
        Preprocess for different models, e.g.: different optimizer for GTN.
        And prepare the dataloader foe train validation and test.
        Last, we will call preprocess_feature.
        """
        if self.args.model == 'GTN':
            if hasattr(self.args, 'adaptive_lr_flag') and self.args.adaptive_lr_flag == True:
                self.optimizer = torch.optim.Adam([{'params': self.model.gcn.parameters()},
                                                   {'params': self.model.linear1.parameters()},
                                                   {'params': self.model.linear2.parameters()},
                                                   {"params": self.model.layers.parameters(), "lr": 0.5}
                                                   ], lr=0.005, weight_decay=0.001)
            else:
                # self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
                pass
        elif self.args.model == 'MHNF':
            if hasattr(self.args, 'adaptive_lr_flag') and self.args.adaptive_lr_flag == True:
                self.optimizer = torch.optim.Adam([{'params': self.model.HSAF.HLHIA_layer.gcn_list.parameters()},
                                                   {'params': self.model.HSAF.channel_attention.parameters()},
                                                   {'params': self.model.HSAF.layers_attention.parameters()},
                                                   {'params': self.model.linear.parameters()},
                                                   {"params": self.model.HSAF.HLHIA_layer.layers.parameters(),
                                                    "lr": 0.5}
                                                   ], lr=0.005, weight_decay=0.001)

            else:
                # self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
                pass
        elif self.args.model == 'RHGNN':
            print(f'get node data loader...')
            self.train_loader, self.val_loader, self.test_loader = get_node_data_loader(
                self.args.node_neighbors_min_num,
                self.args.num_layers,
                self.hg.to('cpu'),
                batch_size=self.args.batch_size,
                sampled_node_type=self.category,
                train_idx=self.train_idx,
                valid_idx=self.val_idx,
                test_idx=self.test_idx)

        super(NodeClassification, self).preprocess()  # call the base_flow preprocess functhion,
        # and the following are useless maybe?

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            elif self.args.use_same_standard:
                train_loss, ndcgs = self._same_standard_train_step()
            else:
                # the train_loss is from this function
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:  # true
                    # the test mode is add in
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                elif self.args.use_same_standard:
                    metric_dict, losses = self._same_standard_val_step(modes=['valid'])
                    metric_dict['train'] = {'ndcgs': ndcgs}
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                # every epoch output is in this line
                # if self.args.Ismetatrain != "meta_test":
                #     self.logger.train_info(
                #         f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. " +
                #         self.logger.metric2str(metric_dict))
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break
        # the train epoch is over.
        stopper.load_model(self.model)
        if self.args.prediction_flag:  # false here
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts

        if self.args.test_flag:
            if self.args.dataset[:4] == 'HGBn' or self.args.dataset == 'yelp4HeGAN':
                # save results for HGBn
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, val_loss = self._mini_test_step(modes=['valid'])
                elif self.args.use_same_standard:
                    metric_dict, val_loss = self._same_standard_test_step(modes=['test'])
                else:
                    metric_dict, val_loss = self._full_test_step(modes=['valid'])
                self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
                if not self.args.use_same_standard:
                    self.model.eval()
                    with torch.no_grad():
                        h_dict = self.model.input_feature()
                        logits = self.model(self.hg, h_dict)[self.category]
                        self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
                # else:
                #     test_metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
                #     self.logger.train_info('[Test Info]' + self.logger.metric2str(test_metric_dict))
                return dict(metric=metric_dict, epoch=epoch)
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                metric_dict, _ = self._mini_test_step(modes=['valid', 'test'])
            else:
                metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
        # this h_dict is the node feature. num_n_type*64
        h_dict = self.model.input_feature()
        # here the self.hg contain the information of the graph
        self.hg = self.hg.to("cuda:0")
        # here the logits return the nodetype that I want.
        logits = self.model(self.hg, h_dict)[self.category]
        # this is where I need to make some change.
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self):
        self.model.train()

        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
            if self.to_homo_flag:
                # input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                seeds = to_hetero_idx(self.g, self.hg, seeds)
            elif isinstance(input_nodes, dict):
                for key in input_nodes:
                    input_nodes[key] = input_nodes[key].to(self.device)
            # elif not isinstance(input_nodes, dict):
            #     input_nodes = {self.category: input_nodes}
            emb = self.model.input_feature.forward_nodes(input_nodes)
            # if self.to_homo_flag:
            #     emb = to_homo_feature(self.hg.ntypes, emb)
            lbl = self.labels[seeds[self.category]].to(self.device)
            logits = self.model(blocks, emb)[self.category]
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)

    def _same_standard_train_step(self):
        self.model.train()
        # print(self.model.parameters())
        # this is the data that we need to train.
        # this h_dict is the node feature. num_n_type*64
        # there is nothing wrong.
        h_dict = self.model.input_feature()
        # here the self.hg contain the information of the graph
        self.hg = self.hg.to("cuda:0")
        # here the logits return the nodetype that I want.
        node_emd = self.model(self.hg, h_dict)
        # this is where I need to make some change.
        # loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        from graphnas.same_standard.dill_with_graph import node_loss, scheme_loss
        n_loss, ress = node_loss(self.all_target_node_index['train'], self.all_pos_node_index['train'],
                                 self.all_neg_node_index['train'], node_emd,
                                 self.args.dataset_name, self.model.link_relation_dict)

        # network schema loss
        s_loss = scheme_loss(self.all_neibor_node_idx['train'], self.remove_list['train'], node_emd, self.train_idx,
                             self.category, self.model.node_encoder)
        loss = n_loss + (0.5 * s_loss)
        ndcgs = []
        from graphnas.same_standard.dill_with_graph import ndcg_at_k
        for i in ress:
            ai = np.zeros(len(i[0]))
            ai[0] = 1
            # this argsort sort the array and return the index
            # i.argsort(descending=True)
            ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending=True)]
        ndcgs = np.average(ndcgs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), ndcgs

    def _same_standard_val_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        loss: dict[str, float]
            the loss item
        """
        from graphnas.same_standard.dill_with_graph import ndcg_at_k
        self.model.eval()
        metric_dict = {}
        loss_dict = {}
        for mode in modes:
            with torch.no_grad():
                h_dict = self.model.input_feature()
                # here the self.hg contain the information of the graph
                self.hg = self.hg.to("cuda:0")
                # here the logits return the nodetype that I want.
                node_emd = self.model(self.hg, h_dict)
                # this is where I need to make some change.
                # loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
                from graphnas.same_standard.dill_with_graph import node_loss, scheme_loss
                n_loss, ress = node_loss(self.all_target_node_index['val'], self.all_pos_node_index['val'],
                                         self.all_neg_node_index['val'], node_emd,
                                         self.args.dataset_name, self.model.link_relation_dict)
                # network schema loss
                s_loss = scheme_loss(self.all_neibor_node_idx['val'], self.remove_list['val'], node_emd, self.val_idx,
                                     self.category, self.model.node_encoder)
                loss = n_loss + (0.5 * s_loss)
                ndcgs = []
                for i in ress:
                    ai = np.zeros(len(i[0]))
                    ai[0] = 1
                    # this argsort sort the array and return the index
                    # i.argsort(descending=True)
                    ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending=True)]
                ndcgs = np.average(ndcgs)
                metric_dict = {mode: {'ndcgs': ndcgs}}
                loss_dict = {mode: loss.item()}
        return metric_dict, loss_dict

    def _same_standard_test_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        loss: dict[str, float]
            the loss item
        """
        from graphnas.same_standard.dill_with_graph import ndcg_at_k
        self.model.eval()
        metric_dict = {}
        loss_dict = {}
        for mode in modes:
            with torch.no_grad():
                h_dict = self.model.input_feature()
                # here the self.hg contain the information of the graph
                self.hg = self.hg.to("cuda:0")
                # here the logits return the nodetype that I want.
                node_emd = self.model(self.hg, h_dict)
                # this is where I need to make some change.
                # loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
                from graphnas.same_standard.dill_with_graph import node_loss, scheme_loss
                n_loss, ress = node_loss(self.all_target_node_index['test'], self.all_pos_node_index['test'],
                                         self.all_neg_node_index['test'], node_emd,
                                         self.args.dataset_name, self.model.link_relation_dict)
                # network schema loss
                s_loss = scheme_loss(self.all_neibor_node_idx['test'], self.remove_list['test'], node_emd,
                                     self.test_idx,
                                     self.category, self.model.node_encoder)
                loss = n_loss + (0.5 * s_loss)
                ndcgs = []
                for i in ress:
                    ai = np.zeros(len(i[0]))
                    ai[0] = 1
                    # this argsort sort the array and return the index
                    # i.argsort(descending=True)
                    ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending=True)]
                ndcgs = np.average(ndcgs)
                metric_dict = {mode: {'ndcgs': ndcgs}}
                loss_dict = {mode: loss.item()}
        return metric_dict, loss_dict

    def _full_test_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.
        logits: dict[str, th.Tensor]
            given logits, default `None`.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model.my_forward(self.hg, h_dict)[self.category]
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.val_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []
                for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                    if self.to_homo_flag:
                        # input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                        seeds = to_hetero_idx(self.g, self.hg, seeds)
                    elif not isinstance(input_nodes, dict):
                        input_nodes = {self.category: input_nodes}
                    emb = self.model.input_feature.forward_nodes(input_nodes)
                    # if self.to_homo_flag:
                    #     emb = to_homo_feature(self.hg.ntypes, emb)
                    lbl = self.labels[seeds[self.category]].to(self.device)
                    logits = self.model(blocks, emb)[self.category]
                    loss = self.loss_fn(logits, lbl)
                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss
        return metric_dict, loss_dict

    def _full_prediction_step(self):
        """

        Returns
        -------
        """
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.model(self.hg, h_dict)[self.category]
            return logits

    def _mini_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            loader_tqdm = tqdm(self.pred_loader, ncols=120)
            indices = []
            y_predicts = []
            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if self.to_homo_flag:
                    input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                    seeds = to_hetero_idx(self.g, self.hg, seeds)
                elif not isinstance(input_nodes, dict):
                    input_nodes = {self.category: input_nodes}
                emb = self.model.input_feature.forward_nodes(input_nodes)
                if self.to_homo_flag:
                    emb = to_homo_feature(self.hg.ntypes, emb)
                logits = self.model(blocks, emb)[self.category]
                seeds = seeds[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts
