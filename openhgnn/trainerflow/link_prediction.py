import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping, add_reverse_edges, get_ntypes_from_canonical_etypes
import numpy as np


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """
    Link Prediction trainer flows.
    Here is a tutorial teach you how to train a GNN for
    `link prediction <https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html>_`.

    When training, you will need to remove the edges in the test set from the original graph.
    DGL recommends you to treat the pairs of nodes as another graph, since you can describe a pair of nodes with an edge.
    In link prediction, you will have a positive graph consisting of all the positive examples as edges,
    and a negative graph consisting of all the negative examples.
    The positive graph and the negative graph will contain the same set of nodes as the original graph.
    This makes it easier to pass node features among multiple graphs for computation.
    As you will see later, you can directly feed the node representations computed on the entire graph to the positive
    and the negative graphs for computing pair-wise scores.
    """

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        ------------
        target_link: list
            list of edge types which are target link type to be predicted

        score_fn: str
            score function used in calculating the scores of links, supported function: distmult[Default if not specified] & dot product

        r_embedding: nn. ParameterDict
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            The dimension of embedding is `self.args.hidden_dim`.

        """
        super(LinkPrediction, self).__init__(args)
        # 'HGBl-amazon', 'HGBl-LastFM','HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB' this five dataset
        # the target_link return
        # [('user', 'user-artist', 'artist')] predict the link
        self.target_link = self.task.dataset.target_link
        # return the two point type of the link. like, user,artist.
        self.args.out_node_type = self.task.get_out_ntype()
        self.train_hg = self.task.get_train()
        if hasattr(self.args, 'flag_add_reverse_edges') \
                or self.args.dataset in ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2']:
            self.train_hg = add_reverse_edges(self.train_hg)
        if not hasattr(self.args, 'out_dim'):
            self.args.out_dim = self.args.hidden_dim

        self.model = build_model(self.model).build_model_from_args(self.args, self.train_hg).to("cuda:0")

        if not hasattr(self.args, 'score_fn'):
            self.args.score_fn = 'distmult'
        if self.args.score_fn == 'distmult':
            """
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            """
            self.r_embedding = nn.ParameterDict({etype[1]: nn.Parameter(th.Tensor(1, self.args.out_dim))
                                                 for etype in self.train_hg.canonical_etypes}).to(self.device)
            for _, para in self.r_embedding.items():
                nn.init.xavier_uniform_(para)
        else:
            self.r_embedding = None

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        if self.args.score_fn == 'distmult':
            self.optimizer.add_param_group({'params': self.r_embedding.parameters()})
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link).to(self.device)
        # this place is for the data sample.

        if self.args.mini_batch_flag:
            self.fanouts = [-1] * self.args.num_layers
            train_eid_dict = {
                etype: self.train_hg.edges(etype=etype, form='eid')
                for etype in self.target_link}
            sampler = dgl.dataloading.NeighborSampler(self.fanouts)
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(2)
            sampler = dgl.dataloading.as_edge_prediction_sampler(sampler=sampler, negative_sampler=negative_sampler)
            self.dataloader = dgl.dataloading.DataLoader(
                self.train_hg, train_eid_dict, sampler,
                batch_size=self.args.batch_size,
                shuffle=True)
            self.category = self.hg.ntypes[0]
        if self.args.use_same_standard:
            if self.args.dataset_name == 'HGBl-LastFM':
                from graphnas.same_standard.dill_with_graph import HGBl_LastFM_sample_subgraph
                self.all_target_node_index, self.all_pos_node_index, self.all_neg_node_index, self.all_neibor_node_idx, self.really_list = HGBl_LastFM_sample_subgraph(
                    self.hg)
            if self.args.dataset_name == 'HGBl-amazon':
                from graphnas.same_standard.dill_with_graph import HGBl_amazon_sample_subgraph
                self.all_target_node_index, self.all_pos_node_index, self.all_neg_node_index, self.all_neibor_node_idx, self.really_list = HGBl_amazon_sample_subgraph(
                    self.hg)

    def preprocess(self):
        """
        In link prediction, you will have a positive graph consisting of all the positive examples as edges,
        and a negative graph consisting of all the negative examples.
        The positive graph and the negative graph will contain the same set of nodes as the original graph.
        """
        super(LinkPrediction, self).preprocess()
        # to('cpu') & to('self.device')
        self.train_hg = self.train_hg.to("cuda:0")

    def train(self):
        # preprocess
        self.preprocess()
        stopper = EarlyStopping(self.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch), colour="blue"):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            elif self.args.use_same_standard:
                loss, ndcgs = self._same_standard_train_step()
            else:
                # they all use the _full_train_step()
                loss = self._full_train_setp()
            if epoch % self.evaluate_interval == 0:
                if self.args.use_same_standard:
                    val_metric = self._same_standard_val_step(modes=['valid'])
                    val_metric['train'] = {'ndcgs': ndcgs}
                else:
                    val_metric = self._test_step('valid')
                if self.args.Ismetatrain != "meta_test":
                    self.logger.train_info(
                        f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " + self.logger.metric2str(val_metric))
                early_stop = stopper.loss_step(val_metric['valid']['loss'], self.model)
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break
        stopper.load_model(self.model)
        # Test
        if self.args.test_flag:  # True
            if self.args.dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
                # Test in HGB datasets.
                self.model.eval()
                with torch.no_grad():
                    if self.args.use_same_standard:
                        val_metric = self._same_standard_test_step(modes=['test'])
                    else:
                        if self.args.dataset in ['HGBl-amazon']:
                            val_metric = self._test_step('valid')
                        else:
                            val_metric = self._test_step('test')
                        h_dict = self.model.input_feature()
                        embedding = self.model(self.train_hg, h_dict)
                        score = th.sigmoid(self.task.ScorePredictor(self.task.test_hg, embedding, self.r_embedding))
                        self.task.dataset.save_results(hg=self.task.test_hg, score=score,
                                                       file_path=self.args.HGB_results_path)
                    self.logger.train_info('[Test Info]' + self.logger.metric2str(val_metric))

                return dict(metric=val_metric, epoch=epoch)
            else:
                test_score = self._test_step(split="test")
                self.logger.train_info(self.logger.metric2str(test_score))
                return dict(metric=test_score, epoch=epoch)
        elif self.args.prediction_flag:  # False
            if self.args.mini_batch_flag:
                prediction_res = self._mini_prediction_step()
            else:
                prediction_res = self._full_prediction_step()
            return prediction_res

    def construct_negative_graph(self, hg):
        e_dict = {
            etype: hg.edges(etype=etype, form='eid')
            for etype in hg.canonical_etypes}
        neg_srcdst = self.negative_sampler(hg, e_dict)
        neg_pair_graph = dgl.heterograph(neg_srcdst,
                                         {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes})
        return neg_pair_graph

    def _same_standard_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        # here the self.hg contain the information of the graph
        self.hg = self.hg.to("cuda:0")
        node_emd = self.model(self.hg, h_dict)
        # if self.args.dataset_name == 'HGBl-LastFM':
        from graphnas.same_standard.dill_with_graph import link_node_loss, link_scheme_loss
        n_loss, ress = link_node_loss(self.all_target_node_index['train'], self.all_pos_node_index['train'],
                                      self.all_neg_node_index['train'], node_emd,
                                      self.model.link_relation_dict, self.args.dataset_name)

        # network schema loss
        s_loss = link_scheme_loss(self.all_target_node_index['train'], self.all_neibor_node_idx['train'],
                                  self.really_list['train'], node_emd, self.model.node_encoder, self.args.dataset_name)
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

    def _full_train_setp(self):
        self.model.train()
        h_dict = self.model.input_feature()
        # the embedding return num_node*out_dim 64 or 128
        embedding = self.model(self.train_hg, h_dict)
        # construct a negative graph according to the positive graph in each training epoch.
        negative_graph = self.task.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)
        # negative_graph = self.construct_negative_graph(self.train_hg)
        # loss = self.loss_calculation(self.train_hg, negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
                from graphnas.same_standard.dill_with_graph import link_node_loss, link_scheme_loss
                n_loss, ress = link_node_loss(self.all_target_node_index['val'], self.all_pos_node_index['val'],
                                              self.all_neg_node_index['val'], node_emd,
                                              self.model.link_relation_dict, self.args.dataset_name)

                # network schema loss
                s_loss = link_scheme_loss(self.all_target_node_index['val'], self.all_neibor_node_idx['val'],
                                          self.really_list['val'], node_emd, self.model.node_encoder,
                                          self.args.dataset_name)
                loss = n_loss + (0.5 * s_loss)
                ndcgs = []
                for i in ress:
                    ai = np.zeros(len(i[0]))
                    ai[0] = 1
                    ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending=True)]
                ndcgs = np.average(ndcgs)
                metric_dict = {mode: {'ndcgs': ndcgs, 'loss': loss.item()}}
                # loss_dict = {mode: loss.item()}
        return metric_dict

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
                from graphnas.same_standard.dill_with_graph import link_node_loss, link_scheme_loss
                n_loss, ress = link_node_loss(self.all_target_node_index['test'], self.all_pos_node_index['test'],
                                              self.all_neg_node_index['test'], node_emd,
                                              self.model.link_relation_dict, self.args.dataset_name)

                # network schema loss
                s_loss = link_scheme_loss(self.all_target_node_index['test'], self.all_neibor_node_idx['test'],
                                          self.really_list['test'], node_emd, self.model.node_encoder,
                                          self.args.dataset_name)
                loss = n_loss + (0.5 * s_loss)
                ndcgs = []
                for i in ress:
                    ai = np.zeros(len(i[0]))
                    ai[0] = 1
                    # this argsort sort the array and return the index
                    # i.argsort(descending=True)
                    ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in i.argsort(descending=True)]
                ndcgs = np.average(ndcgs)
                metric_dict = {mode: {'ndcgs': ndcgs, 'loss': loss.item()}}
                # loss_dict = {mode: loss.item()}
        return metric_dict

    def _mini_train_step(self, ):
        self.model.train()
        all_loss = 0
        loader_tqdm = tqdm(self.dataloader, ncols=120)
        for input_nodes, positive_graph, negative_graph, blocks in loader_tqdm:
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            if type(input_nodes) == th.Tensor:
                input_nodes = {self.category: input_nodes}
            input_features = self.model.input_feature.forward_nodes(input_nodes)
            logits = self.model(blocks, input_features)
            loss = self.loss_calculation(positive_graph, negative_graph, logits)
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        p_score = self.task.ScorePredictor(positive_graph, embedding, self.r_embedding)
        n_score = self.task.ScorePredictor(negative_graph, embedding, self.r_embedding)
        p_label = th.ones(len(p_score), device=self.device)
        n_label = th.zeros(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(th.cat((p_score, n_score)), th.cat((p_label, n_label)))
        return loss

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def _test_step(self, split=None):
        if self.args.mini_batch_flag:
            return self._mini_test_step(split=split)
        else:
            return self._full_test_step(split=split)

    def _full_test_step(self, split=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _mini_test_step(self, split=None):
        print('mini test...\n')
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=self.fanouts, g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _full_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=[-1] * self.args.num_layers, g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_embedding(self, model, fanouts, g, device, dim, ntypes, batch_size):
        model.eval()
        with th.no_grad():
            sampler = dgl.dataloading.NeighborSampler(fanouts)
            indices = {ntype: torch.arange(g.num_nodes(ntype)).to(device) for ntype in ntypes}
            embedding = {ntype: torch.zeros(g.num_nodes(ntype), dim).to(device) for ntype in ntypes}
            dataloader = dgl.dataloading.DataLoader(
                g, indices, sampler,
                device=device,
                batch_size=batch_size)
            loader_tqdm = tqdm(dataloader, ncols=120)

            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if not isinstance(input_nodes, dict):
                    input_nodes = {self.category: input_nodes}
                input_emb = model.input_feature.forward_nodes(input_nodes)
                output_emb = model(blocks, input_emb)

                for ntype, idx in seeds.items():
                    embedding[ntype][idx] = output_emb[ntype]
            return embedding
