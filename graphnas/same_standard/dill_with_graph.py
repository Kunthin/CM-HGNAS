import math
import time
import random
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F

yelp_etype_list = {
    'bre': ['business', 'reservation'],
    'bse': ['business', 'service'],
    'bst': ['business', 'stars'],
    'bus': ['business', 'user'],
    'reb': ['reservation', 'business'],
    'seb': ['service', 'business'],
    'stb': ['stars', 'business'],
    'usb': ['user', 'business']
}
half_nodeclass_etype_dict = {
    'HGBn-ACM': ['paper-author', 'paper-cite-paper', 'paper-subject', 'paper-term'],
    'HGBn-IMDB': ['movie->actor', 'movie->director', 'movie->keyword'],
    'HGBl-LastFM': ['user-artist-rev', 'artist-tag', 'user-user'],
    'HGBl-amazon': ['product-product-0', 'product-product-1'],
    'yelp4HeGAN': ['bre', 'bse', 'bst', 'bus'],
    'HGBn-DBLP': ['paper-author', 'paper-term', 'paper-venue']
}
node_neg_sample_size = 20


def set_node_type(etype, name):
    if name in ['HGBn-ACM', 'HGBn-DBLP']:
        elist = etype.split('-')
    elif name in ['HGBn-IMDB']:
        elist = etype.split('->')
    elif name == 'yelp4HeGAN':
        # ['bre', 'bse', 'bst', 'bus', 'reb', 'seb', 'stb', 'usb']
        return yelp_etype_list[etype][0], yelp_etype_list[etype][1]
    return elist[0], elist[-1]


def get_sub_node_idx(hg, idx, type, mode):
    t_dict = {}
    for ntype in hg.ntypes:
        if ntype == type:
            if torch.is_tensor(idx):
                t_idx = idx.numpy().tolist()
            else:
                t_idx = idx
            # if mode == 'train':
            #     t_idx = t_idx[:int(len(t_idx)*0.2)]
            t_dict[ntype] = t_idx
        else:
            t_dict[ntype] = list(range(hg.number_of_nodes(ntype)))
    return t_dict


def get_ori_idx(name, t_type, idx, ori_list):
    if name == t_type:
        return ori_list[[idx]]
    return idx


# we only need to make some change to the really_list of node classification
def half_etype_pos_neg_sample(hg, etypes, d_name, really_list, ori_list, all_target_node_index, all_pos_node_index,
                              all_neg_node_index, all_neibor_node_idx):
    for etype in etypes:
        temp_pos_idx = torch.Tensor([])
        temp_tar_idx = torch.Tensor([])
        tar_name, pos_name = set_node_type(etype, d_name)
        pos_len = hg.number_of_nodes(pos_name)
        all_neg = set(list(range(pos_len)))
        temp_neg_idx = []
        temp_neibor_idx = {}
        node_dict = hg.edges(etype=etype)
        tar_index = node_dict[0].cpu()
        pos_index = node_dict[1].cpu()
        start_idx = end_idx = 0
        tar_index_len = len(tar_index)
        t_really_list = []
        while end_idx < tar_index_len:
            if tar_index[end_idx] == tar_index[start_idx]:
                all_neg.discard(int(pos_index[end_idx]))
                end_idx += 1
            else:
                perm = np.random.randint(start_idx, end_idx)
                temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                         dim=0)
                temp_pos_idx = torch.cat([temp_pos_idx, pos_index[[perm]]],
                                         dim=0)
                if len(all_neg) <= node_neg_sample_size:
                    neg_perm = list(all_neg)
                else:
                    neg_perm = random.sample(all_neg, node_neg_sample_size)
                temp_neg_idx.append(neg_perm)
                temp_neibor_idx[int(tar_index[perm])] = pos_index[start_idx:end_idx]
                t_really_list.append(int(tar_index[perm]))
                all_neg = set(list(range(pos_len)))
                start_idx = end_idx
        # do the else again for the last node
        perm = np.random.randint(start_idx, end_idx)
        temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                 dim=0)
        temp_pos_idx = torch.cat([temp_pos_idx, pos_index[[perm]]],
                                 dim=0)
        if len(all_neg) <= node_neg_sample_size:
            neg_perm = list(all_neg)
        else:
            neg_perm = random.sample(all_neg, node_neg_sample_size)
        temp_neg_idx.append(neg_perm)
        temp_neibor_idx[int(tar_index[perm])] = pos_index[start_idx:end_idx]
        t_really_list.append(int(tar_index[perm]))
        # end of this
        really_list.append(t_really_list)
        pos_node_index = {pos_name: temp_pos_idx}
        target_node_index = {tar_name: temp_tar_idx}
        all_target_node_index[etype] = target_node_index
        all_pos_node_index[etype] = pos_node_index
        all_neg_node_index[etype] = temp_neg_idx
        all_neibor_node_idx.append({pos_name: temp_neibor_idx})
        torch.cuda.empty_cache()


def pos_neg_sample(hg, d_name, t_type, ori_list):
    np.random.seed(int(time.time()))
    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = []
    really_list = []
    etypes = half_nodeclass_etype_dict[d_name]
    half_etype_pos_neg_sample(hg, etypes, d_name, really_list, ori_list, all_target_node_index, all_pos_node_index,
                              all_neg_node_index, all_neibor_node_idx)
    # if d_name == 'yelp4HeGAN':
    #     for etype in hg.etypes:
    #         # postive sample
    #         tar_name, pos_name = set_node_type(etype, d_name)
    #         temp_pos_idx = torch.Tensor([])
    #         temp_tar_idx = torch.Tensor([])
    #         temp_neg_idx = {}
    #         temp_neibor_idx = []
    #         if tar_name != t_type:
    #             pos_name = tar_name
    #             tar_name = t_type
    #             total_pos_num = hg.number_of_nodes(pos_name)
    #             total_tar_num = hg.number_of_nodes(tar_name)
    #             t_all_neg = list(range(total_pos_num))
    #             for idx in range(total_tar_num):
    #                 all_pos = hg.in_edges(idx, etype=etype)[0]
    #                 if all_pos.shape[0] > 0:
    #                     perm = np.random.randint(all_pos.shape[0], size=1)
    #                     temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                              dim=0)
    #                     temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                              dim=0)
    #                     temp_all_pos = all_pos.cpu().numpy().tolist()
    #                     all_neg = t_all_neg[:]
    #                     for t_index in temp_all_pos:
    #                         all_neg.remove(int(t_index))
    #                     neg_perm = np.random.choice(all_neg, size=node_neg_sample_size)
    #                     temp_neg_idx[idx] = neg_perm
    #                     temp_neibor_idx.append(int(all_pos[perm].cpu()))
    #                 else:
    #                     temp_neibor_idx.append(-1)
    #                     remove_list.append(idx)
    #         else:
    #             total_pos_num = hg.number_of_nodes(pos_name)
    #             total_tar_num = hg.number_of_nodes(tar_name)
    #             t_all_neg = list(range(total_pos_num))
    #             for idx in range(total_tar_num):
    #                 all_pos = hg.out_edges(idx, etype=etype)[1]
    #                 if all_pos.shape[0] > 0:
    #                     perm = np.random.randint(all_pos.shape[0], size=1)
    #                     temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                              dim=0)
    #                     temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                              dim=0)
    #                     temp_all_pos = all_pos.cpu().numpy().tolist()
    #                     all_neg = t_all_neg[:]
    #                     for t_index in temp_all_pos:
    #                         all_neg.remove(int(t_index))
    #                     neg_perm = np.random.choice(all_neg, size=node_neg_sample_size)
    #                     temp_neg_idx[idx] = neg_perm
    #                     temp_neibor_idx.append(int(all_pos[perm].cpu()))
    #                 else:
    #                     temp_neibor_idx.append(-1)
    #                     remove_list.append(idx)
    #         pos_node_index = {pos_name: temp_pos_idx}
    #         target_node_index = {tar_name: temp_tar_idx}
    #         all_target_node_index[etype] = target_node_index
    #         all_pos_node_index[etype] = pos_node_index
    #         all_neg_node_index[etype] = temp_neg_idx
    #         all_neibor_node_idx.append({pos_name: torch.Tensor(temp_neibor_idx)})
    #         torch.cuda.empty_cache()
    # else:
    #     for etype in hg.etypes:
    #         # postive sample
    #         tar_name, pos_name = set_node_type(etype, d_name)
    #         temp_pos_idx = torch.Tensor([])
    #         temp_tar_idx = torch.Tensor([])
    #         temp_neg_idx = {}
    #         temp_neibor_idx = []
    #         if tar_name != t_type:
    #             pos_name = tar_name
    #             tar_name = t_type
    #             total_pos_num = hg.number_of_nodes(pos_name)
    #             total_tar_num = hg.number_of_nodes(tar_name)
    #             t_all_neg = list(range(total_pos_num))
    #             for idx in range(total_tar_num):
    #                 all_pos = hg.in_edges(idx, etype=etype)[0]
    #                 if all_pos.shape[0] > 0:
    #                     perm = np.random.randint(all_pos.shape[0], size=1)
    #                     temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                              dim=0)
    #                     temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                              dim=0)
    #                     temp_all_pos = all_pos.cpu().numpy().tolist()
    #                     all_neg = t_all_neg[:]
    #                     for t_index in temp_all_pos:
    #                         all_neg.remove(int(t_index))
    #                     neg_perm = random.sample(all_neg, node_neg_sample_size)
    #                     temp_neg_idx[idx] = neg_perm
    #                     temp_neibor_idx.append(int(all_pos[perm].cpu()))
    #                 else:
    #                     temp_neibor_idx.append(-1)
    #                     remove_list.append(idx)
    #         else:
    #             total_pos_num = hg.number_of_nodes(pos_name)
    #             total_tar_num = hg.number_of_nodes(tar_name)
    #             t_all_neg = list(range(total_pos_num))
    #             for idx in range(total_tar_num):
    #                 all_pos = hg.out_edges(idx, etype=etype)[1]
    #                 if all_pos.shape[0] > 0:
    #                     perm = np.random.randint(all_pos.shape[0], size=1)
    #                     temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                              dim=0)
    #                     temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                              dim=0)
    #                     temp_all_pos = all_pos.cpu().numpy().tolist()
    #                     all_neg = t_all_neg[:]
    #                     for t_index in temp_all_pos:
    #                         all_neg.remove(int(t_index))
    #                     neg_perm = random.sample(all_neg, node_neg_sample_size)
    #                     temp_neg_idx[idx] = neg_perm
    #                     temp_neibor_idx.append(int(all_pos[perm].cpu()))
    #                 else:
    #                     temp_neibor_idx.append(-1)
    #                     remove_list.append(idx)
    #         pos_node_index = {pos_name: temp_pos_idx}
    #         target_node_index = {tar_name: temp_tar_idx}
    #         all_target_node_index[etype] = target_node_index
    #         all_pos_node_index[etype] = pos_node_index
    #         all_neg_node_index[etype] = temp_neg_idx
    #         all_neibor_node_idx.append({pos_name: torch.Tensor(temp_neibor_idx)})
    #         torch.cuda.empty_cache()
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, really_list


# this function need to do such things:
# sample graph from big graph
# and this function need to return the pos index and neg index. this may be a dict.
def sample_subgraph(hg, sample_number, dataset_name, train_idx, val_idx, test_idx, target_type):
    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = {}
    remove_list = {}
    train_dict = get_sub_node_idx(hg, train_idx, target_type, 'train')
    train_graph = dgl.node_subgraph(hg, train_dict)
    train_ori_idx = train_graph.nodes[target_type].data[dgl.NID]
    val_dict = get_sub_node_idx(hg, val_idx, target_type, 'val')
    val_graph = dgl.node_subgraph(hg, val_dict)
    val_ori_idx = val_graph.nodes[target_type].data[dgl.NID]
    test_dict = get_sub_node_idx(hg, test_idx, target_type, 'test')
    test_graph = dgl.node_subgraph(hg, test_dict)
    test_ori_idx = test_graph.nodes[target_type].data[dgl.NID]
    train_all_target_node_index, train_all_pos_node_index, train_all_neg_node_index, train_all_neibor_node_idx, train_remove_list = pos_neg_sample(
        train_graph,
        dataset_name,
        target_type,
        train_ori_idx)
    val_all_target_node_index, val_all_pos_node_index, val_all_neg_node_index, val_all_neibor_node_idx, val_remove_list = pos_neg_sample(
        val_graph,
        dataset_name,
        target_type,
        val_ori_idx)
    test_all_target_node_index, test_all_pos_node_index, test_all_neg_node_index, test_all_neibor_node_idx, test_remove_list = pos_neg_sample(
        test_graph,
        dataset_name,
        target_type,
        test_ori_idx)
    all_target_node_index['train'] = train_all_target_node_index
    all_pos_node_index['train'] = train_all_pos_node_index
    all_neg_node_index['train'] = train_all_neg_node_index
    all_neibor_node_idx['train'] = train_all_neibor_node_idx
    remove_list['train'] = train_remove_list
    all_target_node_index['val'] = val_all_target_node_index
    all_pos_node_index['val'] = val_all_pos_node_index
    all_neg_node_index['val'] = val_all_neg_node_index
    all_neibor_node_idx['val'] = val_all_neibor_node_idx
    remove_list['val'] = val_remove_list
    all_target_node_index['test'] = test_all_target_node_index
    all_pos_node_index['test'] = test_all_pos_node_index
    all_neg_node_index['test'] = test_all_neg_node_index
    all_neibor_node_idx['test'] = test_all_neibor_node_idx
    remove_list['test'] = test_remove_list
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, remove_list


class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''

    def __init__(self, n_hid, n_out, temperature=0.1):
        super(Matcher, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, n_out)  # it seems like that this linear is the relation matrix R.
        self.sqrt_hd = math.sqrt(n_out)
        self.drop = nn.Dropout(0.2)
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    def forward(self, x, ty, use_norm=True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd

    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)


class NodeEncoder(nn.Module):
    def __init__(self, n_hid, n_out):
        super(NodeEncoder, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        return self.drop(self.linear(x))


def node_loss(target_node_idx, pos_node_idx, neg_node_idx, node_emd, d_name, link_relation_dict):
    node_loss = 0
    # this will be very useful in the future.
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    ress = []
    etypes = half_nodeclass_etype_dict[d_name]
    for etype in etypes:
        for key, value in target_node_idx[etype].items():
            target_key = key
            target_index = value
        for key, value in pos_node_idx[etype].items():
            pos_key = key
            pos_index = value
        # set query_emb_idx
        repeat_size = len(neg_node_idx[etype][0])
        query_emb_idx = torch.reshape(target_index, [-1, 1])
        query_emb_idx = query_emb_idx.repeat(1, repeat_size + 1).long()
        # set key_idx
        key_pos_idx = torch.reshape(pos_index, [-1, 1])
        key_neg_idx = torch.Tensor(neg_node_idx[etype])
        key_idx = torch.cat([key_pos_idx, key_neg_idx], dim=1).long()
        if (len(target_index)) > 200:
            scale_perm = random.sample(list(range(len(target_index))), 200)
            query_emb_idx = query_emb_idx[scale_perm]
            key_idx = key_idx[scale_perm]
        query_emb = node_emd[target_key][query_emb_idx]
        key_emb = node_emd[pos_key][key_idx]
        # scale the size of the train node.

        res = link_relation_dict[etype].forward(query_emb, key_emb)
        ress += [res.detach()]
        node_loss += F.log_softmax(res, dim=-1)[:, 0].mean()
    return -node_loss / len(etypes), ress


def link_node_loss(target_node_idx, pos_node_idx, neg_node_idx, node_emd, link_relation_dict, d_name):
    loss, ress = node_loss(target_node_idx, pos_node_idx, neg_node_idx, node_emd, d_name, link_relation_dict)
    return loss, ress


def scheme_loss(neibor_node_idx, really_list, node_emd, tar_idx, tar_name, node_encoder):
    loss = 0
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    temp_really_list = set(list(range(len(tar_idx))))
    for t_really_list in really_list:
        temp_really_list = temp_really_list & set(t_really_list)
    temp_really_list = list(temp_really_list)
    temp_really_list.sort()
    if len(temp_really_list) == 0:
        return loss
    schema_emb = torch.Tensor([]).to('cuda:0')
    for neibor in neibor_node_idx:
        for name, val in neibor.items():
            neibor_idx = []
            for t_val in temp_really_list:
                tmp_ids = np.random.choice(val[t_val], size=2).tolist()
                neibor_idx.append(tmp_ids)
            neibor_idx = torch.Tensor(neibor_idx).long()
            temp_neibor_emd = node_emd[name][neibor_idx]
            # temp_neibor_emd = node_emd[name][val[temp_really_list].long()]
            temp_neibor_emd = node_encoder[name].forward(temp_neibor_emd)
            temp_neibor_emd = torch.mean(temp_neibor_emd, dim=1)
            temp_neibor_emd = torch.unsqueeze(temp_neibor_emd, 0)
            schema_emb = torch.cat([schema_emb, temp_neibor_emd], dim=0)
    schema_emb = torch.mean(schema_emb, dim=0)
    schema_idxs = []
    # set key embedding
    if len(temp_really_list) <= 101:
        for idx in range(len(temp_really_list)):
            t_list = list(range(len(temp_really_list)))
            t_list.pop(idx)
            schema_idxs.append([idx] + t_list)
    else:
        for idx in range(len(temp_really_list)):
            t_list = list(range(len(temp_really_list)))
            t_list.pop(idx)
            perm = random.sample(t_list, 100)
            schema_idxs.append([idx] + perm)
    repeat_size = len(schema_idxs[0])
    schema_idxs = torch.Tensor(schema_idxs).long()
    key_schema_emb = schema_emb[schema_idxs]
    # set query_emb_idx
    t_tar_idx = tar_idx[temp_really_list]
    query_emb_idx = torch.reshape(t_tar_idx, [-1, 1])
    query_emb_idx = query_emb_idx.repeat(1, repeat_size).long()
    query_emb = node_emd[tar_name][query_emb_idx]
    q_k_cos = cos(query_emb, key_schema_emb) / 0.1
    loss += F.log_softmax(q_k_cos, dim=-1)[:, 0].mean()
    return -loss


def link_scheme_loss(target_node_idx, neibor_node_idx, really_list, node_emd, node_encoder, dataset_name):
    loss = 0
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    schema_emb = torch.Tensor([]).to('cuda:0')
    if dataset_name == 'HGBl-LastFM':
        # deal artist-user
        neibor = neibor_node_idx[0]
        for name, val in neibor.items():
            neibor_idx = []
            for val_idx in really_list:
                tmp_ids = np.random.choice(val[val_idx], size=2).tolist()
                neibor_idx.append(tmp_ids)
            neibor_idx = torch.Tensor(neibor_idx).long()
            temp_neibor_emd = node_emd[name][neibor_idx]
            temp_neibor_emd = node_encoder[name].forward(temp_neibor_emd)
            temp_neibor_emd = torch.mean(temp_neibor_emd, dim=1)
            temp_neibor_emd = torch.unsqueeze(temp_neibor_emd, 0)
        schema_emb = torch.cat([schema_emb, temp_neibor_emd], dim=0)
        # deal artist-tag
        neibor = neibor_node_idx[1]
        for name, val in neibor.items():
            neibor_idx = []
            for t_val in val:
                tmp_ids = np.random.choice(t_val, size=2).tolist()
                neibor_idx.append(tmp_ids)
            neibor_idx = torch.Tensor(neibor_idx).long()
            temp_neibor_emd = node_emd[name][neibor_idx]
            temp_neibor_emd = node_encoder[name].forward(temp_neibor_emd)
            temp_neibor_emd = torch.mean(temp_neibor_emd, dim=1)
            temp_neibor_emd = torch.unsqueeze(temp_neibor_emd, 0)
        schema_emb = torch.cat([schema_emb, temp_neibor_emd], dim=0)
        # end of this
        schema_emb = torch.mean(schema_emb, dim=0)
        schema_idxs = []
        # set key embedding
        temp_val = target_node_idx['artist-tag']['artist']
        if len(temp_val) == 0:
            return loss
        if len(temp_val) <= 101:
            for idx in range(len(temp_val)):
                t_list = list(range(len(temp_val)))
                t_list.pop(idx)
                schema_idxs.append([idx] + t_list)
        else:
            for idx in range(len(temp_val)):
                t_list = list(range(len(temp_val)))
                t_list.pop(idx)
                perm = random.sample(t_list, 100)
                schema_idxs.append([idx] + perm)
        repeat_size = len(schema_idxs[0])
        schema_idxs = torch.Tensor(schema_idxs).long()
        key_schema_emb = schema_emb[schema_idxs]
        # set query_emb_idx
        t_tar_idx = temp_val
        query_emb_idx = torch.reshape(t_tar_idx, [-1, 1])
        query_emb_idx = query_emb_idx.repeat(1, repeat_size).long()
        query_emb = node_emd['artist'][query_emb_idx]
        q_k_cos = cos(query_emb, key_schema_emb) / 0.1
        loss = F.log_softmax(q_k_cos, dim=-1)[:, 0].mean()
    elif dataset_name == 'HGBl-amazon':
        i = 0
        for neibor in neibor_node_idx:
            for name, val in neibor.items():
                neibor_idx = []
                for val_idx in val:
                    tmp_ids = np.random.choice(val_idx, size=5).tolist()
                    neibor_idx.append(tmp_ids)
                temp_neibor_emd = node_emd[name][neibor_idx]
                temp_neibor_emd = node_encoder[name].forward(temp_neibor_emd)
                temp_neibor_emd = torch.mean(temp_neibor_emd, dim=1)
            schema_emb = temp_neibor_emd
            schema_idxs = []
            # set key embedding
            if i == 0:
                temp_val = target_node_idx['product-product-0']['product']
            else:
                temp_val = target_node_idx['product-product-1']['product']
            for idx in range(len(temp_val)):
                t_list = list(range(len(temp_val)))
                t_list.pop(idx)
                perm = random.sample(t_list, 50)
                schema_idxs.append([idx] + perm)
            schema_idxs = torch.Tensor(schema_idxs).long()
            key_schema_emb = schema_emb[schema_idxs]
            # set query_emb_idx
            t_tar_idx = temp_val
            query_emb_idx = torch.reshape(t_tar_idx, [-1, 1])
            query_emb_idx = query_emb_idx.repeat(1, 50 + 1).long()
            query_emb = node_emd['product'][query_emb_idx]
            q_k_cos = cos(query_emb, key_schema_emb) / 0.1
            loss += F.log_softmax(q_k_cos, dim=-1)[:, 0].mean()
            i += 1
        loss /= 2
    return -loss


def dcg_of_k_length(r, k):
    r = np.asfarray(r)[:k]  # turn to float array
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_of_k_length(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_of_k_length(r, k) / dcg_max
    # for etype in hg.etypes:
    #     temp_edges = {etype: hg.edges(etype=etype)}
    #     current_etype_number = len(temp_edges[etype][0])
    #     # postive sample
    #     perm = np.random.randint(current_etype_number, size=sample_number)
    #     tar_name, pos_name = set_node_type(etype, dataset_name)
    #     total_pos_num = hg.number_of_nodes(pos_name)
    #     target_node_index = {tar_name: temp_edges[etype][0][perm]}
    #     sup_pos_node_index = {pos_name: temp_edges[etype][1][perm]}
    #     neg_node_index = {}
    #     i = 0
    #     for tem_tar_index in target_node_index[tar_name]:
    #         all_pos = hg.out_edges(tem_tar_index, etype=etype)[1]
    #         temp_all_pos = all_pos.cpu().numpy().tolist()
    #         all_neg = list(range(total_pos_num))
    #         for t_index in temp_all_pos:
    #             all_neg.remove(int(t_index))
    #         neg_perm = random.sample(all_neg, node_neg_sample_size)
    #         neg_node_index[i] = neg_perm
    #         i += 1
    #     all_target_node_index[etype] = target_node_index
    #     all_sup_pos_node_index[etype] = sup_pos_node_index
    #     all_neg_node_index[etype] = neg_node_index

    # for etype in hg.etypes:
    #     # postive sample
    #     tar_name, pos_name = set_node_type(etype, d_name)
    #     temp_pos_idx = torch.Tensor([])
    #     temp_tar_idx = torch.Tensor([])
    #     temp_neg_idx = {}
    #     if tar_name != t_type:
    #         pos_name = tar_name
    #         tar_name = t_type
    #         total_pos_num = hg.number_of_nodes(pos_name)
    #         total_tar_num = hg.number_of_nodes(tar_name)
    #         t_all_neg = list(range(total_pos_num))
    #         for idx in range(total_tar_num):
    #             all_pos = hg.in_edges(idx, etype=etype)[0]
    #             if all_pos.shape[0] > 0:
    #                 perm = np.random.randint(all_pos.shape[0], size=1)
    #                 temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                          dim=0)
    #                 temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                          dim=0)
    #                 temp_all_pos = all_pos.cpu().numpy().tolist()
    #                 all_neg = t_all_neg[:]
    #                 for t_index in temp_all_pos:
    #                     all_neg.remove(int(t_index))
    #                 neg_perm = random.sample(all_neg, node_neg_sample_size)
    #                 temp_neg_idx[idx] = neg_perm
    #     else:
    #         total_pos_num = hg.number_of_nodes(pos_name)
    #         total_tar_num = hg.number_of_nodes(tar_name)
    #         t_all_neg = list(range(total_pos_num))
    #         for idx in range(total_tar_num):
    #             all_pos = hg.out_edges(idx, etype=etype)[1]
    #             if all_pos.shape[0] > 0:
    #                 perm = np.random.randint(all_pos.shape[0], size=1)
    #                 temp_pos_idx = torch.cat([temp_pos_idx, all_pos[perm].cpu()],
    #                                          dim=0)
    #                 temp_tar_idx = torch.cat([temp_tar_idx, ori_list[[idx]].cpu()],
    #                                          dim=0)
    #                 temp_all_pos = all_pos.cpu().numpy().tolist()
    #                 all_neg = t_all_neg[:]
    #                 for t_index in temp_all_pos:
    #                     all_neg.remove(int(t_index))
    #                 neg_perm = random.sample(all_neg, node_neg_sample_size)
    #                 temp_neg_idx[idx] = neg_perm
    #     pos_node_index = {pos_name: temp_pos_idx}
    #     target_node_index = {tar_name: temp_tar_idx}
    #     all_target_node_index[etype] = target_node_index
    #     all_pos_node_index[etype] = pos_node_index
    #     all_neg_node_index[etype] = temp_neg_idx


def get_split(hg, node_type):
    # set train val test to 0.4 0.1 0.5
    test_rate = 0.5
    num_nodes = hg.number_of_nodes(node_type)
    n_test = int(num_nodes * test_rate)
    n_train = num_nodes - n_test
    train, test = torch.utils.data.random_split(range(num_nodes), [n_train, n_test])
    train_idx = torch.tensor(train.indices).long()
    test_idx = torch.tensor(test.indices).long()
    random_int = torch.randperm(len(train_idx))
    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
    train_idx = train_idx[random_int[len(train_idx) // 5:]]
    return train_idx, valid_idx, test_idx


# this four functions I cant make any change
def HGBl_LastFM_pos_neg_sample(hg, ori_list):  # d_name, t_type, ori_list
    np.random.seed(int(time.time()))
    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = []
    really_list = []

    def my_set_node_type(etype):
        elist = etype.split('-')
        if etype == 'user-artist-rev':
            return elist[1], elist[0]
        return elist[0], elist[1]

    for etype in ['user-artist-rev', 'artist-tag', 'user-user']:
        temp_pos_idx = torch.Tensor([])
        temp_tar_idx = torch.Tensor([])
        tar_name, pos_name = my_set_node_type(etype)
        pos_len = hg.number_of_nodes(pos_name)
        all_neg = list(range(pos_len))
        temp_neg_idx = []
        temp_neibor_idx = []
        node_dict = hg.edges(etype=etype)
        tar_index = node_dict[0].cpu()
        pos_index = node_dict[1].cpu()
        start_idx = end_idx = 0
        tar_index_len = len(tar_index)
        t_set = set()
        while end_idx < tar_index_len:
            if tar_index[end_idx] == tar_index[start_idx]:
                t_set.add(int(pos_index[end_idx]))
                end_idx += 1
            else:
                for val in t_set:
                    all_neg.remove(val)
                perm = np.random.randint(start_idx, end_idx)
                if tar_name == 'artist':
                    temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                             dim=0)
                else:
                    temp_tar_idx = torch.cat([temp_tar_idx, tar_index[[perm]]],
                                             dim=0)
                temp_pos_idx = torch.cat([temp_pos_idx, pos_index[[perm]]],
                                         dim=0)
                neg_perm = random.sample(all_neg, node_neg_sample_size)
                temp_neg_idx.append(neg_perm)
                temp_neibor_idx.append(pos_index[start_idx:end_idx])
                if etype == 'artist-tag':
                    really_list.append(int(tar_index[perm]))
                all_neg = list(range(pos_len))
                t_set = set()
                start_idx = end_idx
        # do the else again for the last node
        for val in t_set:
            all_neg.remove(val)
        perm = np.random.randint(start_idx, end_idx)
        if tar_name == 'artist':
            temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                     dim=0)
        else:
            temp_tar_idx = torch.cat([temp_tar_idx, tar_index[[perm]]],
                                     dim=0)
        temp_pos_idx = torch.cat([temp_pos_idx, pos_index[[perm]]],
                                 dim=0)
        neg_perm = random.sample(all_neg, node_neg_sample_size)
        temp_neg_idx.append(neg_perm)
        temp_neibor_idx.append(pos_index[start_idx:end_idx])
        if etype == 'artist-tag':
            really_list.append(int(tar_index[perm]))
        # end of this
        pos_node_index = {pos_name: temp_pos_idx}
        target_node_index = {tar_name: temp_tar_idx}
        all_target_node_index[etype] = target_node_index
        all_pos_node_index[etype] = pos_node_index
        all_neg_node_index[etype] = temp_neg_idx
        all_neibor_node_idx.append({pos_name: temp_neibor_idx})
        torch.cuda.empty_cache()
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, really_list


def HGBl_amazon_pos_neg_sample(hg, ori_list):  # d_name, t_type, ori_list
    np.random.seed(int(time.time()))
    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = []
    really_list = []
    for etype in ['product-product-0', 'product-product-1']:
        temp_pos_idx = torch.Tensor([])
        temp_tar_idx = torch.Tensor([])
        tar_name = pos_name = 'product'
        pos_len = hg.number_of_nodes(pos_name)
        all_neg = list(range(pos_len))
        temp_neg_idx = []
        temp_neibor_idx = []
        node_dict = hg.edges(etype=etype)
        tar_index = node_dict[0].cpu()
        pos_index = node_dict[1].cpu()
        start_idx = end_idx = 0
        tar_index_len = len(tar_index)
        t_set = set()
        while end_idx < tar_index_len:
            if tar_index[end_idx] == tar_index[start_idx]:
                t_set.add(int(pos_index[end_idx]))
                end_idx += 1
            else:
                for val in t_set:
                    all_neg.remove(val)
                perm = np.random.randint(start_idx, end_idx)
                temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                         dim=0)
                temp_pos_idx = torch.cat([temp_pos_idx, ori_list[pos_index[[perm]]].cpu()],
                                         dim=0)
                neg_perm = random.sample(all_neg, node_neg_sample_size)
                temp_neg_idx.append(ori_list[neg_perm].cpu().numpy().tolist())
                temp_neibor_idx.append(ori_list[pos_index[start_idx:end_idx]].cpu().numpy().tolist())
                all_neg = list(range(pos_len))
                t_set = set()
                start_idx = end_idx
        # do the else again for the last node
        for val in t_set:
            all_neg.remove(val)
        perm = np.random.randint(start_idx, end_idx)
        temp_tar_idx = torch.cat([temp_tar_idx, ori_list[tar_index[[perm]]].cpu()],
                                 dim=0)
        temp_pos_idx = torch.cat([temp_pos_idx, ori_list[pos_index[[perm]]].cpu()],
                                 dim=0)
        neg_perm = random.sample(all_neg, node_neg_sample_size)
        temp_neg_idx.append(ori_list[neg_perm].cpu().numpy().tolist())
        temp_neibor_idx.append(ori_list[pos_index[start_idx:end_idx]].cpu().numpy().tolist())
        # end of this
        pos_node_index = {pos_name: temp_pos_idx}
        target_node_index = {tar_name: temp_tar_idx}
        all_target_node_index[etype] = target_node_index
        all_pos_node_index[etype] = pos_node_index
        all_neg_node_index[etype] = temp_neg_idx
        all_neibor_node_idx.append({pos_name: temp_neibor_idx})
        torch.cuda.empty_cache()
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, really_list


def HGBl_LastFM_sample_subgraph(hg):
    # Graph(num_nodes={'artist': 17632, 'tag': 1088, 'user': 1892}, num_edges={('artist', 'artist-tag',
    # 'tag'): 23253, ('artist', 'user-artist-rev', 'user'): 66841, ('tag', 'artist-tag-rev', 'artist'): 23253,
    # ('user', 'user-artist', 'artist'): 66841, ('user', 'user-user', 'user'): 25434,
    # ('user', 'user-user-rev', 'user'): 25434},
    artist_train_idx, artist_val_idx, artist_test_idx = get_split(hg, 'artist')
    user_train_idx, user_val_idx, user_test_idx = get_split(hg, 'user')  # this line is useless

    def get_artist_user_idx(hg, a_idx, u_idx):
        t_dict = {}
        for ntype in hg.ntypes:
            if ntype == 'artist':
                t_dict[ntype] = a_idx.numpy().tolist()
            # elif ntype == 'user':
            #     t_dict[ntype] = u_idx
            else:
                t_dict[ntype] = list(range(hg.number_of_nodes(ntype)))
        return t_dict

    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = {}
    remove_list = {}
    train_dict = get_artist_user_idx(hg, artist_train_idx, user_train_idx)
    train_graph = dgl.node_subgraph(hg, train_dict)
    artist_train_ori_idx = train_graph.nodes['artist'].data[dgl.NID]
    # user_train_ori_idx = train_graph.nodes['user'].data[dgl.NID]
    val_dict = get_artist_user_idx(hg, artist_val_idx, user_val_idx)
    val_graph = dgl.node_subgraph(hg, val_dict)
    artist_val_ori_idx = val_graph.nodes['artist'].data[dgl.NID]
    # user_val_ori_idx = val_graph.nodes['user'].data[dgl.NID]
    test_dict = get_artist_user_idx(hg, artist_test_idx, user_test_idx)
    test_graph = dgl.node_subgraph(hg, test_dict)
    artist_test_ori_idx = test_graph.nodes['artist'].data[dgl.NID]
    # user_test_ori_idx = test_graph.nodes['user'].data[dgl.NID]
    # here are used for test
    # l = hg.edges(etype='user-artist')[0].cpu().numpy().tolist()
    # a = set(l)
    # print(len(l), len(a))
    train_all_target_node_index, train_all_pos_node_index, train_all_neg_node_index, train_all_neibor_node_idx, train_really_list = HGBl_LastFM_pos_neg_sample(
        train_graph, artist_train_ori_idx)
    val_all_target_node_index, val_all_pos_node_index, val_all_neg_node_index, val_all_neibor_node_idx, val_really_list = HGBl_LastFM_pos_neg_sample(
        val_graph, artist_val_ori_idx)
    test_all_target_node_index, test_all_pos_node_index, test_all_neg_node_index, test_all_neibor_node_idx, test_really_list = HGBl_LastFM_pos_neg_sample(
        test_graph, artist_test_ori_idx)
    all_target_node_index['train'] = train_all_target_node_index
    all_pos_node_index['train'] = train_all_pos_node_index
    all_neg_node_index['train'] = train_all_neg_node_index
    all_neibor_node_idx['train'] = train_all_neibor_node_idx
    remove_list['train'] = train_really_list
    all_target_node_index['val'] = val_all_target_node_index
    all_pos_node_index['val'] = val_all_pos_node_index
    all_neg_node_index['val'] = val_all_neg_node_index
    all_neibor_node_idx['val'] = val_all_neibor_node_idx
    remove_list['val'] = val_really_list
    all_target_node_index['test'] = test_all_target_node_index
    all_pos_node_index['test'] = test_all_pos_node_index
    all_neg_node_index['test'] = test_all_neg_node_index
    all_neibor_node_idx['test'] = test_all_neibor_node_idx
    remove_list['test'] = test_really_list
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, remove_list


def HGBl_amazon_sample_subgraph(hg):
    product_train_idx, product_val_idx, product_test_idx = get_split(hg, 'product')

    # a = set(hg.edges(etype='product-product-0')[0])
    # b = set(hg.edges(etype='product-product-0')[1])
    def get_product_user_idx(hg, a_idx):
        t_dict = {}
        for ntype in hg.ntypes:
            if ntype == 'product':
                t_dict[ntype] = a_idx.numpy().tolist()
            # elif ntype == 'user':
            #     t_dict[ntype] = u_idx
            else:
                t_dict[ntype] = list(range(hg.number_of_nodes(ntype)))
        return t_dict

    all_target_node_index = {}
    all_pos_node_index = {}
    all_neg_node_index = {}
    all_neibor_node_idx = {}
    remove_list = {}
    train_dict = get_product_user_idx(hg, product_train_idx)
    train_graph = dgl.node_subgraph(hg, train_dict)
    product_train_ori_idx = train_graph.nodes['product'].data[dgl.NID]
    # user_train_ori_idx = train_graph.nodes['user'].data[dgl.NID]
    val_dict = get_product_user_idx(hg, product_val_idx)
    val_graph = dgl.node_subgraph(hg, val_dict)
    product_val_ori_idx = val_graph.nodes['product'].data[dgl.NID]
    # user_val_ori_idx = val_graph.nodes['user'].data[dgl.NID]
    test_dict = get_product_user_idx(hg, product_test_idx)
    test_graph = dgl.node_subgraph(hg, test_dict)
    product_test_ori_idx = test_graph.nodes['product'].data[dgl.NID]
    # user_test_ori_idx = test_graph.nodes['user'].data[dgl.NID]
    # here are used for test
    # l = hg.edges(etype='user-product')[0].cpu().numpy().tolist()
    # a = set(l)
    # print(len(l), len(a))
    train_all_target_node_index, train_all_pos_node_index, train_all_neg_node_index, train_all_neibor_node_idx, train_really_list = HGBl_amazon_pos_neg_sample(
        train_graph, product_train_ori_idx)
    val_all_target_node_index, val_all_pos_node_index, val_all_neg_node_index, val_all_neibor_node_idx, val_really_list = HGBl_amazon_pos_neg_sample(
        val_graph, product_val_ori_idx)
    test_all_target_node_index, test_all_pos_node_index, test_all_neg_node_index, test_all_neibor_node_idx, test_really_list = HGBl_amazon_pos_neg_sample(
        test_graph, product_test_ori_idx)
    all_target_node_index['train'] = train_all_target_node_index
    all_pos_node_index['train'] = train_all_pos_node_index
    all_neg_node_index['train'] = train_all_neg_node_index
    all_neibor_node_idx['train'] = train_all_neibor_node_idx
    remove_list['train'] = train_really_list
    all_target_node_index['val'] = val_all_target_node_index
    all_pos_node_index['val'] = val_all_pos_node_index
    all_neg_node_index['val'] = val_all_neg_node_index
    all_neibor_node_idx['val'] = val_all_neibor_node_idx
    remove_list['val'] = val_really_list
    all_target_node_index['test'] = test_all_target_node_index
    all_pos_node_index['test'] = test_all_pos_node_index
    all_neg_node_index['test'] = test_all_neg_node_index
    all_neibor_node_idx['test'] = test_all_neibor_node_idx
    remove_list['test'] = test_really_list
    return all_target_node_index, all_pos_node_index, all_neg_node_index, all_neibor_node_idx, remove_list
