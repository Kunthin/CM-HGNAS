import dgl
from .HeteroMLP import HGNNPostMP, HGNNPreMP
from . import BaseModel, register_model
from ..layers import SkipConnection

stage_dict = {
    'stack': SkipConnection.GNNStackStage,
    'skipsum': SkipConnection.GNNSkipStage,
    'skipconcat': SkipConnection.GNNSkipStage,
}


@register_model('homo_GNN')
class homo_GNN(BaseModel):
    r"""
    General homogeneous GNN model for HGNN
    HeteroMLP + HomoGNN + HeteroMLP
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        out_node_type = args.out_node_type
        return cls(args, hg, out_node_type)

    def __init__(self, args, hg, out_node_type, **kwargs):
        super(homo_GNN, self).__init__()
        self.out_node_type = out_node_type
        self.args = args
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, hg.ntypes, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)
        if args.layers_gnn > 0:
            GNNStage = stage_dict[args.stage_type]
            self.gnn = GNNStage(gnn_type=args.gnn_type,
                                stage_type=args.stage_type,
                                dim_in=args.hidden_dim,
                                dim_out=args.hidden_dim,
                                num_layers=args.layers_gnn,
                                skip_every=1,
                                dropout=args.dropout,
                                act=args.activation,
                                has_bn=args.has_bn,
                                num_heads=args.num_heads,
                                has_l2norm=args.has_l2norm,
                                num_etypes=len(hg.etypes),
                                num_ntypes=len(hg.ntypes))

        gnn_out_dim = self.gnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)
        if args.use_same_standard:
            import torch.nn as nn
            self.relation_params = nn.ModuleList()
            self.link_relation_dict = {}
            from graphnas.same_standard.dill_with_graph import Matcher
            from graphnas.same_standard.dill_with_graph import NodeEncoder
            for relation_type in hg.etypes:
                matcher = Matcher(gnn_out_dim, gnn_out_dim)
                self.link_relation_dict[relation_type] = matcher
                self.relation_params.append(matcher)
            self.node_encoder = {}
            for node_type in hg.ntypes:
                encoder = NodeEncoder(gnn_out_dim, gnn_out_dim)
                self.node_encoder[node_type] = encoder
                self.relation_params.append(encoder)
    # this function we also need make some change.
    def forward(self, hg, h_dict):
        with hg.local_scope():
            # print("len", len(hg.ntypes))
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if len(hg.ntypes) == 1:
                hg.ndata['h'] = h_dict[hg.ntypes[0]]
            else:
                hg.ndata['h'] = h_dict
            homo_g = dgl.to_homogeneous(hg, ndata=['h'])
            homo_g = dgl.remove_self_loop(homo_g)
            homo_g = dgl.add_self_loop(homo_g)
            h = homo_g.ndata.pop('h')
            # print("homo_dict", h_dict)
            # print("h", h)
            if hasattr(self, 'gnn'):
                h = self.gnn(homo_g, h)
                if self.args.use_same_standard:
                    if len(hg.ntypes) == 1:
                        out_h = {hg.ntypes[0]: h}
                    else:
                        out_h = self.h2dict(h, hg.ndata['h'], hg.ntypes)
                else:
                    if len(hg.ntypes) == 1:
                        out_h = {hg.ntypes[0]: h}
                    else:
                        out_h = self.h2dict(h, hg.ndata['h'], self.out_node_type)
            if not self.args.use_same_standard:
                if hasattr(self, 'post_mp'):
                    out_h = self.post_mp(out_h)
        # print("homo_out_h", out_h)
        return out_h

    def h2dict(self, h, hdict, node_list):
        pre = 0
        out_h = {}
        for i, value in hdict.items():
            if i in node_list:
                out_h[i] = h[pre:value.shape[0] + pre]
            pre += value.shape[0]
        return out_h
    def my_forward(self, hg, h_dict):
        with hg.local_scope():
            # print("len", len(hg.ntypes))
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if len(hg.ntypes) == 1:
                hg.ndata['h'] = h_dict[hg.ntypes[0]]
            else:
                hg.ndata['h'] = h_dict
            homo_g = dgl.to_homogeneous(hg, ndata=['h'])
            homo_g = dgl.remove_self_loop(homo_g)
            homo_g = dgl.add_self_loop(homo_g)
            h = homo_g.ndata.pop('h')
            if hasattr(self, 'gnn'):
                h = self.gnn(homo_g, h)
                if len(hg.ntypes) == 1:
                    out_h = {hg.ntypes[0]: h}
                else:
                    out_h = self.h2dict(h, hg.ndata['h'], self.out_node_type)
            if hasattr(self, 'post_mp'):
                out_h = self.post_mp(out_h)
        return out_h