import dgl
from ..layers import SkipConnection
from openhgnn.models import BaseModel, register_model
from ..models.HeteroMLP import HGNNPostMP, HGNNPreMP

stage_dict = {
    'stack': SkipConnection.HGNNStackStage,
    'skipsum': SkipConnection.HGNNSkipStage,
    'skipconcat': SkipConnection.HGNNSkipStage,
}


def HG_transformation(hg, metapaths_dict):
    graph_data = {}
    for key, mp in metapaths_dict.items():
        mp_g = dgl.metapath_reachable_graph(hg, mp)
        n_edge = mp_g.canonical_etypes[0]
        graph_data[(n_edge[0], key, n_edge[2])] = mp_g.edges()
        # print("graph_data",graph_data)
    return dgl.heterograph(graph_data)


@register_model('general_HGNN')
class general_HGNN(BaseModel):  # the BaseModel is a abstract class
    """
    General heterogeneous GNN model
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        # print("hg:",hg)
        out_node_type = args.out_node_type
        # args.subgraph_extraction = 'metapath'


        if args.subgraph_extraction == 'relation':
            new_hg = hg
            # this step seems useless.
            print('relation extraction!')
        elif args.subgraph_extraction == 'metapath':
            if hasattr(args, 'meta_paths_dict'):
                new_hg = HG_transformation(hg, args.meta_paths_dict)
                print('metapath extraction!')
            else:
                raise ValueError('No meta-path is specified!')
        elif args.subgraph_extraction == 'mixed':
            relation_dict = args.meta_paths_dict
            for etype in hg.canonical_etypes:
                relation_dict[etype[1]] = [etype]
            new_hg = HG_transformation(hg, relation_dict)
            print('mixed extraction!')
            pass
        else:
            raise ValueError('subgraph_extraction only supports relation, metapath and mixed')
        # print("new_hg:", new_hg)
        return cls(args, new_hg, out_node_type)

    def __init__(self, args, hg, out_node_type, **kwargs):
        """
        """
        super(general_HGNN, self).__init__()
        self.hg = hg
        self.out_node_type = out_node_type
        self.args = args

        # the first linear is operated in outside of model (in trainerflow)
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, self.hg.ntypes, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)

        if args.layers_gnn > 0:
            HGNNStage = stage_dict[args.stage_type]
            # the layers_gnn define the layers of self.hgnn
            self.hgnn = HGNNStage(gnn_type=args.gnn_type,
                                  rel_names=self.hg.etypes,
                                  stage_type=args.stage_type,
                                  dim_in=args.hidden_dim,
                                  dim_out=args.hidden_dim,
                                  num_layers=args.layers_gnn,
                                  skip_every=1,
                                  dropout=args.dropout,
                                  act=args.activation,
                                  has_bn=args.has_bn,
                                  has_l2norm=args.has_l2norm,
                                  num_heads=args.num_heads,
                                  macro_func=args.macro_func)
        gnn_out_dim = self.hgnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)
        if args.use_same_standard:
            import torch.nn as nn
            self.relation_params = nn.ModuleList()
            self.link_relation_dict = {}
            from graphnas.same_standard.dill_with_graph import Matcher
            from graphnas.same_standard.dill_with_graph import NodeEncoder
            for relation_type in hg.etypes:
                matcher = Matcher(gnn_out_dim, gnn_out_dim)
                # self.neg_queue[source_type][relation_type] = torch.FloatTensor([]).to(device)
                self.link_relation_dict[relation_type] = matcher
                self.relation_params.append(matcher)
            self.node_encoder = {}
            for node_type in hg.ntypes:
                encoder = NodeEncoder(gnn_out_dim, gnn_out_dim)
                self.node_encoder[node_type] = encoder
                self.relation_params.append(encoder)
    # this function also need to make some change.
    def forward(self, hg, h_dict):
        with hg.local_scope():
            hg = self.hg
            h_dict = {key: value for key, value in h_dict.items() if key in hg.ntypes}
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if hasattr(self, 'hgnn'):
                h_dict = self.hgnn(hg, h_dict)
            if not self.args.use_same_standard:
                if hasattr(self, 'post_mp'):
                    out_h = {}
                    # I think here the h_dict is what I want
                    for key, value in h_dict.items():
                        if key in self.out_node_type:
                            out_h[key] = value
                    out_h = self.post_mp(out_h)
            # here is what we need to write
            else:
                # I think here the h_dict is what I want
                out_h = h_dict
        return out_h

    def my_forward(self, hg, h_dict):
        with hg.local_scope():
            hg = self.hg
            h_dict = {key: value for key, value in h_dict.items() if key in hg.ntypes}
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if hasattr(self, 'hgnn'):
                h_dict = self.hgnn(hg, h_dict)
            if hasattr(self, 'post_mp'):
                out_h = {}
                # I think here the h_dict is what I want
                for key, value in h_dict.items():
                    if key in self.out_node_type:
                        out_h[key] = value
                out_h = self.post_mp(out_h)
            # here is what we need to write
        return out_h