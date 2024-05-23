import torch


class MacroSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            # Define operators in search space
            self.search_space = {
                "model_family": ['homo_GNN', 'relation', 'metapath'],  # 'general_HGNN' for the last two
                "gnn_type": ['gcnconv', 'gatconv', 'ginconv', 'sageconv'],  # Micro-level aggr
                "macro_func": ['attention', 'sum', 'mean', 'max'],  # Macro-level aggr
                # the following four lines
                "has_bn_l2norm": [1, 2, 3, 4],  # bn and l2norm TT,TF,FT,FF
                # "dropout": [0.0],  # drop out
                "act-layers-post": [1, 2, 3, 4, 5, 6],
                # "activation": ['lrelu', 'elu', 'tanh'],  # activation
                # "layers_post_mp": [1, 2],  # post-process layer
                # "has_l2norm": [True, False],  # l2

                # the following four lines
                "stage_type_hidden_dim": [1, 2, 3, 4],
                # "stage_type": ['skipsum', 'skipconcat'],  # layer connectivity
                # "layers_pre_mp": [1],  # pre-process layer
                "layers_gnn": [1, 2, 3, 4, 5, 6],  # message passing layer
                "dropout": [0.0, 0.3]
                # the following four lines
                # "optimizer": ['Adam'],  # optimizer
                # "lr": [0.01],  # learning rate
                # "max_epoch": [400],  # train epoch
                # "hidden_dim": [64, 128],  # hidden dimension
            }

    def get_search_space(self):
        return self.search_space

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


if __name__ == '__main__':
    test = MacroSearchSpace()
    print(test.generate_action_list())
