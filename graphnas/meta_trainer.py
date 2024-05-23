import argparse
import glob
import os
import time

import numpy as np
import scipy.signal
# from scipy import signal
import torch
import yaml

import graphnas.utils.tensor_utils as utils

from openhgnn.utils import Logger


# here the ::-1 means reverse.
def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def _get_optimizer():
    # if name.lower() == 'sgd':
    #     optim = torch.optim.SGD
    # elif name.lower() == 'adam':
    optim = torch.optim.Adam
    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.max_length = self.args.shared_rnn_max_length

        self.with_retrain = False
        self.submodel_manager = None
        self.controller = None
        self.space_args = None
        self.dataset_class = None
        self.logger = None
        # self.build_model()  # build controller and sub-model
        #
        # controller_optimizer = _get_optimizer(self.args.controller_optim)
        # self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        if self.args.mode == "derive":
            self.load_model()

    def scale(self, value, scale_value=1.):
        '''
        scale value into [-scale_value, scale_value], according to best reward
        '''
        # print("history", history)
        max_reward = self.best_history[0]
        print(max_reward)
        if max_reward == 0:
            return value
        temp_value = scale_value / max_reward * np.array(value)
        print("temp_value", temp_value)
        # if temp_value > scale_value:
        #     temp_value = scale_value
        # elif temp_value < -scale_value:
        #     temp_value = -scale_value
        return [temp_value]

    def get_dataset(self, dataset_list):
        _dict = {}
        # dataset,seed,
        self.space_args.seed = 1
        from openhgnn.dataset import build_dataset
        for name in dataset_list:
            if name in ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBn-Freebase', 'HNE-PubMed']:
                self.space_args.task = "node_classification"
            elif name in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']:
                self.space_args.task = "link_prediction"
            else:
                self.space_args.task = "recommendation"
            self.space_args.dataset = name
            _dict[name] = build_dataset(self.space_args)
        # print(_dict)
        return _dict

    def build_model(self):
        self.args.share_param = False  # this share_param is defined by this function
        self.with_retrain = True
        self.args.shared_initial_step = 0
        if self.args.search_mode == "macro":
            # generate model description in macro way (generate entire network description)
            from graphnas.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            # this action list return the key of the paras*2
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # build RNN controller
            from graphnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)

            # if self.args.dataset in ["cora", "citeseer", "pubmed"]:
            #     # implements based on dgl
            #     self.submodel_manager = CitationGNNManager(self.args)
            # if self.args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            #     # implements based on pyg
            #     self.submodel_manager = GeoCitationManager(self.args)

        if self.args.search_mode == "micro":
            self.args.format = "micro"
            self.args.predict_hyper = True
            if not hasattr(self.args, "num_of_cell"):
                self.args.num_of_cell = 2
            from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
            search_space_cls = IncrementSearchSpace()
            search_space = search_space_cls.get_search_space()
            from graphnas.graphnas_controller import SimpleNASController
            from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
            self.submodel_manager = MicroCitationManager(self.args)
            self.search_space = search_space
            action_list = search_space_cls.generate_action_list(cell=self.args.num_of_cell)
            if hasattr(self.args, "predict_hyper") and self.args.predict_hyper:
                self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
            else:
                self.action_list = action_list
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            if self.cuda:
                self.controller.cuda()

        if self.cuda:
            self.controller.cuda()

    def build_controller_model(self):
        # the macro are the same with it.
        self.args.share_param = False  # this share_param is defined by this function
        self.with_retrain = True
        self.args.shared_initial_step = 0
        if self.args.search_mode == "macro":
            # generate model description in macro way (generate entire network description)
            from graphnas.search_space import MacroSearchSpace
            search_space_cls = MacroSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            # this action list return the key of the paras*2
            self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)
            # build RNN controller
            from graphnas.graphnas_controller import SimpleNASController
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)

        if self.args.search_mode == "micro":
            self.args.format = "micro"
            self.args.predict_hyper = True
            if not hasattr(self.args, "num_of_cell"):
                self.args.num_of_cell = 2
            from graphnas_variants.micro_graphnas.micro_search_space import IncrementSearchSpace
            search_space_cls = IncrementSearchSpace()
            search_space = search_space_cls.get_search_space()
            from graphnas.graphnas_controller import SimpleNASController
            from graphnas_variants.micro_graphnas.micro_model_manager import MicroCitationManager
            self.submodel_manager = MicroCitationManager(self.args)
            self.search_space = search_space
            action_list = search_space_cls.generate_action_list(cell=self.args.num_of_cell)
            if hasattr(self.args, "predict_hyper") and self.args.predict_hyper:
                self.action_list = action_list + ["learning_rate", "dropout", "weight_decay", "hidden_unit"]
            else:
                self.action_list = action_list
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda)
            if self.cuda:
                self.controller.cuda()

        if self.cuda:
            self.controller.cuda()
        return self.controller

    def form_gnn_info(self, gnn, type):
        dict = {
            "model_family": gnn[0],  # 'general_HGNN' for the last two
            "gnn_type": gnn[1],  # Micro-level aggr
            "macro_func": gnn[2],  # Macro-level aggr
            # the following four lines
            # "has_bn_l2norm": gnn[3],  # bn and l2norm TT,TF,FT,FF
            "dropout": gnn[7],  # drop out
            # "activation": gnn[4],  # activation
            # "has_l2norm": [True, False],  # l2

            # the following four lines
            # "stage_type": gnn[5],  # layer connectivity
            "layers_pre_mp": 1,  # pre-process layer
            "layers_gnn": gnn[6],  # message passing layer
            # "layers_post_mp": gnn[7],  # post-process layer

            # the following four lines
            "optimizer": 'Adam',  # optimizer
            "lr": 0.01,  # learning rate
            "max_epoch": 100,  # train epoch
            # "hidden_dim": gnn[8],  # hidden dimension
            "mini_batch_flag": False,
            "weight_decay": 0.0001,
            "patience": 40,
            "num_heads": 4,  # [1, 2, 4, 8]
            'feat': 0,
            # "featn": 0,  # [0, 1, 2]
            # "featl": 0,  # [0, 2]
            "loss_fn": None  # 'dot-product'
        }
        if gnn[3] == 1:
            dict['has_bn'] = True
            dict['has_l2norm'] = True
        elif gnn[3] == 2:
            dict['has_bn'] = True
            dict['has_l2norm'] = False
        elif gnn[3] == 3:
            dict['has_bn'] = False
            dict['has_l2norm'] = True
        else:
            dict['has_bn'] = False
            dict['has_l2norm'] = False
        if type == 'link':
            dict["score_fn"] = 'distmult'  # distmult,dot-product
            dict["feat"] = 2
        if gnn[4] in [1, 4]:
            dict['activation'] = 'lrelu'
        elif gnn[4] in [2, 5]:
            dict['activation'] = 'elu'
        else:
            dict['activation'] = 'tanh'
        if gnn[4] in [1, 2, 3]:
            dict['layers_post_mp'] = 1
        else:
            dict['layers_post_mp'] = 2
        if gnn[5] in [1, 2]:
            dict['stage_type'] = 'skipsum'
        else:
            dict['stage_type'] = 'skipconcat'
        if gnn[5] in [1, 3]:
            dict['hidden_dim'] = 64
        else:
            dict['hidden_dim'] = 128
        return dict

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        # load the datasets.
        # TODO
        # maybe we just use the first two dataset.
        datasets_node = ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBn-Freebase', 'HNE-PubMed']
        datasets_link = ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']
        datasets_rec = ['yelp4HeGAN', 'DoubanMovie']
        # 'HGBn-Freebase' not enough memeory
        # train_tasks = ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-ACM',
        #                'HGBl-DBLP']

        train_tasks = ['yelp4HeGAN']
        # 'HGBn-ACM', 'HGBn-IMDB','HGBn-DBLP' 'HGBl-amazon', 'HGBl-LastFM', 'HGBl-DBLP',
        # 'HGBl-IMDB','yelp4HeGAN','aifb'
        test_tasks = ['HGBn-DBLP']  # 'HGBl-ACM'

        print('Meta training tasks {0} :{1}'.format(len(train_tasks), train_tasks))
        print('Meta testing  tasks {0} :{1}'.format(len(test_tasks), test_tasks))

        # create the model of controller.
        import learn2learn as l2l
        model = self.build_controller_model()
        maml = l2l.algorithms.MAML(model, lr=self.args.adapt_lr, first_order=True, allow_unused=False)
        controller_optimizer = _get_optimizer()
        self.controller_optim = controller_optimizer(maml.parameters(), lr=self.args.meta_lr)

        # the following are prepare the args for the logger args
        space_parser = argparse.ArgumentParser()
        space_parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
        # space_parser.add_argument('--subgraph_extraction', '-u', default='metapath', type=str,
        #                           help='subgraph_extraction of models')
        space_parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
        space_parser.add_argument('--dataset', '-d', default='HGBl-PubMed', type=str, help='name of datasets')
        space_parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
        space_parser.add_argument('--repeat', '-r', default='3', type=int, help='-1 means cpu')
        space_parser.add_argument('--gnn_type', '-a', default='gcnconv', type=str, help='aggregation type')
        space_parser.add_argument('--times', '-s', default=1, type=int, help='which yaml file')
        # it seems that the times arg control the configure of the model
        space_parser.add_argument('--key', '-k', default='dataset_name', type=str, help='attribute')
        space_parser.add_argument('--value', '-v', default='True', type=str, help='value')
        space_parser.add_argument('--configfile', '-c', default='config', type=str,
                                  help='The file path to load the configuration.')
        space_parser.add_argument('--yamlfile', '-ui', default='config', type=str,
                                  help='The file path to load the configuration.')
        # is seems that the configfile arg control the really path to the yaml file
        space_parser.add_argument('--predictfile', '-p', default='predict', type=str,
                                  help='The file path to store predict files.')
        self.space_args = space_parser.parse_args()

        # self.dataset_class = self.get_dataset(train_tasks)  # return a dict
        ## end of it

        def random_dic(task_lists):
            import random
            random.shuffle(task_lists)
            return task_lists

        # self.args.derive_meta_test = False
        if not self.args.derive_meta_test:  # meta-training
            # self.args.dataset = 'meta_train'
            # Meta Training
            # here we meta train 10 times
            self.space_args.Ismetatrain = "meta_train"
            self.logger = Logger(self.space_args)
            self.space_args.logger = self.logger
            for self.epoch in range(self.start_epoch, self.args.max_epoch):
                self.logger.info("*" * 35 + "Meta Training " + str(self.epoch) + " Epoch" + "*" * 35)
                maml.train()  # meta-train,open dropout and batch normalization
                # Random  all graph data sets.

                meta_train_tasks = random_dic(train_tasks)
                meta_train_loss = 0.0  # cuda
                # Training all controllers on different data sets.
                for name in meta_train_tasks:
                    self.best_args = None
                    self.logger.info("task  name:" + name)
                    # copy a controller
                    learner = maml.clone(first_order=True)
                    query_loss = self.meta_train_controller(name, learner)
                    meta_train_loss += query_loss
                    torch.cuda.empty_cache()

                meta_train_loss = meta_train_loss / len(meta_train_tasks)

                self.logger.info(50 * '-')
                self.logger.info('Meta Train Loss:' + str(meta_train_loss))
                self.logger.info(50 * '-')

                self.controller_optim.zero_grad()
                meta_train_loss.backward()
                self.controller_optim.step()

                torch.cuda.empty_cache()

                # if self.epoch % self.args.save_epoch == 0:
                #    self.save_model()

            # save the model
            def get_local_time():
                r"""Get current time

                Returns:
                    str: current time
                """
                import datetime
                cur = datetime.datetime.now()
                cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

                return cur

            save_path = 'model_parameters/meta_train_epoch{0}_adapt_steps{1}_{2}'.format(self.args.max_epoch,
                                                                                         self.args.adapt_steps,
                                                                                         get_local_time())
            utils.makedirs(save_path)
            torch.save(maml.state_dict(), save_path + '/model_params.pkl')
            print("save sucesss")
            print(save_path)
            # print("maml.state_dict()", maml.state_dict())
            # print("@"*30)
            # print("self.controller_optim.state_dict()", self.controller_optim.state_dict())
            torch.save(self.controller_optim.state_dict(), save_path + '/optimizer.pkl')
            self.logger.info('meta train is over!!!')
            return

        # meta test
        if self.args.derive_meta_test:
            self.args.load_model = True
            self.space_args.Ismetatrain = "meta_test"
            self.logger = Logger(self.space_args)
            self.space_args.logger = self.logger
            self.best_args = None
            self.yaml_file_dict = {}
            if self.args.load_model:
                self.logger.info("*" * 35 + "Loading Model " + "*" * 35)
                save_path = '../model_parameters/meta_train_epoch10_adapt_steps5_Mar-20-2023_18-15-25'
                # save_path = 'model_parameters/only_predictor'  # only predictor / GraphNAS-predictor
                controller_path = save_path + '/model_params.pkl'
                controller_optimizer_path = save_path + '/optimizer.pkl'

                maml.load_state_dict(torch.load(controller_path))
                # print("maml.state_dict()", maml.state_dict()['module.lstm.weight_ih'][0])
                self.controller_optim.load_state_dict(torch.load(controller_optimizer_path))
                self.logger.info(f'[*] LOADED: {controller_path}')

            #  Record time
            begain_time = time.time()
            end_time = 0
            best_time = 0

            #  adapt_meta_test_steps=100
            self.args.adapt_steps = 500  # 100 i think it is not need 1000
            for name in test_tasks:
                # self.args.dataset = 'meta_test_' + name
                self.logger.info("task  name:" + name)
                # 1. Training the controller parameters theta
                learner = maml.clone(first_order=True)
                self.args.predict = False  # decide use or no use predictive model
                if self.args.predict is True:  # use predictor
                    # _, best_time = self.train_controller_kl(name, learner)
                    _, best_time = self.train_controller(name, learner)
                else:  # without predictor
                    _ = self.meta_train_controller(name, learner)

                end_time = time.time()
                self.logger.info("Finding the best arch time:{0}".format(
                    round(end_time - begain_time, 4)))  # the top-1 arch is the best one
                if best_time != 0:
                    self.logger.info("Finding best structure time:{0}".format(round(end_time - best_time, 4)))

                # # re-evaluate top-k(=1) archs  from history to select the best one
                # self.select_best_archs_from_history(k=1)

                # # 2. Derive architectures
                # # re-sample architectures（10）
                # self.derive(sample_num=self.args.derive_num_sample)
                # if self.args.derive_finally:
                #     best_actions = self.derive()
                #     print("best structure:" + str(best_actions))

            self.logger.info('meta test is over!!!')
            self.logger.info('best reward is:{0}'.format(self.best_args))
            self.logger.info('yaml_file is:{0}'.format(self.yaml_file_dict))
        # if self.args.derive_finally:
        #     best_actions = self.derive()
        #     print("best structure:" + str(best_actions))
        # self.save_model()

    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = self.submodel_manager.train(gnn, format=self.args.format)
                self.logger.info(f"{gnn}, val_score:{val_score}")
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)

    def generate_yaml(self, dict, dataset_name, gnn_type, type):
        dicts = {}
        dicts[dataset_name] = dict
        if type == "node":
            aproject = {'node_classification': dicts
                        }
        elif type == "link":
            aproject = {'link_prediction': dicts
                        }
        else:
            aproject = {"recommendation": dicts
                        }
        fileNamePath = os.path.split(os.path.realpath(__file__))[0]
        path = fileNamePath + '/config/{}'.format("meta_train")
        if not os.path.exists(path):
            os.makedirs(path)
        path = '{}/{}/{}'.format(path, "dataset_name", dataset_name)
        if not os.path.exists(path):
            os.makedirs(path)
        curr_time = str(time.strftime('%Y_%m-%d_%H-%M-%S'))
        name = gnn_type + '_' + curr_time + '.yaml'
        yamlPath = os.path.join(path, name)
        with open(yamlPath, 'w') as f:
            yaml.dump(aproject, f)
            print('Generate yaml file successfully!')
        return yamlPath

    def identify_is_max(self, list, dict, num, yamlfile):
        for val in list:
            # print("compare",num,val)
            if float(num) > val:
                list.remove(val)
                list.append(num)
                list.sort()
                self.logger.info("here is list:{}".format(list))
                # if val != 0:
                #     del (dict[val])
                dict[num] = yamlfile
                self.logger.info("here is dict:{}".format(dict))
                return True
        return False

    def get_reward(self, gnn_list, entropies, hidden, dataset_name):
        """
        Computes the reward of a single sampled model on validation data.
        """
        # it seems that the following four line are useless
        # if not isinstance(entropies, np.ndarray):
        #     entropies = entropies.data.cpu().numpy()
        # # if isinstance(gnn_list, dict):
        # #     gnn_list = [gnn_list]
        # if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
        #     pass
        # else:
        #     gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        if dataset_name in ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBn-Freebase', 'HNE-PubMed', 'yelp4HeGAN', 'aifb']:
            task_type = 'node'
        elif dataset_name in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']:
            task_type = 'link'
        else:
            task_type = "recommendation"
        for gnn in gnn_list:
            gnn_dict = self.form_gnn_info(gnn, task_type)
            yaml_file = self.generate_yaml(gnn_dict, dataset_name, gnn_dict["gnn_type"], task_type)
            # here we need to return the reward, and it is one dimension
            from graphnas.space4hgnn import Space4HGNN
            from graphnas.utils.space_utils import read_config
            # space_args = space_parser.parse_args()
            if gnn_dict["model_family"] == 'homo_GNN':
                self.space_args.model = 'homo_GNN'
            elif gnn_dict["model_family"] == 'relation':
                self.space_args.model = 'general_HGNN'
                self.space_args.subgraph_extraction = 'relation'
            elif gnn_dict["model_family"] == 'metapath':
                self.space_args.model = 'general_HGNN'
                if dataset_name in ['HGBl-amazon', 'HGBl-DBLP']:
                    self.space_args.subgraph_extraction = 'relation'
                else:
                    self.space_args.subgraph_extraction = 'metapath'
            else:
                print('something wrong')
            if task_type == 'node':
                self.space_args.task = "node_classification"
            elif task_type == 'link':
                self.space_args.task = "link_prediction"
            elif task_type == 'recommendation':
                self.space_args.task = "recommendation"
            else:
                print('something wrong.')

            self.space_args.dataset = dataset_name
            self.space_args.gnn_type = gnn_dict["gnn_type"]
            self.space_args.yamlfile = yaml_file
            self.space_args.configfile = "meta_train"
            self.space_args.value = dataset_name
            self.space_args.use_same_standard = self.args.use_same_standard
            if self.space_args.Ismetatrain == "meta_test":
                self.space_args.use_same_standard = False
            if self.space_args.use_same_standard:
                if hasattr(self.space_args, 'subgraph_extraction'):
                    self.space_args.subgraph_extraction = 'relation'
            self.space_args = read_config(self.space_args)
            self.space_args.sample_batch_size = self.args.sample_batch_size
            # self.space_args.sperate_task = self.dataset_class[dataset_name]
            reward = Space4HGNN(args=self.space_args)
            # if reward is None:  # cuda error hanppened
            #     reward = 0
            # else:
            #     reward = reward[1]

            reward_list.append(reward)
            if self.space_args.Ismetatrain == "meta_test":
                if self.best_args == None:
                    self.best_args = [0., 0., 0., 0., 0., 0., 0., 0., 0., float(reward)]
                    self.yaml_file_dict = {float(reward): yaml_file}
                    self.logger.info("here is list:{}".format(self.best_args))
                    self.logger.info("here is list:{}".format(self.yaml_file_dict))
                else:
                    self.identify_is_max(self.best_args, self.yaml_file_dict, reward, yaml_file)
            else:
                if self.best_args == None:
                    self.best_args = [float(reward)]
                else:
                    self.best_args.append(float(reward))
        if self.args.entropy_mode == 'reward':
            rewards = [None] * len(reward_list)
            for dex in range(len(reward_list)):
                rewards[dex] = reward_list[dex] + self.args.entropy_coeff * entropies[dex]
                rewards[dex] = rewards[dex].tolist()
            # rewards = reward_list + self.args.entropy_coeff * entropies

        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0
        for step in range(self.args.controller_max_step):
            # sample graphnas
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=1)
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)

    def meta_train_controller(self, name, learner):
        """
            Train controller to find better structure.
        """
        self.logger.info("*" * 35 + "Dataset:{0}, training controller {1} times".format(name, self.args.adapt_steps) +
                         "*" * 35)
        # model = self.controller
        # model.train()
        self.baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []
        self.best_history = None
        hidden = learner.init_hidden(self.args.batch_size)
        total_loss = 0

        learner.train()  # meta-train,open dropout and batch normalization

        for step in range(self.args.adapt_steps):  # adaptation steps.
            self.logger.info("current step is " + str(step))
            if self.space_args.Ismetatrain == "meta_test":
                _ = self.meta_train_controller_by_trajectory(learner, hidden, adv_history, entropy_history,
                                                             reward_history, total_loss, name, is_adapt=True)
            else:
                _ = self.meta_train_controller_by_trajectory(learner, hidden, adv_history, entropy_history,
                                                             reward_history, total_loss, name, is_adapt=True)
            torch.cuda.empty_cache()

        self.logger.info("*" * 35 + "Dataset:{0}, training controller over".format(name) + "*" * 35)

        learner.eval()
        # compute loss on meta training loss.
        query_loss = self.meta_train_controller_by_trajectory(learner, hidden, adv_history, entropy_history,
                                                              reward_history, total_loss, name, is_adapt=False)
        return query_loss

    def meta_train_controller_by_trajectory(self, learner, hidden, adv_history, entropy_history,
                                            reward_history, total_loss, dataset_name, is_adapt):
        structure_list, log_probs, entropies = learner.sample(dataset_name, with_details=True)
        # this np_entropies is used to update the policy
        np_entropies = entropies.data.cpu().numpy()
        results = self.get_reward(structure_list, np_entropies, hidden, dataset_name)
        torch.cuda.empty_cache()

        if results:  # has reward
            rewards, hidden = results
        else:
            return
            # continue  # CUDA Error happens, drop structure and step into next iteration

        # discount
        # print("dicount", self.args.discount)
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount)

        reward_history.extend(rewards)

        entropy_history.extend(np_entropies)
        # print("baseline ", self.baseline)
        # moving average baseline
        if self.baseline is None:
            self.baseline = [rewards[0] * 0.995]
        else:
            # here I want to rewrite it.
            sum_rewards = 0.
            temp_sum_num = len(self.best_args)
            for val in self.best_args:
                if val != 0.:
                    sum_rewards += val
                else:
                    temp_sum_num -= 1
            self.baseline = [sum_rewards / temp_sum_num]
            # decay = self.args.ema_baseline_decay
            # # self.baseline = decay * self.baseline[0] + (1 - decay) * rewards[0]
            #
            # ta = decay * np.array(self.baseline)
            # tb = (1 - decay) * np.array(rewards)
            # self.baseline = list(ta + tb)
        self.logger.info("before calculate{0},{1},{2}".format(rewards, self.baseline, self.baseline[0]))
        adv = [i - j for i, j in zip(rewards, self.baseline)]
        self.logger.info("after calculate" + str(adv))
        # advv = [rewards[0][i] - self.baseline[0][i] for i in range(len(rewards[0]))]
        history.append(adv)
        if self.best_history == None:
            self.best_history = adv
        elif self.best_history < adv:
            self.best_history = adv
        else:
            self.best_history = [self.best_history[0] * 0.995]
        adv = self.scale(adv, scale_value=10.)
        # self.logger.info("after scale" + str(adv))
        adv_history.extend(adv)

        adv = utils.get_variable(adv, self.cuda, requires_grad=False)

        # policy loss
        loss = -log_probs * adv
        if self.args.entropy_mode == 'regularizer':
            loss -= self.args.entropy_coeff * entropies

        loss = loss.sum()  # or loss.mean()
        self.logger.info("training loss is:{}".format(loss))
        if is_adapt:
            # print('loss:', loss)
            learner.adapt(loss)
            # print("adapt")
            return
        else:
            return loss

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)
        results = self.submodel_manager.retrain(gnn, format=self.args.format)
        if results:
            reward, scores = results
        else:
            return

        self.logger.info(f'eval | {gnn} | reward: {reward:8.2f} | scores: {scores:8.2f}')

    def derive_from_history(self):
        with open(self.args.dataset + "_" + self.args.search_mode + self.args.submanager_log_file, "a") as f:
            lines = f.readlines()

        results = []
        best_val_score = "0"
        for line in lines:
            actions = line[:line.index(";")]
            val_score = line.split(";")[-1]
            results.append((actions, val_score))
        results.sort(key=lambda x: x[-1], reverse=True)
        best_structure = ""
        best_score = 0
        for actions in results[:5]:
            actions = eval(actions[0])
            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)
            val_scores_list = []
            for i in range(20):
                val_acc, test_acc = self.submodel_manager.evaluate(actions)
                val_scores_list.append(val_acc)

            tmp_score = np.mean(val_scores_list)
            if tmp_score > best_score:
                best_score = tmp_score
                best_structure = actions

        print("best structure:" + str(best_structure))
        # train from scratch to get the final score
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        test_scores_list = []
        for i in range(100):
            # manager.shuffle_data()
            val_acc, test_acc = self.submodel_manager.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure

    def derive(self, sample_num=None):
        """
        sample a serial of structures, and return the best structure.
        """
        if sample_num is None and self.args.derive_from_history:
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample

            gnn_list, _, entropies = self.controller.sample(sample_num, with_details=True)

            max_R = 0
            best_actions = None
            filename = self.model_info_filename
            for action in gnn_list:
                gnn = self.form_gnn_info(action)
                reward = self.submodel_manager.test_with_param(gnn, format=self.args.format,
                                                               with_retrain=self.with_retrain)

                if reward is None:  # cuda error hanppened
                    continue
                else:
                    results = reward[1]

                if results > max_R:
                    max_R = results
                    best_actions = action

            self.logger.info(f'derive |action:{best_actions} |max_R: {max_R:8.6f}')
            self.evaluate(best_actions)
            return best_actions

    @property
    def model_info_filename(self):
        return f"{self.args.dataset}_{self.args.search_mode}_{self.args.format}_results.txt"

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        self.logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            self.logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        self.logger.info(f'[*] LOADED: {self.controller_path}')
