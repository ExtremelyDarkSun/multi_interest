import argparse
import os
import random
import shutil
import numpy as np
import torch

from torch.utils.data import DataLoader
from DNN import DNN
from Pop import Pop
from GRU4Rec import GRU4Rec
from MIND import MIND
from ComiRec import ComiRec_DR, ComiRec_SA
from REMI import REMI
from DisMIR import DisMIR
from DASD_DisMIR import DASD_DisMIR


def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='train', help='train | test') # train or test or output
    parser.add_argument('--dataset', type=str, default='book', help='book | taobao') # 数据集
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--hidden_size', type=int, default=64) # 隐藏层维度、嵌入维度
    parser.add_argument('--interest_num', type=int, default=4) # 兴趣的数量
    parser.add_argument('--model_type', type=str, default='MIND', help='DNN | GRU4Rec | MIND | ..') # 模型类型
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate') # 学习率
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=30, help='(k), the number of steps after which the learning rate decay')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)') # 最大迭代次数，单位是k（1000）
    parser.add_argument('--loss_print_interval', type=int, default=100, help='print detailed loss every N iterations')
    parser.add_argument('--patience', type=int, default=50) # patience，用于early stopping
    parser.add_argument('--topN', type=int, default=50) # default=50
    parser.add_argument('--gpu', type=str, default=None) # None -> cpu
    parser.add_argument('--coef', default=None) # 多样性，用于test
    parser.add_argument('--exp', default='e1')
    parser.add_argument('--add_pos', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--sampled_n', type=int, default=1280)
    parser.add_argument('--sampled_loss', type=str, default='sampled')
    parser.add_argument('--sample_prob', type=int, default=0)

    # For REMI
    parser.add_argument('--rbeta', type=float, default=0)
    parser.add_argument('--rlambda', type=float, default=0)

    # [DisMIR] Disentangled Multi-interest Representation Learning
    # [PAPER_REF] Du et al. KDD '24
    parser.add_argument('--dlambda', type=float, default=0.1,
                        help='[DisMIR] Trade-off coefficient for partition loss [EXP_REF] [0.01,0.1,1.0]')
    parser.add_argument('--partition_groups', type=int, default=64,
                        help='[DisMIR] Number of item partition groups K [EXP_REF] default 64')
    parser.add_argument('--num_negatives', type=int, default=100,
                        help='[DisMIR] Number of negative samples for contrastive learning [EXP_REF] default 100')
    parser.add_argument('--hard_neg_candidates', type=int, default=10,
                        help='[DisMIR] Number of candidates for hard negative mining [PAPER_REF] default 10')

    # [Overlapped Partition] Soft partition assignment [PAPER_REF] Sec 4.1.2
    # NOTE: Paper uses raw embeddings as w_ik (no softmax, no temperature)
    parser.add_argument('--use_overlapped_partition', type=int, default=0,
                        help='[DisMIR] Use overlapped partition (0/1) [PAPER_REF] Sec 4.1.2, default 0 (non-overlapped)')

    # [P1_FIX] Additional hyperparameters for DisMIR training stability
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='[DisMIR] Temperature for contrastive learning in partition loss [P1_FIX] default 0.1')
    parser.add_argument('--use_layer_norm', type=int, default=1,
                        help='[DisMIR] Whether to use LayerNorm in capsule network [P1_FIX] 0/1')

    # [DASD-DisMIR] Knowledge Distillation hyperparameters
    # [DASD_REF] DASD_REMI default values
    parser.add_argument('--lambda_recon', type=float, default=0.1,
                        help='[DASD-DisMIR] Teacher MSE reconstruction loss weight (pretrain & finetune), default 0.1')
    parser.add_argument('--lambda_select', type=float, default=1.0,
                        help='[DASD-DisMIR] Select BPR distillation loss weight (finetune only), default 1.0')
    parser.add_argument('--lambda_bpr', type=float, default=1.0,
                        help='[DASD-DisMIR] DisMIR direct recommendation BPR loss weight (finetune floor signal), default 1.0')
    parser.add_argument('--lambda_diversity', type=float, default=0.01,
                        help='[DASD-DisMIR] Diversity entropy loss weight for interest selector (finetune only), default 0.01')
    parser.add_argument('--lambda_align', type=float, default=0.1,
                        help='[DASD-DisMIR] Alignment (Chamfer) loss weight [DASD_REF] default 0.1')
    parser.add_argument('--lambda_infonce', type=float, default=0.1,
                        help='[DASD-DisMIR] InfoNCE contrastive loss weight [DASD_REF] default 0.1')

    # [Partition-Aware DASD-DisMIR] Partition enhancement hyperparameters
    parser.add_argument('--lambda_partition_align', type=float, default=0.3,
                        help='[Partition-Aware DASD-DisMIR] Partition alignment loss weight, default 0.3')
    parser.add_argument('--partition_temperature', type=float, default=0.5,
                        help='[Partition-Aware DASD-DisMIR] Temperature for partition enhancement (<1 for sharper), default 0.5')
    parser.add_argument('--partition_align_temperature', type=float, default=1.0,
                        help='[Partition-Aware DASD-DisMIR] Temperature for partition alignment, default 1.0')

    # [Teacher Pretrain] Two-stage training for DASD-DisMIR
    # 0: joint training (default, original behaviour)
    # 1: pretrain Teacher only, save weights to best_model/{exp_name}_teacher/
    # 2: load pretrained Teacher weights then run normal joint training
    parser.add_argument('--pretrain', type=int, default=0,
                        help='[DASD-DisMIR] Teacher pretrain stage: 0=joint(default), 1=pretrain Teacher, 2=joint with pretrained Teacher')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='[DASD-DisMIR] Path to pretrained teacher checkpoint for stage-2 (default: auto-generate from exp_name)')

    # [Multi-Target Distillation] Extra future labels for DASD-DisMIR alignment
    parser.add_argument('--num_future_labels', type=int, default=2,
                        help='[DASD-DisMIR] Number of extra future-click targets for multi-target Teacher distillation (default: 2, total 3 targets)')

    return parser


class DataIterator(torch.utils.data.IterableDataset):

    def __init__(self, source,
                 batch_size=128,
                 seq_len=100,
                 train_flag=1,
                 time_span = 128,
                 num_future_labels = 2
                ):
        print("Using time span", time_span)
        self.read(source) # 读取数据，获取用户列表和对应的按时间戳排序的物品序列，每个用户对应一个物品list
        self.users = list(self.users) # 用户列表

        self.time_span = time_span
        self.batch_size = batch_size # 用于训练
        self.eval_batch_size = batch_size # 用于验证、测试
        self.train_flag = train_flag # train_flag=1表示训练
        self.seq_len = seq_len # 历史物品序列的最大长度
        self.index = 0 # 验证和测试时选择用户的位置的标记
        self.num_future_labels = num_future_labels  # [Multi-Target] Number of extra future labels
        print("total user:", len(self.users))
        print("total items:", len(self.items))

    def __iter__(self):
        return self
    
    # def next(self):
    #     return self.__next__()

    def read(self, source):
        self.graph = {} # key:user_id，value:一个list，放着该user_id所有(item_id,time_stamp)元组，排序后value只保留item_id
        self.time_graph = {}
        self.users = set()
        self.items = set()
        self.times = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if len(conts) == 3:
                    time_stamp = int(conts[2])
                else:
                    idx = int(conts[2])
                    time_stamp = int(conts[3])
                self.users.add(user_id)
                self.items.add(item_id)
                self.times.add(time_stamp)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items(): # 每个user的物品序列按时间戳排序
            value.sort(key=lambda x: x[1])
            time_list = list(map(lambda x: x[1], value))
            time_min = min(time_list)
            # self.graph[user_id] = list(map(lambda x: [x[0], ], items))
            self.graph[user_id] = [x[0] for x in value] # 排序后只保留了item_id
            self.time_graph[user_id] = [int(round((x[1] - time_min) / 86400.0) + 1) for x in value] # 排序后只保留了item_id
        self.users = list(self.users) # 用户列表
        self.items = list(self.items) # 物品列表

    def compute_time_matrix(self, time_seq, item_num):
        time_matrix = np.zeros([self.seq_len, self.seq_len], dtype=np.int32)
        for i in range(item_num):
            for j in range(item_num):
                span = abs(time_seq[i] - time_seq[j])
                if span > self.time_span:
                    time_matrix[i][j] = self.time_span
                else:
                    time_matrix[i][j] = span
        return time_matrix.tolist()

    def compute_adj_matrix(self, mask_seq, item_num):
        node_num = len(mask_seq)

        adj_matrix = np.zeros([node_num, node_num + 2], dtype=np.int32)

        adj_matrix[0][0] = 1
        adj_matrix[0][1] = 1
        adj_matrix[0][-1] = 1

        adj_matrix[item_num - 1][item_num - 1] = 1
        adj_matrix[item_num - 1][item_num] = 1
        adj_matrix[item_num - 1][-1] = 1

        for i in range(1, item_num - 1):
            adj_matrix[i][i] = 1
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][-1] = 1

        if (item_num < node_num):
            for i in range(item_num, node_num):
                adj_matrix[i][0] = 1
                adj_matrix[i][1] = 1
                adj_matrix[i][-1] = 1

        return adj_matrix.tolist()

    def __next__(self):
        if self.train_flag == 1: # 训练
            user_id_list = random.sample(self.users, self.batch_size) # 随机抽取batch_size个user
        else: # 验证、测试，按顺序选取eval_batch_size个user，直到遍历完所有user
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        hist_time_list = []
        hist_item_list = []
        time_matrix_list = []
        hist_mask_list = []
        adj_matrix_list = []
        future_labels_list = []  # [Multi-Target] extra future targets for each sample

        for user_id in user_id_list:
            item_list = self.graph[user_id] # 排序后的user的item序列
            time_list = self.time_graph[user_id] # 排序后的user的item序列
            # 这里训练和（验证、测试）采取了不同的数据选取方式
            if self.train_flag == 1: # 训练，选取训练时的label
                k = random.choice(range(4, len(item_list))) # 从[4,len(item_list))中随机选择一个index
                item_id_list.append(item_list[k]) # 该index对应的item加入item_id_list

                # [Multi-Target] Build future_labels for this sample
                num_future = self.num_future_labels
                future = []
                # Try to use real future clicks: item_list[k+1], item_list[k+2], ...
                for delta in range(1, num_future + 1):
                    if k + delta < len(item_list):
                        future.append(item_list[k + delta])
                # Fill with recent history items if not enough future clicks
                fill_idx = 1
                while len(future) < num_future:
                    if k - fill_idx >= 0:
                        future.append(item_list[k - fill_idx])
                    else:
                        future.append(item_list[k])  # extreme fallback: repeat current label
                    fill_idx += 1
                future_labels_list.append(future)  # (num_future_labels,)

            else: # 验证、测试，选取该user后20%的item用于验证、测试
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            # k前的item序列为历史item序列
            if k >= self.seq_len: # 选取seq_len个物品
                hist_item_list.append(item_list[k-self.seq_len: k])
                hist_mask_list.append([1.0] * self.seq_len)
                hist_time_list.append(time_list[k-self.seq_len: k])
                time_matrix_list.append(self.compute_time_matrix(time_list[k - self.seq_len: k], self.seq_len))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * self.seq_len, self.seq_len))

            else:
                hist_item_list.append(item_list[:k] + [0] * (self.seq_len - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.seq_len - k))
                hist_time_list.append(time_list[:k] + [0] * (self.seq_len - k))
                time_matrix_list.append(self.compute_time_matrix(time_list[:k] + [0] * (self.seq_len - k), k))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * k + [0.0] * (self.seq_len - k), k))

        # 返回用户列表（batch_size）、物品列表（label）（batch_size）、
        # 历史物品列表（batch_size，seq_len）、历史物品的mask列表（batch_size，seq_len）
        if self.train_flag == 1:
            # Training: also return future_labels_list (batch_size, num_future_labels)
            return user_id_list, item_id_list, hist_item_list, hist_mask_list, (time_matrix_list, adj_matrix_list), future_labels_list
        else:
            return user_id_list, item_id_list, hist_item_list, hist_mask_list, (time_matrix_list, adj_matrix_list)
        # return user_id_list, item_id_list, hist_item_list, hist_mask_list, hist_time_list


def get_DataLoader(source, batch_size, seq_len, train_flag=1, args=None):
    num_future_labels = getattr(args, 'num_future_labels', 2) if args is not None else 2
    dataIterator = DataIterator(source, batch_size, seq_len, train_flag,
                                num_future_labels=num_future_labels)
    return DataLoader(dataIterator, batch_size=None, batch_sampler=None)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# 获取模型
def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, routing_times=3, args=None, device=None):
# def get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, beta, routing_times=3,):
    add_pos = True
    if args:
        add_pos = args.add_pos == 1
    if model_type == 'DNN':
        model = DNN(item_count, hidden_size, batch_size, seq_len)
    elif model_type == 'Pop':
        model = Pop(item_count, hidden_size, batch_size, seq_len, device)
    elif model_type == 'GRU4Rec':
        # todo check this back
        model = GRU4Rec(item_count, hidden_size, batch_size, seq_len, num_layers=args.layers, dropout=args.dropout)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = MIND(item_count, hidden_size, batch_size, interest_num, seq_len, routing_times=routing_times, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        relu_layer = False
        hard_readout = False if dataset == 'kindle' else True
        model = ComiRec_DR(item_count, hidden_size, batch_size, interest_num, seq_len, routing_times=routing_times, relu_layer=relu_layer, hard_readout=hard_readout)
    elif model_type in ['ComiRec-SA']:
        # import pdb; pdb.set_trace()
        model = ComiRec_SA(item_count, hidden_size, batch_size, interest_num, seq_len, add_pos=add_pos, args = args, device = device)
    elif model_type == "REMI":
        model = REMI(item_count, hidden_size, batch_size, interest_num, seq_len, add_pos=add_pos, beta=args.rbeta,
                           args=args, device=device)
    elif model_type == "DisMIR":
        # [DisMIR] Model instantiation
        # [PAPER_REF] Du et al. KDD '24
        # Key hyperparameters from paper experiments:
        # - partition_groups (K): 64, must equal hidden_size for shared representation
        # - interest_num (F): 4 for Gowalla
        # - dlambda (λ): [0.01, 0.1, 1.0] dataset-specific
        # - num_negatives (N_v): 100 for partition loss
        # - hard_neg_candidates: 10 for BPR hard negative mining
        model = DisMIR(item_count, hidden_size, batch_size, interest_num, seq_len,
                       partition_groups=args.partition_groups,
                       lambda_coef=args.dlambda,
                       num_negatives=args.num_negatives,
                       hard_neg_candidates=args.hard_neg_candidates,
                       add_pos=False,
                       beta=args.rbeta,
                       use_overlapped_partition=getattr(args, 'use_overlapped_partition', 0) == 1,
                       args=args,
                       device=device)
    elif model_type == "DASD-DisMIR":
        # [DASD-DisMIR] DASD Knowledge Distillation with DisMIR
        # [DASD_REF] Teacher-Student architecture with ContextGatedTokenizer
        # First create the base DisMIR model (Student)
        dismir_model = DisMIR(item_count, hidden_size, batch_size, interest_num, seq_len,
                              partition_groups=args.partition_groups,
                              lambda_coef=args.dlambda,
                              num_negatives=args.num_negatives,
                              hard_neg_candidates=args.hard_neg_candidates,
                              add_pos=False,
                              beta=args.rbeta,
                              use_overlapped_partition=getattr(args, 'use_overlapped_partition', 0) == 1,
                              args=args,
                              device=device)
        # Wrap with DASD_DisMIR for knowledge distillation
        model = DASD_DisMIR(dismir_model, args)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    model.name = model_type
    return model


# 生成实验名称
def get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=True, exp='e1'):
    extr_name = exp
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(hidden_size), 
                            'len'+str(seq_len), 'in'+str(interest_num), 'top'+str(topN)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('best_model/' + exp_name) and save:
        # flag = input('The exp name already exists. Do you want to cover? (y/n)')
        # if flag == 'y' or flag == 'Y':
        shutil.rmtree('best_model/' + exp_name)
        break
        # else:
        #     extr_name = input('Please input the experiment name: ')
        #     exp_name = para_name + '_' + extr_name

    return exp_name


def save_model(model, Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
    torch.save(model.state_dict(), Path + 'model.pt')


def load_model(model, path):
    model.load_state_dict(torch.load(path + 'model.pt'))
    print('model loaded from %s' % path)


def to_tensor(var, device):
    var = torch.Tensor(var)
    var = var.to(device)
    return var.long()


# 读取物品类别信息，返回一个dict，key:item_id，value:cate_id
def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


# 计算物品多样性，item_list中的所有item两两计算
def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity
