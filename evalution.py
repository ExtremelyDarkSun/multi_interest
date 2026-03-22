from collections import defaultdict
import math
import sys
import time
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import signal


error_flag = {'sig':0}

def sig_handler(signum, frame):
    error_flag['sig'] = signum
    print("segfault core", signum)

signal.signal(signal.SIGSEGV, sig_handler)

from utils import get_DataLoader, get_exp_name, get_model, load_model, save_model, to_tensor, load_item_cate, compute_diversity
def evaluate_pop(model, test_data, hidden_size, device, topN=20):
    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0
    for _, (users, targets, items, mask, times) in enumerate(test_data):
        res = model.full_sort_predict(1)


        item_list = res.argsort(0, True)[:topN]
        assert len(item_list.size()) == 1
        assert item_list.size(0) == topN

        for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            recall = 0
            dcg = 0.0
            # item_list = set(res[0])  # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
            for no, iid in enumerate(item_list):  # 对于每一个label物品
                if iid in iid_list:  # 如果该label物品在近邻搜索的结果中
                    recall += 1
                    dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            if recall > 0:  # recall>0当然表示有命中
                total_ndcg += dcg / idcg
                total_hitrate += 1

        total += len(targets)  # total增加每个批次的用户数量

    recall = total_recall / total  # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total  # NDCG
    hitrate = total_hitrate * 1.0 / total  # 命中率
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}

def evaluate(model, test_data, hidden_size, device, k=20, coef=None, item_cate_map=None, args=None):
    if model.name == 'Pop':
        return evaluate_pop(model, test_data, hidden_size, device, k)
    topN = k # 评价时选取topN
    if coef is not None:
        coef = float(coef)

    gpu_indexs = [None]
    for i in range(1000):
        try:
            item_embs = model.output_items().cpu().detach().numpy()
            res = faiss.StandardGpuResources()  # 使用单个GPU
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = device.index

            gpu_indexs[0] = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)  # 建立GPU index用于Inner Product近邻搜索
            gpu_indexs[0].add(item_embs) # 给index添加向量数据
            if error_flag['sig'] == 0:
                break
            else:
                print("core received", error_flag['sig'])
                error_flag['sig'] = 0
        except Exception as e:
            print("error received", e)
        print("Faiss re-try", i)
        time.sleep(5)


    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for _, (users, targets, items, mask, times) in enumerate(test_data): # 一个batch的数据
        
        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        time_mat, adj_mat = times
        time_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))
        user_embs,_ = model(to_tensor(items, device), None, to_tensor(mask, device), time_tensor, device, train=False)
        user_embs = user_embs.cpu().detach().numpy()
        gpu_index = gpu_indexs[0]
        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2: # 非多兴趣模型评估
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list = set(I[i]) # I[i]是一个batch中第i个用户的近邻搜索结果，i∈[0, batch_size)
                for no, iid in enumerate(item_list): # 对于每一个label物品
                    if iid in iid_list: # 如果该label物品在近邻搜索的结果中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(I[i], item_cate_map) # 两个参数分别为推荐物品列表和物品类别字典
        else: # 多兴趣模型评估
            ni = user_embs.shape[1] # num_interest
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]]) # shape=(batch_size*num_interest, embedding_dim)
            D, I = gpu_index.search(user_embs, topN) # Inner Product近邻搜索，D为distance，I是index
            for i, iid_list in enumerate(targets): # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                recall = 0
                dcg = 0.0
                item_list_set = set()
                if coef is None: # 不考虑物品多样性
                    # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True) # 降序排序，内积越大，向量越近
                    for j in range(len(item_list)): # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else: # 考虑物品多样性
                    coef = float(coef)
                    # 所有兴趣向量的近邻物品集中起来按距离再次排序
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = [] # 存放（item_id, distance, item_cate）三元组，要用到物品类别，所以只存放有类别的物品
                    tmp_item_set = set() # 近邻推荐物品中有类别的物品的集合
                    for (x, y) in origin_item_list: # x为索引，y为距离
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN): # 选出topN个物品
                        max_index = 0
                        # score = distance - λ * 已选出的物品中与该物品的类别相同的物品的数量（score越大越好）
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)): # 遍历所有候选物品，每个循环找出一个score最大的item
                            # 第一次遍历必然先选出第一个物品
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score: # 当距离得分小于max_score时，后续物品得分一定小于max_score
                                break
                        item_list_set.add(item_list[max_index][0])
                        # 选出来的物品的类别对应的value加1，这里是为了尽可能选出类别不同的物品
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index) # 候选物品列表中删掉选出来的物品



                # 上述if-else只是为了用不同方式计算得到最后推荐的结果item列表
                for no, iid in enumerate(item_list_set): # 对于每一个推荐的物品
                    if iid in iid_list: # 如果该推荐的物品在label物品列表中
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list) # len(iid_list)表示label数量
                if recall > 0: # recall>0当然表示有命中
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if coef is not None:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(targets) # total增加每个批次的用户数量
    
    recall = total_recall / total # 召回率，每个用户召回率的平均值
    ndcg = total_ndcg / total # NDCG
    hitrate = total_hitrate * 1.0 / total # 命中率
    if coef is None:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    diversity = total_diversity * 1.0 / total # 多样性
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

torch.set_printoptions(
    precision=2,    # 精度，保留小数点后几位，默认4
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)

def train(device, train_file, valid_file, test_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, max_iter, test_iter, decay_step, lr_decay, patience, exp, args):
    # if model_type in ['MIND', 'ComiRec-DR']:
    #     lr = 0.005

    print("Param lr=" + str(lr))
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, exp=exp) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径

    # prepare data
    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1, args=args)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0, args=args)

    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len, args=args, device=device)
    model = model.to(device)
    model.set_device(device)

    # [DisMIR] Load confidence matrix for partition loss
    if model_type == "DisMIR":
        model.load_confidence_matrix(dataset, data_path='./data/')

    # [DASD-DisMIR] Load confidence matrix for the underlying DisMIR model
    if model_type == "DASD-DisMIR":
        model.load_confidence_matrix(dataset, data_path='./data/')

    model.set_sampler(args, device=device)

    # [Stage-2] Load pretrained Teacher weights before joint training
    if model_type == "DASD-DisMIR" and getattr(args, 'pretrain', 0) == 2:
        # Use custom path if provided, otherwise auto-generate
        if getattr(args, 'teacher_ckpt', None) is not None:
            teacher_model_path = args.teacher_ckpt
            # Ensure path ends with /
            if not teacher_model_path.endswith('/'):
                teacher_model_path += '/'
        else:
            teacher_model_path = "best_model/" + exp_name + "_teacher/"
        load_teacher_weights(model, teacher_model_path)
        print(f"[Pretrain Stage-2] Teacher weights loaded from {teacher_model_path}; proceeding with joint training")

    loss_fn = nn.CrossEntropyLoss()

    # Stage-2：tokenizer 使用小学习率微调，其余参数使用默认 lr
    if model_type == "DASD-DisMIR" and getattr(args, 'pretrain', 0) == 2:
        tokenizer_lr_ratio = getattr(args, 'tokenizer_lr_ratio', 0.1)
        tok_ids    = set(id(p) for p in model.tokenizer.parameters())
        tok_params = list(model.tokenizer.parameters())
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in tok_ids]
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': lr},
            {'params': tok_params,   'lr': lr * tokenizer_lr_ratio},
        ], weight_decay=args.weight_decay)
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Stage-2] tokenizer lr={lr * tokenizer_lr_ratio:.2e}, "
              f"others lr={lr:.2e}, trainable={trainable}/{total} params")
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=args.weight_decay
        )

    trials = 0

    print('training begin')
    sys.stdout.flush()

    start_time = time.time()
    model.loss_fct = loss_fn
    try:
        total_loss, total_loss_1, total_loss_2, total_loss_3, total_loss_4, total_loss_5  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Loss accumulators for detailed logging
        loss_accumulators = {}
        iter = 0
        best_metric = 0 # 最佳指标值，在这里是最佳recall值
        loss_print_interval = getattr(args, 'loss_print_interval', 100)
        #scheduler.step()
        for i, batch in enumerate(train_data):
            # [Multi-Target] Training batches now contain 6 elements when num_future_labels > 0:
            # (users, targets, items, mask, times, future_labels)
            # Eval/test batches (train_flag=0) still have 5 elements; future_labels = None.
            if len(batch) == 6:
                users, targets, items, mask, times, future_labels_raw = batch
                future_labels = to_tensor(future_labels_raw, device)  # (B, num_future_labels)
            else:
                users, targets, items, mask, times = batch
                future_labels = None
            model.train()
            iter += 1
            optimizer.zero_grad()
            pos_items = to_tensor(targets, device)
            interests, atten, readout, selection = None, None, None, None
            time_mat, adj_mat = times
            times_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))
            if model_type in ['ComiRec-SA', "REMI"]:
                interests, scores, atten, readout, selection = model(to_tensor(items, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'ComiRec-DR':
                interests, scores, readout = model(to_tensor(items, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'MIND':
                interests, scores, readout, selection = model(to_tensor(items, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type in ['GRU4Rec', 'DNN']:
                readout, scores = model(to_tensor(items, device), pos_items, to_tensor(mask, device), times_tensor, device)

            if model_type == 'Pop':
                loss = model.calculate_loss(pos_items)
            elif model_type == "DisMIR":
                # [DisMIR] BPR loss with hard negative mining + Partition loss
                # [PAPER_REF] Eq 6: L = L_Rec + λ * L_Partition
                interests, scores, atten, readout, selection = model(
                    to_tensor(items, device), pos_items, to_tensor(mask, device), times_tensor, device
                )

                # Check for NaN in model outputs
                if torch.isnan(readout).any():
                    print(f"[DisMIR Warning] readout contains NaN at iter {iter}")
                if torch.isnan(interests).any():
                    print(f"[DisMIR Warning] interests contains NaN at iter {iter}")

                # BPR loss with hard negative mining
                bpr_loss = model.compute_bpr_loss_with_hard_negative(readout, pos_items, model.hard_neg_candidates)

                # Check for NaN in BPR loss
                if torch.isnan(bpr_loss):
                    print(f"[DisMIR Warning] BPR loss is NaN at iter {iter}, using fallback")
                    bpr_loss = torch.tensor(1.0, device=device)  # Fallback loss

                loss = bpr_loss

                # Initialize loss dict for detailed logging
                iter_loss_dict = {'bpr_loss': bpr_loss.item()}

                # Item partition loss
                if args.dlambda > 0:
                    items_tensor = to_tensor(items, device)
                    mask_tensor = to_tensor(mask, device)
                    # 使用确定性种子确保可复现性（与DASD-DisMIR一致）
                    deterministic_seed = 42 + items_tensor.sum().item() % 10000
                    partition_loss = model.compute_partition_loss(
                        items_tensor,
                        mask_tensor,
                        seed=deterministic_seed
                    )
                    if not torch.isnan(partition_loss):
                        loss = loss + args.dlambda * partition_loss
                        iter_loss_dict['partition_loss'] = partition_loss.item()
                    else:
                        print(f"[DisMIR Warning] partition_loss is NaN at iter {iter}, skipping")
                        iter_loss_dict['partition_loss'] = 0.0

                # Optional: Routing regularization
                if args.rlambda > 0:
                    atten_loss = model.calculate_atten_loss(atten)
                    loss = loss + args.rlambda * atten_loss
                    iter_loss_dict['atten_loss'] = atten_loss.item()

                # Accumulate losses for detailed logging
                for key, val in iter_loss_dict.items():
                    if key not in loss_accumulators:
                        loss_accumulators[key] = 0.0
                    loss_accumulators[key] += val

                # Print detailed losses at intervals
                if iter % loss_print_interval == 0 and iter > 0:
                    loss_detail_str = ', '.join([f"{k}: {v/loss_print_interval:.4f}" for k, v in loss_accumulators.items()])
                    print(f"[DisMIR Loss Details @ iter {iter}] {loss_detail_str}")
                    # Reset accumulators
                    loss_accumulators = {}

            elif model_type == "DASD-DisMIR":
                # [DASD-DisMIR] Knowledge Distillation with DisMIR
                # Returns (interests, total_loss, loss_dict) in training mode
                interests, total_loss, loss_dict = model(
                    to_tensor(items, device), pos_items, to_tensor(mask, device),
                    times_tensor, device, train=True,
                    future_labels=future_labels  # [Multi-Target] pass extra labels (or None)
                )
                loss = total_loss

                # Accumulate losses for detailed logging
                for key, val in loss_dict.items():
                    if key not in loss_accumulators:
                        loss_accumulators[key] = 0.0
                    loss_accumulators[key] += val

                # Print detailed losses at intervals
                if iter % loss_print_interval == 0 and iter > 0:
                    avg_losses = {k: v/loss_print_interval for k, v in loss_accumulators.items()}
                    print(f"[DASD-DisMIR Loss Details @ iter {iter}] "
                          f"bpr: {avg_losses.get('dismir_bpr', 0):.4f}, "
                          f"false_bpr: {avg_losses.get('false_bpr', 0):.4f} "
                          f"(hard={avg_losses.get('false_bpr_hard', 0):.4f}, "
                          f"all={avg_losses.get('false_bpr_all', 0):.4f}), "
                          f"teacher_mse: {avg_losses.get('teacher_mse', 0):.4f}, "
                          f"chamfer: {avg_losses.get('chamfer_loss', 0):.4f}, "
                          f"vq: {avg_losses.get('vq_loss', 0):.4f}, "
                          f"partition: {avg_losses.get('partition_loss', 0):.4f}, "
                          f"atten: {avg_losses.get('atten_loss', 0):.4f}, "
                          f"total: {avg_losses.get('total_loss', 0):.4f}")
                    # Reset accumulators
                    loss_accumulators = {}
            else:
                loss = model.calculate_sampled_loss(readout, pos_items, selection, interests) if model.is_sampler else model.calculate_full_loss(loss_fn, scores, to_tensor(targets, device), interests)

            if model_type == "REMI":
                loss += args.rlambda * model.calculate_atten_loss(atten)

            loss.backward()

            optimizer.step()

            total_loss += loss

            if iter%test_iter == 0:
                model.eval()
                metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args)
                log_str = 'iter: %d, train loss: %.4f' % (iter, total_loss / test_iter) # 打印loss
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(exp_name)
                print(log_str)

                # 保存recall最佳的模型
                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        save_model(model, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience: # early stopping
                            print("early stopping!")
                            break
                
                # 每次test之后loss_sum置零
                total_loss = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time-start_time)/60.0))
                sys.stdout.flush()

            if iter >= max_iter * 1000: # 超过最大迭代次数，退出训练
                break

    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')

    load_model(model, best_model_path)
    model.eval()

    # 训练结束后用valid_data测试一次
    metrics = evaluate(model, valid_data, hidden_size, device, topN, args=args)
    print(', '.join(['Valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    # 训练结束后用test_data测试一次
    print("Test result:")
    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0,  args=args)
    metrics = evaluate(model, test_data, hidden_size, device, 20, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@20' + '=%.6f' % value)

    metrics = evaluate(model, test_data, hidden_size, device, 50, args=args)
    for key, value in metrics.items():
        print('test ' + key + '@50' + '=%.6f' % value)


def test(device, test_file, cate_file, dataset, model_type, item_count, batch_size, lr, seq_len, 
            hidden_size, interest_num, topN, coef=None, exp='test'):
    
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=False, exp=exp) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径A

    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    # [DisMIR] Load confidence matrix for partition loss
    if model_type == "DisMIR":
        model.load_confidence_matrix(dataset, data_path='./data/')

    # [DASD-DisMIR] Load confidence matrix for the underlying DisMIR model
    if model_type == "DASD-DisMIR":
        model.load_confidence_matrix(dataset, data_path='./data/')

    test_data = get_DataLoader(test_file, batch_size, seq_len, train_flag=0)
    item_cate_map = load_item_cate(cate_file) # 读取物品的类型
    metrics = evaluate(model, test_data, hidden_size, device, topN, coef=coef, item_cate_map=item_cate_map)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def save_teacher_weights(model, teacher_model_path):
    """Save only the Tokenizer (Teacher) weights + teacher_embeddings."""
    if not os.path.exists(teacher_model_path):
        os.makedirs(teacher_model_path)
    state = {
        'tokenizer': model.tokenizer.state_dict(),
        'teacher_embeddings': model.teacher_embeddings.state_dict(),
    }
    torch.save(state, teacher_model_path + 'teacher.pt')
    print(f'Teacher weights saved to {teacher_model_path}teacher.pt')


def load_teacher_weights(model, teacher_model_path):
    """Load Tokenizer (Teacher) weights + teacher_embeddings.
    dismir.embeddings (Student) is intentionally NOT restored;
    Stage-2 always starts with freshly-initialised student embeddings."""
    path = teacher_model_path + 'teacher.pt'
    state = torch.load(path, map_location='cpu')
    model.tokenizer.load_state_dict(state['tokenizer'])
    model.teacher_embeddings.load_state_dict(state['teacher_embeddings'])
    print(f'Teacher (tokenizer + teacher_embeddings) weights loaded from {path}')
    print('[Stage-2] dismir.embeddings kept at fresh random init (not loaded from Stage-1)')


def train_teacher_pretrain(device, train_file, valid_file, dataset, model_type,
                           item_count, batch_size, lr, seq_len, hidden_size,
                           interest_num, topN, max_iter, test_iter, decay_step,
                           lr_decay, patience, exp, args):
    """
    Stage-1: Pretrain only the Teacher (Tokenizer) of DASD-DisMIR.

    Optimises recon_loss + infonce_loss from forward_teacher_pretrain().
    Saves the LATEST checkpoint (not best) to best_model/{exp_name}_teacher_{timestamp}/.
    """
    from utils import get_exp_name, get_DataLoader, get_model, save_model, to_tensor
    from datetime import datetime

    print("[Pretrain Stage-1] Teacher-only pretraining")
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size,
                            seq_len, interest_num, topN, exp=exp)
    # Add timestamp suffix to folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_model_path = "best_model/" + exp_name + "_teacher_" + timestamp + "/"

    train_data = get_DataLoader(train_file, batch_size, seq_len, train_flag=1, args=args)
    valid_data = get_DataLoader(valid_file, batch_size, seq_len, train_flag=0, args=args)

    model = get_model(dataset, model_type, item_count, batch_size, hidden_size,
                      interest_num, seq_len, args=args, device=device)
    model = model.to(device)
    model.set_device(device)
    model.load_confidence_matrix(dataset, data_path='./data/')
    model.set_sampler(args, device=device)

    # Only optimise Teacher-related parameters (tokenizer + teacher_embeddings)
    teacher_params = (
        list(model.tokenizer.parameters()) +
        list(model.teacher_embeddings.parameters())
    )
    optimizer = torch.optim.Adam(teacher_params, lr=lr,
                                 weight_decay=args.weight_decay)

    trials = 0
    best_val_loss = float('inf')
    best_recall = 0.0  # Track best recall for checkpoint saving
    loss_print_interval = getattr(args, 'loss_print_interval', 100)
    loss_accumulators = {}
    start_time = time.time()

    print('[Pretrain Stage-1] training begin')
    sys.stdout.flush()

    try:
        iter_count = 0
        for i, batch in enumerate(train_data):
            # [Multi-Target] Training batches have 6 elements; ignore future_labels here
            # since forward_teacher_pretrain does not use them.
            if len(batch) == 6:
                users, targets, items, mask, times, _ = batch
            else:
                users, targets, items, mask, times = batch
            model.train()
            iter_count += 1
            optimizer.zero_grad()

            pos_items = to_tensor(targets, device)
            time_mat, adj_mat = times
            times_tensor = (to_tensor(time_mat, device), to_tensor(adj_mat, device))

            total_loss, loss_dict = model.forward_teacher_pretrain(
                to_tensor(items, device), pos_items,
                to_tensor(mask, device), times_tensor, device
            )
            total_loss.backward()
            optimizer.step()

            # Accumulate for periodic printing
            for key, val in loss_dict.items():
                loss_accumulators[key] = loss_accumulators.get(key, 0.0) + val

            if iter_count % loss_print_interval == 0:
                avg = {k: v / loss_print_interval for k, v in loss_accumulators.items()}
                print(f"[Pretrain-Teacher @ iter {iter_count}] "
                      f"recon: {avg.get('recon_loss', 0):.4f}, "
                      f"w_recon: {avg.get('weighted_recon_loss', 0):.4f}, "
                      f"vq: {avg.get('vq_loss', 0):.4f}, "
                      f"w_vq: {avg.get('weighted_vq_loss', 0):.4f}, "
                      f"total: {avg.get('total_loss', 0):.4f}")
                loss_accumulators = {}

            if iter_count % test_iter == 0:
                # Validation: measure all losses on valid set
                model.eval()
                val_accumulators = {}
                val_count = 0
                with torch.no_grad():
                    for _, (v_users, v_targets, v_items, v_mask, v_times) in enumerate(valid_data):
                        # In eval mode, v_targets is a list of lists (per-user test items).
                        # Take the first item of each user's target list to match training format.
                        v_targets_flat = [t[0] if isinstance(t, (list, tuple)) else t for t in v_targets]
                        v_pos = to_tensor(v_targets_flat, device)
                        v_tm, v_am = v_times
                        v_times_t = (to_tensor(v_tm, device), to_tensor(v_am, device))
                        _, v_loss_dict = model.forward_teacher_pretrain(
                            to_tensor(v_items, device), v_pos,
                            to_tensor(v_mask, device), v_times_t, device
                        )
                        for key, val in v_loss_dict.items():
                            val_accumulators[key] = val_accumulators.get(key, 0.0) + val
                        val_count += 1

                val_avg = {k: v / max(val_count, 1) for k, v in val_accumulators.items()}
                val_recon_avg = val_avg.get('recon_loss', 0.0)

                # ========== Teacher-specific Recall evaluation (Single Token) ==========
                # During pretrain, Student is not trained, so we must use Teacher
                # to generate single token and compute recall against item embeddings
                # NOTE: Use the same label for train and eval (first label only)
                total_recall, total_ndcg, total_hitrate, total = 0.0, 0.0, 0, 0

                with torch.no_grad():
                    # Prepare faiss index using teacher_embeddings (Stage-1 space)
                    # Normalize to match recon_target (normalized in encode_with_teacher)
                    item_embs = F.normalize(model.teacher_embeddings.weight, dim=-1).cpu().numpy()
                    res = faiss.StandardGpuResources()
                    flat_config = faiss.GpuIndexFlatConfig()
                    flat_config.device = device.index
                    gpu_index = faiss.GpuIndexFlatIP(res, hidden_size, flat_config)
                    gpu_index.add(item_embs)

                    codebook_size = getattr(args, 'vq_num_embeddings', 5000)
                    used_codes = set()  # 收集整个 valid 集上使用过的 codebook entry

                    total_cosine = 0.0  # for avg recon cosine diagnostic
                    num_cosine_batches = 0
                    for _, (v_users, v_targets, v_items, v_mask, v_times) in enumerate(valid_data):
                        target_labels = [t[0] if isinstance(t, (list, tuple)) else t for t in v_targets]

                        # encode_with_teacher 返回 (recon_target, vq_indices)
                        teacher_token, vq_indices = model.encode_with_teacher(
                            to_tensor(v_items, device),
                            to_tensor(target_labels, device),
                            to_tensor(v_mask, device),
                            device
                        )  # (B, D), (B, K)

                        # 收集 codebook indices
                        used_codes.update(vq_indices.cpu().numpy().flatten().tolist())

                        # 计算平均重建余弦相似度（诊断 VQ 瓶颈是否过强）
                        label_emb_norm = F.normalize(
                            model.teacher_embeddings(to_tensor(target_labels, device)).detach(), dim=-1
                        )  # (B, D)
                        batch_cosine = (teacher_token * label_emb_norm).sum(dim=-1).mean().item()
                        total_cosine += batch_cosine
                        num_cosine_batches += 1

                        token_np = teacher_token.cpu().numpy()  # (B, D)
                        D, I = gpu_index.search(token_np, topN)

                        for i, target_label in enumerate(target_labels):
                            recall = 0
                            dcg = 0.0
                            target_found = False
                            for no, iid in enumerate(I[i]):
                                if iid == target_label:
                                    recall = 1
                                    dcg = 1.0 / math.log(no + 2, 2)
                                    target_found = True
                                    break
                            total_recall += recall
                            if target_found:
                                total_ndcg += dcg
                                total_hitrate += 1

                        total += len(target_labels)

                # codebook 利用率
                codebook_utilization = len(used_codes) / codebook_size

                # 平均重建余弦相似度（VQ 瓶颈质量诊断：越接近1.0越好）
                avg_recon_cosine = total_cosine / max(num_cosine_batches, 1)

                # Aggregate metrics
                metrics = {
                    'recall': total_recall / total if total > 0 else 0,
                    'ndcg': total_ndcg / total if total > 0 else 0,
                    'hitrate': total_hitrate * 1.0 / total if total > 0 else 0,
                    'codebook_util': codebook_utilization,
                    'recon_cosine': avg_recon_cosine,
                }
                # ========== End Teacher-specific evaluation ==========

                test_time = time.time()
                current_recall = metrics.get('recall', 0)
                print(f"[Pretrain-Teacher @ iter {iter_count}] "
                      f"val_recon: {val_avg.get('recon_loss', 0):.6f}, "
                      f"val_vq: {val_avg.get('vq_loss', 0):.6f}, "
                      f"val_total: {val_avg.get('total_loss', 0):.6f}  |  "
                      f"recall@{topN}: {current_recall:.6f} {'★BEST' if current_recall > best_recall else ''}, "
                      f"ndcg@{topN}: {metrics.get('ndcg', 0):.6f}, "
                      f"hitrate@{topN}: {metrics.get('hitrate', 0):.6f}  |  "
                      f"codebook_util: {metrics.get('codebook_util', 0):.3f} "
                      f"({int(metrics.get('codebook_util', 0) * codebook_size)}/{codebook_size})  "
                      f"recon_cosine: {metrics.get('recon_cosine', 0):.4f}  |  "
                      f"time: {(test_time - start_time) / 60:.2f}min")
                sys.stdout.flush()

                # Save checkpoint when recall improves (primary metric = reconstruction recall)
                if current_recall > best_recall:
                    best_recall = current_recall
                    best_val_loss = val_recon_avg
                    save_teacher_weights(model, teacher_model_path)
                    print(f"[Pretrain-Teacher] Best recall={best_recall:.6f}, checkpoint saved to {teacher_model_path}teacher.pt")
                    trials = 0
                else:
                    trials += 1
                    if trials > patience:
                        print("[Pretrain-Teacher] early stopping!")
                        break

            if iter_count >= max_iter * 1000:
                break

    except KeyboardInterrupt:
        print('-' * 99)
        print('[Pretrain-Teacher] Exiting from training early')

    print(f"[Pretrain Stage-1] Done. Best recall={best_recall:.6f}, best val_recon={best_val_loss:.6f}")
    print(f"[Pretrain Stage-1] Best weights saved to {teacher_model_path}teacher.pt")


def output(device, dataset, model_type, item_count, batch_size, lr, seq_len,
            hidden_size, interest_num, topN, exp='eval'):
    
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, hidden_size, seq_len, interest_num, topN, save=False, exp=exp) # 实验名称
    best_model_path = "best_model/" + exp_name + '/' # 模型保存路径
    
    model = get_model(dataset, model_type, item_count, batch_size, hidden_size, interest_num, seq_len)
    load_model(model, best_model_path)
    model = model.to(device)
    model.eval()

    item_embs = model.output_items() # 获取物品嵌入
    np.save('output/' + exp_name + '_emb.npy', item_embs) # 保存物品嵌入
