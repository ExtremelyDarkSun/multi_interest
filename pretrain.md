# Teacher 预训练两阶段训练方案

## 背景

当前 DASD-DisMIR 的 Teacher（ContextGatedTokenizer）和 Student（DisMIR）同时从头训练，Teacher 早期很弱，可能对 Student 产生负面引导。
本方案通过 `--pretrain` 参数开启两阶段训练，**原有联合训练逻辑完全不变**。

---

## 参数说明

| `--pretrain` 值 | 含义 |
|---|---|
| `0`（默认） | 原有联合训练，行为与之前完全相同 |
| `1` | 仅预训练 Teacher，保存权重到 `best_model/{exp_name}_teacher/` |
| `2` | 加载预训练 Teacher 权重，再进行正常联合训练 |

- Stage 1 中 embedding 允许更新（Teacher 和 Student 共享 embedding table）
- Stage 2 中 Teacher 继续随联合训练微调，不冻结

---

## 修改文件一览

### 1. `utils.py` — 新增 `--pretrain` 参数

在 `get_parser()` 末尾，`return parser` 前新增：

```python
# [Teacher Pretrain] Two-stage training for DASD-DisMIR
# 0: joint training (default, original behaviour)
# 1: pretrain Teacher only, save weights to best_model/{exp_name}_teacher/
# 2: load pretrained Teacher weights then run normal joint training
parser.add_argument('--pretrain', type=int, default=0,
                    help='[DASD-DisMIR] Teacher pretrain stage: 0=joint(default), 1=pretrain Teacher, 2=joint with pretrained Teacher')
```

---

### 2. `DASD_DisMIR.py` — 新增 `forward_teacher_pretrain()` 方法

在 `DASD_DisMIR` 类的兼容接口方法区块之前新增：

```python
def forward_teacher_pretrain(self, item_list, label_list, mask, times, device):
    """
    Stage-1 Teacher-only forward pass.
    只训练 Tokenizer（Teacher），损失 = lambda_recon * recon_loss + lambda_infonce * infonce_loss
    """
    # Shared embedding lookup (gradients flow here)
    label_eb = self.dismir.embeddings(label_list)       # (B, D)
    history_eb = self.dismir.embeddings(item_list)      # (B, L, D)
    history_eb = history_eb * mask.unsqueeze(-1)

    # Partition-aware history enhancement (same as joint training)
    history_residual = self.partition_enhancer(history_eb)
    history_enhanced = self.partition_enhancer_norm(history_eb + history_residual)

    # Teacher forward
    tokens, recon_target = self.tokenizer(label_eb, history_enhanced, mask)

    # Reconstruction loss
    recon_loss = F.mse_loss(
        F.normalize(recon_target, dim=-1),
        F.normalize(label_eb.detach(), dim=-1)
    )

    # InfoNCE loss (Teacher tokens vs target labels)
    infonce_loss = self.calculate_infonce_loss(tokens, label_eb, temperature=0.5)

    total_loss = self.lambda_recon * recon_loss + self.lambda_infonce * infonce_loss

    loss_dict = {
        'recon_loss':   recon_loss.item(),
        'infonce_loss': infonce_loss.item(),
        'total_loss':   total_loss.item(),
    }
    return total_loss, loss_dict
```

---

### 3. `evalution.py` — 新增三个函数

#### `save_teacher_weights(model, teacher_model_path)`
保存 Teacher 相关权重（tokenizer、partition_enhancer、partition_enhancer_norm、embeddings）到 `{path}teacher.pt`。

#### `load_teacher_weights(model, teacher_model_path)`
从 `{path}teacher.pt` 加载上述权重。

#### `train_teacher_pretrain(...)`
Stage-1 训练主循环：
- 只优化 Teacher 相关参数（tokenizer + partition_enhancer + partition_enhancer_norm + embeddings）
- 每 `loss_print_interval` 步打印训练 loss：`recon`、`infonce`、`total`
- 每 `test_iter` 步在验证集上计算并打印：`val_recon`、`val_infonce`、`val_total`
- 以 `val_recon` 为指标做 early stopping，保存最优 Teacher 权重

#### `train()` 函数内（Stage-2 hook）
在 `model.set_sampler()` 之后、创建 optimizer 之前，若 `args.pretrain == 2`，自动调用 `load_teacher_weights()` 加载预训练权重，随后进行正常联合训练。

---

### 4. `train.py` — 根据 `--pretrain` 分发

```python
from evalution import train, test, output, train_teacher_pretrain

if args.p == 'train':
    pretrain_stage = getattr(args, 'pretrain', 0)

    if pretrain_stage == 1:
        # Stage 1: pretrain Teacher only
        train_teacher_pretrain(...)
    else:
        # Stage 0 (joint) or Stage 2 (load Teacher + joint)
        train(...)
```

---

## 运行命令

以下命令基于原先调好的超参（book 数据集）：

### Stage 1：预训练 Teacher

```bash
python src/train.py \
  -p train \
  --dataset book \
  --model_type DASD-DisMIR \
  --hidden_size 64 \
  --interest_num 4 \
  --dlambda 0.2 \
  --lambda_recon 0.1 \
  --lambda_align 0.5 \
  --lambda_infonce 1 \
  --rlambda 0.0 \
  --gpu 0 \
  --lambda_partition_align 0 \
  --num_negatives 1280 \
  --hard_neg_candidates 1280 \
  --pretrain 1 \
  --exp e1
```

Stage 1 只训练 Teacher，结束后权重保存在：
```
best_model/book_DASD-DisMIR_b128_lr0.001_d64_len20_in4_top50_e1_teacher/teacher.pt
```

### Stage 2：加载预训练 Teacher，正常联合蒸馏

```bash
python src/train.py \
  -p train \
  --dataset book \
  --model_type DASD-DisMIR \
  --hidden_size 64 \
  --interest_num 4 \
  --dlambda 0.2 \
  --lambda_recon 0.1 \
  --lambda_align 0.5 \
  --lambda_infonce 1 \
  --rlambda 0.0 \
  --gpu 0 \
  --lambda_partition_align 0 \
  --num_negatives 1280 \
  --hard_neg_candidates 1280 \
  --pretrain 2 \
  --exp e1
```

Stage 2 自动从上面路径加载 Teacher 权重，之后和原来联合训练完全一致。

### 原有方式（不变）

```bash
python src/train.py \
  -p train \
  --dataset book \
  --model_type DASD-DisMIR \
  --hidden_size 64 \
  --interest_num 4 \
  --dlambda 0.2 \
  --lambda_recon 0.1 \
  --lambda_align 0.5 \
  --lambda_infonce 1 \
  --rlambda 0.0 \
  --gpu 0 \
  --lambda_partition_align 0 \
  --num_negatives 1280 \
  --hard_neg_candidates 1280 \
  --exp e1
```

不传 `--pretrain` 或传 `--pretrain 0`，行为与之前完全相同。

---

## Stage 1 输出示例

```
[Pretrain Stage-1] Teacher-only pretraining
[Pretrain-Teacher @ iter 100] recon: 0.0312, infonce: 2.1045, total: 0.2416
[Pretrain-Teacher @ iter 200] recon: 0.0287, infonce: 1.9832, total: 0.2272
...
[Pretrain-Teacher @ iter 1000] val_recon: 0.028543, val_infonce: 1.983201, val_total: 0.221162  time: 1.23min
Teacher weights saved to best_model/.../teacher.pt
...
[Pretrain Stage-1] Done. Best val_recon=0.024871
```

## Stage 2 输出示例

```
Teacher weights loaded from best_model/.../teacher.pt
[Pretrain Stage-2] Teacher weights loaded; proceeding with joint training
training begin
[DASD-DisMIR Loss Details @ iter 100] main: ..., recon: ..., align: ..., ...
```
