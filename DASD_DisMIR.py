import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


# ============ 1. ChamferLoss ============

class ChamferLoss(nn.Module):
    """
    Chamfer Loss (双向对齐损失)

    计算Teacher的Token集合与Student的兴趣向量集合之间的Chamfer Distance。
    使用最优传输思想，强制Student生成的兴趣向量去覆盖Teacher拆解出的真实意图Token。

    包含两个方向的匹配：
    1. Forward Matching: 每个Token都应该被某个兴趣向量覆盖
    2. Backward Matching: 每个兴趣向量都应该对应某个Token（防止生成无意义兴趣）
    """

    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, tokens, interests):
        """
        计算双向Chamfer Distance

        Args:
            tokens: Teacher生成的Token集合 (batch_size, K, hidden_size)
            interests: Student生成的兴趣向量集合 (batch_size, M, hidden_size)

        Returns:
            loss: 双向对齐损失（标量）
        """
        # 计算距离矩阵
        # dist_matrix: (batch_size, K, M)
        # 使用L2距离（p=2）

        tokens = F.normalize(tokens, p=2, dim=-1)#.detach()  # L2归一化
        interests = F.normalize(interests, p=2, dim=-1)

        dist_matrix = torch.cdist(tokens, interests, p=2)

        # Forward Matching: 每个token找最近的interest
        # 确保每个Token都能被某个兴趣向量覆盖
        min_dist_t_to_i, _ = dist_matrix.min(dim=2)  # (batch_size, K)
        loss_t_to_i = min_dist_t_to_i.mean()

        # Backward Matching: 每个interest找最近的token
        # 确保每个兴趣向量都有对应的Token（防止生成无意义的兴趣）
        min_dist_i_to_t, _ = dist_matrix.min(dim=1)  # (batch_size, M)
        loss_i_to_t = min_dist_i_to_t.mean()

        # 总损失为两个方向损失之和
        total_loss = loss_t_to_i + loss_i_to_t

        return total_loss


# ============ 2. Partition Enhancer ============

class PartitionEnhancer(nn.Module):
    """
    Partition 增强模块
    使用 temperature-scaled softmax 突出 item 的主要分区特征
    物理含义：将 embedding 转化为相对概率分布，突出主要分区，压制次要分区
    """
    def __init__(self, hidden_size, temperature=0.5):
        super(PartitionEnhancer, self).__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature  # < 1 使分布更尖锐

    def forward(self, x):
        """
        Args:
            x: (B, L, D) 或 (B, D) - 原始 item embedding
        Returns:
            enhanced: 相同 shape - partition 增强后的 embedding
        """
        # Temperature-scaled softmax: 突出主要分区，压制次要分区
        # dim=-1 表示在 hidden_size 维度上做 softmax
        partition_weights = F.softmax(x / self.temperature, dim=-1)

        # 原始特征加权：高激活维度保留，低激活维度被压制
        enhanced = x * partition_weights

        return enhanced


# ============ 3. Partition Aligned Loss ============

class PartitionAlignedLoss(nn.Module):
    """
    Token 与 Interest 的 Partition 结构对齐损失
    强制 Teacher 的 Tokens 和 Student 的 Interests 在相同的 Partition 维度上激活

    与 ChamferLoss 的区别：
    - ChamferLoss: 原始 embedding 空间的 L2 距离（几何对齐）
    - PartitionAlignedLoss: Softmax 后的 partition 空间的相似度（语义结构对齐）
    """
    def __init__(self, hidden_size, temperature=1.0):
        super(PartitionAlignedLoss, self).__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature

    def forward(self, tokens, interests):
        """
        Args:
            tokens: (B, K, D) - Teacher 生成的 tokens
            interests: (B, M, D) - Student 生成的 interests
        Returns:
            loss: scalar - partition 对齐损失（负值，需要最小化）
        """
        # 1. 提取 Partition 分布（在 hidden_size 维度 softmax）
        # 每个 token/interest 在各 partition 维度上的激活分布
        token_parts = F.softmax(tokens / self.temperature, dim=-1)      # (B, K, D)
        interest_parts = F.softmax(interests / self.temperature, dim=-1)  # (B, M, D)

        # 2. 计算每个 token 与每个 interest 的 partition 相似度
        # (B, K, D) @ (B, D, M) -> (B, K, M)
        part_sim = torch.matmul(token_parts, interest_parts.transpose(1, 2))

        # 3. 双向最佳匹配
        # 每个 token 找最相似的 interest
        best_for_token = part_sim.max(dim=2)[0].mean()  # (B, K) -> scalar
        # 每个 interest 找最相似的 token
        best_for_interest = part_sim.max(dim=1)[0].mean()  # (B, M) -> scalar

        # 4. 最大化相似度（即最小化负相似度）
        loss = -(best_for_token + best_for_interest) / 2

        return loss


# ============ 4. TargetAwareFusion ============
# NOTE: 该类当前未被使用，实际使用的是 dismir.read_out() 进行硬选择
# 保留代码以备后续可能需要注意力融合机制
'''
class TargetAwareFusion(nn.Module):
    """
    Target-Aware Fusion (目标感知融合层)

    将M个兴趣向量通过注意力机制融合为单个向量，用于主任务Loss计算。
    仅在训练时使用，推理阶段不调用。

    Args:
        hidden_size: 隐藏层维度
    """

    def __init__(self, hidden_size):
        super(TargetAwareFusion, self).__init__()
        self.hidden_size = hidden_size

        # 注意力投影层：将target和interest映射到同一空间
        self.attention_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化模型参数"""
        nn.init.kaiming_normal_(self.attention_proj.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, interests, target_emb):
        """
        前向传播

        Args:
            interests: 兴趣向量集合 (batch_size, M, hidden_size)
            target_emb: Target物品的嵌入向量 (batch_size, hidden_size)

        Returns:
            fused_vector: 融合后的向量 (batch_size, hidden_size)
        """
        # 计算每个兴趣向量与target的相似度
        # 使用target作为query，interests作为key和value

        # 将target扩展为query: (batch_size, 1, hidden_size)
        target_query = target_emb.unsqueeze(1)  # (batch_size, 1, hidden_size)
        target_query_proj = self.attention_proj(target_query)  # (batch_size, 1, hidden_size)

        # 将interests投影: (batch_size, M, hidden_size)
        interests_proj = self.attention_proj(interests)  # (batch_size, M, hidden_size)

        # 计算注意力分数: (batch_size, 1, M)
        attention_scores = torch.matmul(target_query_proj, interests_proj.transpose(1, 2))

        # Softmax归一化得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, M)

        # 加权求和得到融合向量
        fused_vector = torch.matmul(attention_weights, interests)  # (batch_size, 1, hidden_size)
        fused_vector = fused_vector.squeeze(1)  # (batch_size, hidden_size)

        return fused_vector
'''


# ============ 3. Tokenizer 相关组件 ============

class HistoryEncoderLayer(nn.Module):
    """
    单层History编码器：类似Transformer Encoder Layer
    用于递进式编码history序列，每层Decoder对应一层History Encoder

    升级：添加显式残差连接，便于梯度传播，加速pretrain收敛
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention + FFN (使用TransformerEncoderLayer，包含残差连接)
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm结构，更稳定
        )

        # 显式层归一化（用于残差连接后的归一化）
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] - 输入序列
            mask: [batch_size, seq_len] - 1=valid, 0=padding
        Returns:
            encoded: [batch_size, seq_len, hidden_size]
        """
        residual = x

        # 处理mask: Transformer需要key_padding_mask (True=padding)
        key_pad_mask = (mask == 0) if mask is not None else None

        # Transformer encoding (内部包含Self-Attn + FFN + Residual)
        out = self.layer(x, src_key_padding_mask=key_pad_mask)

        # 显式残差连接 + 层归一化（双重残差，便于梯度传播）
        out = self.norm(out + residual)

        return out


class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SimpleRotaryEmbedding(nn.Module):
    """简化版的 RoPE，用于 ContextGatedTokenizer"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 确保 dim 是偶数
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # x: [batch_size, num_heads, seq_len, hidden_size]
        if seq_len is None:
            seq_len = x.shape[-2]

        # 生成位置索引
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # freqs: [seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextCrossAttention(nn.Module):
    """
    基于 Qwen3NextAttention 改造的交叉注意力模块

    支持两种模式：
    1. 第一层模式 (is_first_layer=True):
       - 输入单个 target_emb [B, D]
       - 输出多个 tokens [B, num_tokens, D]
       - 支持 use_token_split 参数

    2. 后续层模式 (is_first_layer=False):
       - 输入多个 tokens [B, num_tokens, D]
       - 输出多个 tokens [B, num_tokens, D]
       - 每个 token 作为独立的 query 进行 Cross-Attention
    """

    def __init__(self, hidden_size, num_tokens, num_key_value_heads=None, dropout=0.1,
                 use_token_split=False, is_first_layer=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_heads = num_tokens
        self.num_key_value_heads = num_key_value_heads or num_tokens
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = hidden_size ** -0.5
        self.attention_dropout = dropout
        self.use_token_split = use_token_split
        self.is_first_layer = is_first_layer

        # Query 投影：根据 is_first_layer 选择不同的投影方式
        if is_first_layer:
            # 第一层：从单个 target_emb 生成多个 tokens
            if use_token_split:
                # 检查hidden_size是否能被num_tokens整除
                if hidden_size % num_tokens != 0:
                    raise ValueError(f"use_token_split=True时，hidden_size ({hidden_size}) 必须能被 num_tokens ({num_tokens}) 整除")
                # 每个切分的token独立映射
                self.token_dim = hidden_size // num_tokens
                self.q_proj = nn.Linear(self.token_dim, hidden_size, bias=False)
            else:
                # 从完整的target_emb生成num_tokens个query
                self.q_proj = nn.Linear(hidden_size, num_tokens * hidden_size, bias=False)
        else:
            # 后续层：输入已经是 [B, num_tokens, D]，每个 token 独立映射
            # 输入维度：hidden_size，输出维度：hidden_size
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Key 投影：处理 history_emb，维度保持为 hidden_size
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * hidden_size,
            bias=False
        )
        # Value 投影：处理 history_emb，维度保持为 hidden_size
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * hidden_size,
            bias=False
        )
        self.target_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        # Q/K 归一化 - 现在归一化整个 hidden_size 维度
        self.q_norm = Qwen3NextRMSNorm(hidden_size, eps=1e-6)
        self.k_norm = Qwen3NextRMSNorm(hidden_size, eps=1e-6)

        # RoPE 位置编码 - 使用 hidden_size
        self.rotary_emb = SimpleRotaryEmbedding(
            hidden_size,
            max_position_embeddings=2048
        )
        self.token_dropout = nn.Dropout(1 / num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_emb, history_emb, key_padding_mask=None):
        """
        Args:
            target_emb:
                - 第一层: [batch_size, hidden_size] - 单个 query 来源
                - 后续层: [batch_size, num_tokens, hidden_size] - 多个 tokens
            history_emb: [batch_size, seq_len, hidden_size] - 多个 key/value
            key_padding_mask: [batch_size, seq_len] - True 表示 padding

        Returns:
            tokens: [batch_size, num_tokens, hidden_size] - 生成的 tokens
            attn_weights: 注意力权重（可选）
        """
        batch_size = target_emb.size(0)
        seq_len = history_emb.size(1)

        # 1. Query 投影 - 根据 is_first_layer 选择不同的方式
        if self.is_first_layer:
            # 第一层：从单个 target_emb 生成多个 tokens
            if self.use_token_split:
                # 方式1: 先切分target_emb，再每个token独立映射
                # [batch_size, hidden_size] -> [batch_size, num_tokens, token_dim]
                target_emb = self.target_proj(target_emb)
                target_tokens = target_emb.view(batch_size, self.num_tokens, self.token_dim)

                # 每个token独立映射成query: [batch_size, num_tokens, token_dim] -> [batch_size, num_tokens, hidden_size]
                # [batch_size, num_tokens, token_dim] -> [batch_size * num_tokens, token_dim]
                target_tokens_flat = target_tokens.reshape(batch_size * self.num_tokens, self.token_dim)
                # [batch_size * num_tokens, token_dim] -> [batch_size * num_tokens, hidden_size]
                query_states_flat = self.q_proj(target_tokens_flat)
                # [batch_size * num_tokens, hidden_size] -> [batch_size, num_tokens, hidden_size]
                query_states = query_states_flat.view(batch_size, self.num_tokens, self.hidden_size)

            else:
                # 方式2: 从完整target_emb生成num_tokens个query
                # [batch_size, hidden_size] -> [batch_size, num_tokens * hidden_size]
                q_proj_out = self.q_proj(target_emb)
                # [batch_size, num_tokens * hidden_size] -> [batch_size, num_tokens, hidden_size]
                query_states = q_proj_out.view(batch_size, self.num_tokens, self.hidden_size)
        else:
            # 后续层：输入已经是 [batch_size, num_tokens, hidden_size]
            # 每个 token 独立映射
            # [batch_size, num_tokens, hidden_size] -> [batch_size * num_tokens, hidden_size]
            target_tokens_flat = target_emb.reshape(batch_size * self.num_tokens, self.hidden_size)
            # [batch_size * num_tokens, hidden_size] -> [batch_size * num_tokens, hidden_size]
            query_states_flat = self.q_proj(target_tokens_flat)
            # [batch_size * num_tokens, hidden_size] -> [batch_size, num_tokens, hidden_size]
            query_states = query_states_flat.view(batch_size, self.num_tokens, self.hidden_size)

        # 2. Key 投影和 Value 投影
        # key_states: [batch_size, seq_len, num_key_value_heads, hidden_size]
        key_states = self.k_proj(history_emb).view(batch_size, seq_len, self.num_key_value_heads, self.hidden_size)

        # value_states: [batch_size, seq_len, num_key_value_heads, hidden_size]
        value_states = self.v_proj(history_emb).view(batch_size, seq_len, self.num_key_value_heads, self.hidden_size)

        # 3. Q/K 归一化
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # 4. 转置为 [batch_size, num_heads, seq_len, hidden_size]
        # query_states: [B, num_tokens, hidden_size] -> [B, num_tokens, 1, hidden_size]
        query_states = query_states.unsqueeze(2)  # [B, num_tokens, 1, hidden_size]
        key_states = key_states.transpose(1, 2)   # [B, num_kv_heads, seq_len, hidden_size]
        value_states = value_states.transpose(1, 2)  # [B, num_kv_heads, seq_len, hidden_size]

        # 5. 重复 KV（如果使用 GQA）
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 6. 计算注意力
        # query_states: [B, num_tokens, 1, hidden_size]
        # key_states: [B, num_tokens, seq_len, hidden_size]
        # [B, num_tokens, 1, hidden_size] @ [B, num_tokens, hidden_size, seq_len] -> [B, num_tokens, 1, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # 7. 应用 padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, seq_len], True 表示 padding
            # 转换为注意力 mask: [B, 1, 1, seq_len]
            attn_mask = key_padding_mask[:, None, None, :].float()
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            attn_weights = attn_weights + attn_mask

        # 8. Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)

        # 9. 加权求和
        # [B, num_tokens, 1, seq_len] @ [B, num_tokens, seq_len, hidden_size] -> [B, num_tokens, 1, hidden_size]
        attn_output = torch.matmul(attn_weights, value_states)

        # 10. 去掉多余的维度 [B, num_tokens, 1, hidden_size] -> [B, num_tokens, hidden_size]
        attn_output = attn_output.squeeze(2)

        tokens = attn_output
        return tokens, attn_weights


class ContextGatedTokenizer(nn.Module):
    """
    Context-Gated Aspect Tokenizer (Teacher模型) - 使用 Qwen3NextAttention

    升级点:
    1. 使用 Qwen3NextAttention 替换原有的 MultiheadAttention
    2. 集成 RoPE 位置编码
    3. 加入 Q/K 归一化和门控机制
    4. num_tokens 直接对应注意力头数
    5. 支持token切分模式（use_token_split）
    6. 支持多层叠加，每层包含：Cross-Attention + Self-Attention + FFN（标准Decoder Layer）
    7. 第一层不使用残差连接（强制模型学习用target查询history，避免target信息直接传递），后续层使用残差连接
    """

    def __init__(self, hidden_size, num_tokens=4, num_heads=4, num_key_value_heads=None,
                 dropout=0.1, num_decoder_layers=4, num_interaction_layers=1, use_token_split=True):
        super(ContextGatedTokenizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.use_token_split = use_token_split
        self.num_decoder_layers = num_decoder_layers

        # --- 递进式 History Encoder ---
        # 每层Decoder对应一层History Encoder，递进式编码
        self.history_enc_layers = nn.ModuleList()
        self.history_pos_emb = nn.Embedding(1000, hidden_size)  # 位置编码（只在第一层前加）

        for layer_idx in range(num_decoder_layers):
            history_enc = HistoryEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_tokens if hidden_size % num_tokens == 0 else 4,
                dropout=dropout
            )
            self.history_enc_layers.append(history_enc)

        # --- 多层 Decoder Layer：Cross-Attention + Self-Attention + FFN ---
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.cross_attn_dropouts = nn.ModuleList()

        self.self_attn_layers = nn.ModuleList()  # Self-Attention (Token Interaction)
        self.ffn_layers = nn.ModuleList()  # Feed-Forward Network

        for layer_idx in range(num_decoder_layers):
            # 1. Cross-Attention
            is_first_layer = (layer_idx == 0)

            cross_attn = Qwen3NextCrossAttention(
                hidden_size=hidden_size,
                num_tokens=num_tokens,
                num_key_value_heads=num_key_value_heads,
                dropout=dropout,
                use_token_split=use_token_split if is_first_layer else False,
                is_first_layer=is_first_layer
            )

            self.cross_attn_layers.append(cross_attn)
            self.cross_attn_norms.append(Qwen3NextRMSNorm(hidden_size, eps=1e-6))
            self.cross_attn_dropouts.append(nn.Dropout(dropout))

            # 2. Self-Attention + FFN (使用 TransformerEncoderLayer)
            # TransformerEncoderLayer 包含：Self-Attention + FFN + LayerNorm + Residual
            self_attn_ffn = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_tokens if hidden_size % num_tokens == 0 else 4,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )
            self.self_attn_layers.append(self_attn_ffn)

        # --- Gating Block (用于最后的门控) ---
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_gate = Qwen3NextRMSNorm(hidden_size, eps=1e-6)

        # --- Step 5: Reconstruction Block ---
        self.recon_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm_out = Qwen3NextRMSNorm(hidden_size, eps=1e-6)

        self.dimension_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        # 可选：添加温度参数来控制softmax的锐利程度
        self.gate_temperature = nn.Parameter(torch.ones(1))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, target_emb, history_emb, mask):
        """
        Args:
            target_emb: [batch_size, hidden_size] - 单个 query 来源
            history_emb: [batch_size, seq_len, hidden_size] - 历史序列
            mask: [batch_size, seq_len] - 1 for valid, 0 for padding

        Returns:
            tokens: [batch_size, num_tokens, hidden_size] - 生成的 aspect tokens
            reconstructed: [batch_size, hidden_size] - 重构的 target embedding
        """
        target_emb = target_emb.detach()
        batch_size = target_emb.size(0)

        # 转换 mask: True 表示 padding/ignore
        key_padding_mask = (mask == 0)

        # ===== 递进式 History Encoding + Decoder Layer 叠加 =====
        # 每层结构：HistoryEncoder → Cross-Attention → Self-Attention → FFN

        # 添加初始位置编码到history
        seq_len = history_emb.size(1)
        positions = torch.arange(seq_len, device=history_emb.device)
        h = history_emb + self.history_pos_emb(positions).unsqueeze(0)

        # 第一层：递进式History编码 + Cross-Attention
        # 1. 第一层History Encoding
        h = self.history_enc_layers[0](h, mask)

        # 2. Cross-Attention（不使用残差连接，强制模型学习用target查询history）
        context, _ = self.cross_attn_layers[0](
            target_emb=target_emb,  # [B, hidden_size]
            history_emb=h,  # 使用递进式编码后的history
            key_padding_mask=key_padding_mask
        )
        # context: [B, num_tokens, hidden_size]
        context = self.cross_attn_dropouts[0](self.cross_attn_norms[0](context))
        # 第一层不使用残差连接：强制模型通过Cross-Attention学习target-history交互，
        # 而不是直接将target信息传递到后续层

        # 3. Self-Attention + FFN（TransformerEncoderLayer 内部有残差连接）
        context = self.self_attn_layers[0](context)

        # 后续层：递进式History编码 + 使用残差连接
        for layer_idx in range(1, self.num_decoder_layers):
            # 1. 递进式History Encoding：在上层history基础上继续编码
            h = self.history_enc_layers[layer_idx](h, mask)

            # 2. Cross-Attention with residual
            residual = context
            normed_context = self.cross_attn_norms[layer_idx](context)
            new_context, _ = self.cross_attn_layers[layer_idx](
                target_emb=normed_context,
                history_emb=h,  # 使用递进式编码后的history
                key_padding_mask=key_padding_mask
            )
            context = residual + self.cross_attn_dropouts[layer_idx](new_context)

            # 3. Self-Attention + FFN（TransformerEncoderLayer 内部有残差连接）
            context = self.self_attn_layers[layer_idx](context)

        # ===== 最终处理 =====
        # Gating
        tokens = self.norm_gate(context)

        gate_logits = self.dimension_gate(tokens)  # [B, num_tokens, hidden_size]
        # 使用温度参数控制softmax锐利程度（可选）
        gate_logits = gate_logits / torch.clamp(self.gate_temperature, min=0.1)
        # 在token维度上进行softmax，确保每个维度上所有token权重和为1
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, num_tokens, hidden_size]
        # 应用门控权重
        gated_tokens = tokens * gate_weights  # [B, num_tokens, hidden_size]

        # Reconstruction
        reconstructed = gated_tokens.sum(dim=1)  # [B, hidden_size]

        # 返回：
        # - tokens: [batch_size, num_tokens, hidden_size]
        # - reconstructed: [batch_size, hidden_size]
        return tokens, reconstructed


# ============ 4. DASD_DisMIR 主类 ============
from BasicModel import BasicModel
class DASD_DisMIR(BasicModel):
    """
    DASD-DisMIR 完整封装模型

    封装DisMIR (Student) + ContextGatedTokenizer (Teacher) + TargetAwareFusion + Loss计算
    提供统一的训练和推理接口。

    设计原则：
    - 保持DisMIR原有功能完全不变
    - 训练时使用完整框架（Teacher + Student + Fusion + Alignment）
    - 推理时仅使用DisMIR部分
    """

    def __init__(self, dismir_model, args):
        """
        初始化DASD_DisMIR模型

        Args:
            dismir_model: DisMIR模型实例（Student）
            args: 参数配置对象，需包含：
                - lambda_recon: 重构Loss权重
                - lambda_align: 对齐Loss权重
                - lambda_infonce: InfoNCE Loss权重
                - rlambda: DisMIR原有Attention Loss权重
                - dropout: dropout率
                - interest_num: 兴趣数量
                - hidden_size: 隐藏层维度
        """
        super(DASD_DisMIR, self).__init__(
            dismir_model.item_num,
            dismir_model.hidden_size,
            dismir_model.batch_size
        )

        self.dismir = dismir_model  # 原有的DisMIR模型（Student）

        # Loss权重（使用getattr确保兼容性，使用DASD调好的默认值）
        self.lambda_recon = getattr(args, 'lambda_recon', 0.1)
        self.lambda_align = getattr(args, 'lambda_align', 0.1)
        self.lambda_infonce = getattr(args, 'lambda_infonce', 0.1)
        self.rlambda = getattr(args, 'rlambda', 0.0)

        # InfoNCE温度参数（CLIP-style log自适应温度）
        self.infonce_logit_scale = nn.Parameter(torch.ones(1) * np.log(1/0.07))

        # 保持DisMIR的name属性，用于evaluate函数中的模型识别
        self.name = 'DASD-DisMIR'

        # 获取参数
        hidden_size = args.hidden_size
        interest_num = args.interest_num
        dropout = args.dropout

        # 初始化Tokenizer (Teacher模型)
        # 硬编码DASD调好的参数值
        num_decoder_layers = 1  # 简化为1层，加速pretrain收敛
        use_token_split = True   # ContextGatedTokenizer默认值

        # Pretrain阶段dropout默认设为0，便于稳定收敛
        pretrain_dropout = getattr(args, 'pretrain_dropout', 0.0)
        self.tokenizer = ContextGatedTokenizer(
            hidden_size=hidden_size,
            num_tokens=interest_num,
            num_heads=interest_num,
            dropout=pretrain_dropout,
            num_decoder_layers=num_decoder_layers,
            use_token_split=use_token_split
        )

        # 初始化TargetAwareFusion (目标感知融合层)
        # NOTE: 当前未使用，实际使用 dismir.read_out() 进行硬选择
        # self.fusion = TargetAwareFusion(hidden_size)

        # Loss模块
        self.chamfer_loss = ChamferLoss()

        # Partition-aware 模块
        self.partition_enhancer = PartitionEnhancer(
            hidden_size=hidden_size,
            temperature=getattr(args, 'partition_temperature', 0.5)
        )
        self.partition_enhancer_norm = nn.LayerNorm(hidden_size)  # 添加 LayerNorm
        self.partition_align_loss = PartitionAlignedLoss(
            hidden_size=hidden_size,
            temperature=getattr(args, 'partition_align_temperature', 1.0)
        )
        self.lambda_partition_align = getattr(args, 'lambda_partition_align', 0.3)

        # Teacher 专用 embedding（从预训练权重加载）
        self.teacher_embeddings = nn.Embedding(self.dismir.item_num, hidden_size)
        # 默认初始化为与 student 相同的值（可选，便于热启动）
        with torch.no_grad():
            self.teacher_embeddings.weight.copy_(self.dismir.embeddings.weight)

        self.reset_parameters()
    def forward(self, item_list, label_list, mask, times, device, train=True):
        """
        前向传播

        Args:
            item_list: 历史物品序列 (batch_size, seq_len)
            label_list: 目标物品 (batch_size,)
            mask: 序列mask (batch_size, seq_len)
            times: 时间信息元组 (time_matrix, adj_matrix)
            device: 设备
            train: 是否为训练模式

        Returns:
            训练模式:
                interests: 用户兴趣向量 (batch_size, interest_num, hidden_size)
                total_loss: 总损失
                loss_dict: 各Loss分量的字典
            推理模式:
                interests: 用户兴趣向量 (batch_size, interest_num, hidden_size)
                None
        """
        # 1. DisMIR前向传播（Student）
        dismir_output = self.dismir(item_list, label_list, mask, times, device, train=train)

        if not train:
            # 推理时直接返回DisMIR的输出
            # dismir_output在推理时是 (interests, None)
            return dismir_output[0], None

        # 训练模式：解包DisMIR输出
        # DisMIR forward返回: (interests, scores, atten, readout, selection)
        interests, scores, atten, readout, selection = dismir_output

        # 2. Tokenizer前向传播（Teacher）
        # 获取target和历史序列的嵌入（Teacher使用独立的teacher_embeddings）
        label_eb = self.teacher_embeddings(label_list).detach()  # (batch_size, hidden_size)
        history_eb = self.teacher_embeddings(item_list)  # (batch_size, seq_len, hidden_size)
        # 应用mask
        history_eb = history_eb * mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_size)

        # 【Partition 增强】对 history 进行 partition-aware 增强（带残差连接和LayerNorm）
        history_residual = self.partition_enhancer(history_eb)
        history_enhanced = self.partition_enhancer_norm(history_eb + history_residual)

        # Tokenizer生成tokens和重构target（使用增强后的 history）
        tokens, recon_target = self.tokenizer(label_eb, history_enhanced, mask)

        # 3. Target-Aware Fusion (Hard Selection - 与DisMIR一致)
        # 使用read_out方法选择与target最相关的单个兴趣
        fused_pred, _ = self.dismir.read_out(interests, label_eb)  # (batch_size, hidden_size)

        # 4. 计算所有Loss

        # 4.1 Main Loss (使用融合向量，基于BPR with hard negative)
        # 使用DisMIR的BPR loss计算，但用fused_pred替代readout
        main_loss = self.dismir.compute_bpr_loss_with_hard_negative(
            fused_pred, label_list, self.dismir.hard_neg_candidates
        )

        # 4.2 Reconstruction Loss
        # 确保Tokenizer能有效重构Target
        recon_loss = F.mse_loss(F.normalize(recon_target, dim=-1), F.normalize(label_eb, dim=-1))

        # 4.3 Alignment Loss (双向对齐)
        # Teacher和Student独立训练，只通过Loss进行知识蒸馏
        align_loss = self.chamfer_loss(tokens, interests)

        # 4.3b Partition 结构对齐损失
        # 强制 Token 和 Interest 在相同的 partition 维度上激活
        partition_align_loss = self.partition_align_loss(tokens, interests)

        # 4.4 InfoNCE Loss
        # 拉近同一target的tokens和对应label_emb的距离，拉远与batch内其他label_emb的距离
        infonce_loss = self.calculate_infonce_loss(tokens, label_eb, base_temperature=0.08)

        # 4.5 DisMIR原有分区损失（Partition Loss）
        # 使用确定性种子确保与DisMIR一致
        deterministic_seed = 42 + item_list.sum().item() % 10000
        partition_loss = self.dismir.compute_partition_loss(item_list, mask, seed=deterministic_seed)

        # 4.6 DisMIR原有路由正则化损失（如果rlambda > 0）
        atten_loss = 0.0
        if atten is not None and self.rlambda > 0:
            atten_loss = self.dismir.calculate_atten_loss(atten)

        # 5. 总Loss组合公式
        # total_loss = main_loss + lambda_recon * recon_loss + lambda_align * align_loss
        #              + lambda_partition_align * partition_align_loss
        #              + lambda_infonce * infonce_loss + partition_loss + rlambda * atten_loss
        total_loss = (
            main_loss +
            self.lambda_recon * recon_loss +
            self.lambda_align * align_loss +
            self.lambda_partition_align * partition_align_loss +
            self.lambda_infonce * infonce_loss +
            self.dismir.lambda_coef * partition_loss +
            self.rlambda * atten_loss
        )

        # 6. 返回结果和Loss详情
        loss_dict = {
            'main_loss': main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss,
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'align_loss': align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss,
            'partition_align_loss': partition_align_loss.item() if isinstance(partition_align_loss, torch.Tensor) else partition_align_loss,
            'infonce_loss': infonce_loss.item() if isinstance(infonce_loss, torch.Tensor) else infonce_loss,
            'partition_loss': partition_loss.item() if isinstance(partition_loss, torch.Tensor) else partition_loss,
            'atten_loss': atten_loss.item() if isinstance(atten_loss, torch.Tensor) else atten_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }

        return interests, total_loss, loss_dict

    def calculate_infonce_loss(self, tokens, label_eb, base_temperature=0.07):
        """
        计算InfoNCE Loss (使用CLIP-style log自适应温度)

        拉近同一target的tokens（多个）和对应label_emb的距离，
        拉远与batch内其他label_emb的距离。

        Args:
            tokens: Teacher生成的tokens (batch_size, num_tokens, hidden_size)
            label_eb: 目标物品嵌入 (batch_size, hidden_size)
            base_temperature: 基础温度参数，默认0.07

        Returns:
            infonce_loss: InfoNCE损失值
        """
        batch_size, num_tokens, hidden_size = tokens.shape

        # CLIP-style log自适应温度
        # 限制logit_scale的范围防止极端值: [log(1/100), log(100)]
        logit_scale = torch.clamp(self.infonce_logit_scale, min=np.log(1/100), max=np.log(100))
        # effective_scale = exp(logit_scale), 温度 = base_temperature / effective_scale
        # 或者直接: similarity * exp(logit_scale) / base_temperature
        effective_scale = torch.exp(logit_scale)

        # 归一化tokens和label_eb
        tokens_norm = F.normalize(tokens, dim=-1)  # (batch_size, num_tokens, hidden_size)
        label_eb_norm = F.normalize(label_eb, dim=-1)  # (batch_size, hidden_size)

        # 计算相似度矩阵
        tokens_flat = tokens_norm.view(batch_size * num_tokens, hidden_size)  # (batch_size * num_tokens, hidden_size)
        label_eb_all = label_eb_norm  # (batch_size, hidden_size)

        # 计算相似度: (batch_size * num_tokens, batch_size)
        # 使用自适应scale: similarity * effective_scale / base_temperature
        similarity = torch.matmul(tokens_flat, label_eb_all.t()) * effective_scale / base_temperature

        # 创建标签：每个token对应的正样本是其所属样本的label
        labels = torch.arange(batch_size, device=tokens.device).repeat_interleave(num_tokens)

        # 计算InfoNCE loss (交叉熵)
        infonce_loss = F.cross_entropy(similarity, labels)

        # 监控温度值（训练时）
        if self.training and batch_size % 100 == 0:  # 每隔一定batch打印
            current_temp = base_temperature / effective_scale.item()
            print(f"[InfoNCE] temp={current_temp:.4f}, logit_scale={logit_scale.item():.4f}")

        return infonce_loss

    def encode_with_teacher(self, item_list, label_list, mask, device):
        """
        使用 Teacher (Tokenizer) 编码用户历史，label 作为 target。

        用于 Pretrain 阶段的评估，生成 interest tokens 用于计算 recall。

        Args:
            item_list: 历史物品序列 (batch_size, seq_len)
            label_list: 目标物品 (batch_size,) - 作为 target 输入给 Teacher
            mask: 序列 mask (batch_size, seq_len)
            device: 设备

        Returns:
            tokens: (batch_size, num_tokens, hidden_size) - Teacher 生成的 tokens
        """
        # 获取 label 和 history 的 embedding（Teacher使用独立的teacher_embeddings）
        label_eb = self.teacher_embeddings(label_list).detach()  # (batch_size, hidden_size)
        history_eb = self.teacher_embeddings(item_list)  # (batch_size, seq_len, hidden_size)
        history_eb = history_eb * mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_size)

        # 应用 partition_enhancer
        history_residual = self.partition_enhancer(history_eb)
        history_enhanced = self.partition_enhancer_norm(history_eb + history_residual)

        # 调用 tokenizer 生成 tokens
        tokens, _ = self.tokenizer(label_eb, history_enhanced, mask)

        return tokens  # (batch_size, num_tokens, hidden_size)

    def forward_teacher_pretrain(self, item_list, label_list, mask, times, device):
        """
        Stage-1 Teacher-only forward pass.

        Only the Tokenizer (Teacher) is trained; the Student (DisMIR) is run in
        no-grad mode so its embeddings are still updated through the shared
        embedding table, but its routing/capsule parameters receive no gradient.

        Args:
            item_list:  historical item ids  (batch_size, seq_len)
            label_list: target item ids      (batch_size,)
            mask:       sequence mask        (batch_size, seq_len)
            times:      time info tuple      (time_matrix, adj_matrix)
            device:     torch device

        Returns:
            total_loss: scalar tensor (recon + infonce)
            loss_dict:  per-component loss values
        """
        # Teacher使用独立的teacher_embeddings
        label_eb = self.teacher_embeddings(label_list).detach()       # (B, D)
        history_eb = self.teacher_embeddings(item_list)      # (B, L, D)
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
        infonce_loss = self.calculate_infonce_loss(tokens, label_eb, base_temperature=0.5)

        total_loss = self.lambda_recon * recon_loss + self.lambda_infonce * infonce_loss

        # Partition loss using Teacher embeddings (not Student embeddings)
        deterministic_seed = 42 + item_list.sum().item() % 10000
        partition_loss = self.compute_partition_loss_with_embeddings(
            item_list, mask, self.teacher_embeddings, seed=deterministic_seed
        )

        weighted_partition_loss = self.dismir.lambda_coef * partition_loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_infonce * infonce_loss +
            weighted_partition_loss
        )

        # Calculate current temperature for logging
        # CLIP-style: temp = base_temperature / exp(logit_scale)
        base_temp = 0.5  # Same as used in calculate_infonce_loss for pretrain
        with torch.no_grad():
            logit_scale = torch.clamp(self.infonce_logit_scale, min=np.log(1/100), max=np.log(100))
            effective_scale = torch.exp(logit_scale)
            current_temp = base_temp / effective_scale.item()

        loss_dict = {
            'recon_loss':   recon_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'partition_loss': partition_loss.item(),
            'weighted_partition_loss': weighted_partition_loss.item(),
            'lambda_coef': self.dismir.lambda_coef,
            'total_loss':   total_loss.item(),
            'infonce_temp': current_temp,  # Current adaptive temperature
            'infonce_logit_scale': logit_scale.item(),  # Raw logit scale
        }
        return total_loss, loss_dict

    def compute_partition_loss_with_embeddings(self, items, mask, embeddings, seed=None):
        """
        Compute partition loss using specified embeddings (not self.dismir.embeddings).
        This allows Teacher to use its own embeddings during pretraining.

        Args:
            items: (batch_size, seq_len) item ids
            mask: (batch_size, seq_len) padding mask
            embeddings: nn.Embedding to use for lookup
            seed: Optional random seed for reproducible sampling

        Returns:
            partition_loss: scalar tensor
        """
        import numpy as np

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        batch_size, seq_len = items.shape
        partition_groups = self.dismir.partition_groups
        num_negatives = self.dismir.num_negatives
        item_num = self.dismir.item_num

        # Get item embeddings using provided embedding table
        item_eb = embeddings(items)  # (B, L, D)
        mask_expanded = mask.unsqueeze(-1).float()
        item_eb = item_eb * mask_expanded

        # Flatten for batch processing (B*L, K)
        item_eb_flat = item_eb.view(-1, partition_groups)
        item_weights = item_eb_flat

        # Sample positive neighbors from confidence matrix (reuse dismir's method)
        pos_samples = self.dismir.sample_positive_neighbors(items, num_pos=1)
        pos_samples_flat = pos_samples.view(-1)
        pos_eb = embeddings(pos_samples_flat)  # (B*L, K)

        # Sample negative items
        neg_samples = torch.randint(0, item_num,
                                    (batch_size * seq_len, num_negatives),
                                    device=items.device)
        neg_eb = embeddings(neg_samples)  # (B*L, N, K)

        pos_weights = pos_eb
        neg_weights = neg_eb

        # Compute similarities
        pos_sim = (item_weights * pos_weights).sum(dim=-1)  # (B*L,)
        neg_sim = torch.matmul(
            item_weights.unsqueeze(1),
            neg_weights.transpose(-2, -1)
        ).squeeze(1)  # (B*L, N)

        # InfoNCE loss
        temperature = 1.0
        pos_sim = torch.clamp(pos_sim / temperature, min=-10, max=10)
        neg_sim = torch.clamp(neg_sim / temperature, min=-10, max=10)

        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim).sum(dim=-1)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-9))

        # Apply mask
        mask_flat = mask.view(-1).float()
        loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-9)

        if torch.isnan(loss):
            print(f"[Warning] partition_loss is NaN, returning 0")
            return torch.tensor(0.0, device=items.device)

        return loss

    # ============ 兼容DisMIR接口的方法 ============

    def set_device(self, device):
        """设置设备（兼容DisMIR接口）"""
        if hasattr(self.dismir, 'set_device'):
            self.dismir.set_device(device)
        self.device = device

    def set_sampler(self, args, device=None):
        """设置采样器（兼容DisMIR接口）"""
        if hasattr(self.dismir, 'set_sampler'):
            self.dismir.set_sampler(args, device=device)

    def output_items(self):
        """输出物品嵌入（用于评估）"""
        return self.dismir.output_items()

    def load_confidence_matrix(self, dataset_name, data_path='./data/'):
        """加载置信度矩阵（兼容DisMIR接口）"""
        return self.dismir.load_confidence_matrix(dataset_name, data_path)

    def calculate_disloss(self, readout, pos_items, selection, interests, atten):
        """
        计算DisMIR损失（兼容DisMIR接口，但这里不直接使用）
        实际的损失计算在forward中完成
        """
        return self.dismir.calculate_disloss(readout, pos_items, selection, interests, atten)

    # NOTE: 该方法当前未被使用，实际加载Teacher权重使用 evalution.py 中的 load_teacher_weights 函数
    # 保留代码以备后续可能需要独立加载功能
    '''
    def load_teacher_from_pretrain(self, checkpoint_path):
        """
        从预训练 checkpoint 加载 Teacher 的 embedding 权重

        Args:
            checkpoint_path: 预训练 checkpoint 路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 尝试从 checkpoint 中获取 embedding 权重
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 查找 embedding 权重（可能是 'dismir.embeddings.weight' 或 'embeddings.weight'）
        embedding_key = None
        for key in state_dict.keys():
            if 'embeddings.weight' in key and 'teacher' not in key:
                embedding_key = key
                break

        if embedding_key and embedding_key in state_dict:
            self.teacher_embeddings.weight.data.copy_(state_dict[embedding_key])
            print(f"Loaded teacher embedding from {embedding_key}")
        else:
            print("Warning: Could not find embedding weights in checkpoint")
    '''
