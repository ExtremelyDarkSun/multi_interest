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

    def __init__(self, alpha_t_to_i: float = 0.3):
        """
        Args:
            alpha_t_to_i: token→interest 方向的权重（小于1），
                          interest→token 方向权重固定为1.0。
                          设计动机：不强求所有 token 被 interest 全覆盖，
                          但每个 interest 必须能在 token 集合中找到对应。
        """
        super(ChamferLoss, self).__init__()
        self.alpha_t_to_i = alpha_t_to_i

    def forward(self, tokens, interests):
        """
        非对称双向 Chamfer Distance

        Args:
            tokens:    Teacher 生成的 token 集合  (B, K, D)
            interests: Student 生成的兴趣向量集合 (B, M, D)

        Returns:
            loss: 标量
        """
        tokens    = F.normalize(tokens,    p=2, dim=-1)
        interests = F.normalize(interests, p=2, dim=-1)

        # dist_matrix: (B, K, M)，K=token数，M=interest数
        dist_matrix = torch.cdist(tokens, interests, p=2)

        # interest→token（主方向）：每个 interest 找最近的 token
        # min over K(token) → (B, M)
        min_i_to_t, _ = dist_matrix.min(dim=1)   # (B, M)
        loss_i_to_t   = min_i_to_t.mean()

        # token→interest（次方向，小权重）：每个 token 找最近的 interest
        # min over M(interest) → (B, K)
        min_t_to_i, _ = dist_matrix.min(dim=2)   # (B, K)
        loss_t_to_i   = min_t_to_i.mean()

        return loss_i_to_t + self.alpha_t_to_i * loss_t_to_i




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

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] - 输入序列
            mask: [batch_size, seq_len] - 1=valid, 0=padding
        Returns:
            encoded: [batch_size, seq_len, hidden_size]
        """
        # 处理mask: Transformer需要key_padding_mask (True=padding)
        key_pad_mask = (mask == 0) if mask is not None else None

        # Transformer encoding (内部已包含Self-Attn + FFN + Residual + LayerNorm)
        return self.layer(x, src_key_padding_mask=key_pad_mask)


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
       - 先通过 target_proj 映射到 num_tokens * hidden_size，再切分成 num_tokens 个 query
       - 输出多个 tokens [B, num_tokens, D]

    2. 后续层模式 (is_first_layer=False):
       - 输入多个 tokens [B, num_tokens, D]
       - 输出多个 tokens [B, num_tokens, D]
       - 每个 token 作为独立的 query 进行 Cross-Attention
    """

    def __init__(self, hidden_size, num_tokens, num_key_value_heads=None, dropout=0.1,
                 is_first_layer=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.num_heads = num_tokens
        self.num_key_value_heads = num_key_value_heads or num_tokens
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = hidden_size ** -0.5
        self.attention_dropout = dropout
        self.is_first_layer = is_first_layer

        # Query 投影：根据 is_first_layer 选择不同的投影方式
        if is_first_layer:
            # 第一层：target_emb [B, D] -> [B, num_tokens * D]，再切分为 num_tokens 个 query
            self.q_proj = nn.Linear(hidden_size, num_tokens * hidden_size, bias=False)
        else:
            # 后续层：输入已经是 [B, num_tokens, D]，每个 token 独立映射
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
            # 第一层：target_emb [B, D] -> [B, num_tokens * D]，再切分为 num_tokens 个 query
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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25,
                 revival_threshold: int = 1):
        """
        revival_threshold: codes used < threshold per batch will be reset to random input vectors.
        Set to 0 to disable dead code revival.
        """
        super().__init__()
        self.embedding_dim      = embedding_dim
        self.num_embeddings     = num_embeddings
        self.commitment_cost    = commitment_cost
        self.revival_threshold  = revival_threshold
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, tokens):          # tokens: [B, K, D]
        B, K, D = tokens.shape
        flat = tokens.view(-1, D)       # [B*K, D]
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2.0 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))   # [B*K, C]
        idx  = dist.argmin(dim=1)                        # [B*K]
        q    = self.codebook(idx)                        # [B*K, D]

        codebook_loss = F.mse_loss(q.detach(), flat)
        commit_loss   = F.mse_loss(q, flat.detach())
        vq_loss       = self.commitment_cost * codebook_loss + commit_loss

        # Dead code revival: reset rarely-used codes to random input vectors
        if self.training and self.revival_threshold > 0:
            # Count usage for each code in this batch
            usage = torch.zeros(self.num_embeddings, device=flat.device)
            unique_idx, counts = idx.unique(return_counts=True)
            usage[unique_idx] = counts.float()
            dead_codes = (usage < self.revival_threshold).nonzero(as_tuple=True)[0]
            if len(dead_codes) > 0:
                n_dead = dead_codes.shape[0]
                rand_idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
                with torch.no_grad():
                    self.codebook.weight[dead_codes] = flat[rand_idx].detach()

        q_st = flat + (q - flat).detach()               # straight-through
        return q_st.view(B, K, D), vq_loss, idx.view(B, K)


class ContextGatedTokenizer(nn.Module):
    """
    Context-Gated Aspect Tokenizer (Teacher模型) - 使用 Qwen3NextAttention

    升级点:
    1. 使用 Qwen3NextAttention 替换原有的 MultiheadAttention
    2. 集成 RoPE 位置编码
    3. 加入 Q/K 归一化和门控机制
    4. num_tokens 直接对应注意力头数
    5. 第一层将 target_emb 映射到 num_tokens * hidden_size 再切分为 num_tokens 个 query
    6. 支持多层叠加，每层包含：Cross-Attention + Self-Attention + FFN（标准Decoder Layer）
    7. 第一层不使用残差连接（强制模型学习用target查询history，避免target信息直接传递），后续层使用残差连接
    """

    def __init__(self, hidden_size, num_tokens=4, num_heads=4, num_key_value_heads=None,
                 dropout=0.1, num_decoder_layers=4, num_interaction_layers=1,
                 num_embeddings=256, vq_commitment_cost=0.25):
        super(ContextGatedTokenizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
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

        # --- Step 5: Token 互注意力组合模块（替代 recon_proj）---
        self.combine_q   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.combine_k   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.combine_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.combine_scale = hidden_size ** -0.5

        self.dimension_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        # 可选：添加温度参数来控制softmax的锐利程度
        self.gate_temperature = nn.Parameter(torch.ones(1))

        self.vq = VectorQuantizer(num_embeddings, hidden_size, vq_commitment_cost, revival_threshold=0)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def combine_tokens(self, tokens: torch.Tensor,
                       override_weights: torch.Tensor = None) -> torch.Tensor:
        """
        用 token 互注意力组合 K 个 token → 单向量表示

        Args:
            tokens:           [B, K, D]
            override_weights: [B, K, K] 可选。若传入则跳过学习到的注意力权重，
                              直接使用此矩阵（用于生成 hard_false_target）
        Returns:
            combined: [B, D]
        """
        if override_weights is None:
            Q = self.combine_q(tokens)                                    # [B, K, D]
            K = self.combine_k(tokens)                                    # [B, K, D]
            attn = torch.bmm(Q, K.transpose(1, 2)) * self.combine_scale  # [B, K, K]
            weights = F.softmax(attn, dim=-1)                             # [B, K, K]
        else:
            weights = override_weights                                    # [B, K, K]

        out = torch.bmm(weights, tokens).mean(dim=1)   # [B, D]（K 个输出取均值）
        return self.combine_out(out)                   # [B, D]

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

        # VQ 量化（保留 idx 用于 codebook 利用率统计）
        quantized_tokens, vq_loss, vq_indices = self.vq(gated_tokens)   # [B, K, D], scalar, [B, K]

        # 互注意力组合 K tokens → recon
        reconstructed = self.combine_tokens(quantized_tokens)   # [B, D]

        # 返回：
        # - quantized_tokens: [B, K, D]
        # - reconstructed:    [B, D]
        # - vq_loss:          scalar
        # - vq_indices:       [B, K]  codebook entry indices，用于利用率统计
        return quantized_tokens, reconstructed, vq_loss, vq_indices


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
        self.lambda_recon = getattr(args, 'lambda_recon', 0.1)       # teacher_mse 权重（预训练 & 微调）
        self.lambda_bpr   = getattr(args, 'lambda_bpr',   1.0)   # DisMIR BPR loss 权重
        self.lambda_align = getattr(args, 'lambda_align', 0.1)  # ChamferLoss 对齐权重
        self.rlambda = getattr(args, 'rlambda', 0.0)
        # 保持DisMIR的name属性，用于evaluate函数中的模型识别
        self.name = 'DASD-DisMIR'

        # 获取参数
        hidden_size = args.hidden_size
        interest_num = args.interest_num
        dropout = args.dropout

        # 初始化Tokenizer (Teacher模型)
        # 硬编码DASD调好的参数值
        num_decoder_layers = 1  # 简化为1层，加速pretrain收敛

        # Pretrain阶段dropout默认设为0，便于稳定收敛
        pretrain_dropout = getattr(args, 'pretrain_dropout', 0.0)
        self.tokenizer = ContextGatedTokenizer(
            hidden_size=hidden_size,
            num_tokens=interest_num,
            num_heads=interest_num,
            dropout=pretrain_dropout,
            num_decoder_layers=num_decoder_layers,
            num_embeddings=getattr(args, 'vq_num_embeddings', 5000),
            vq_commitment_cost=getattr(args, 'vq_commitment_cost', 0.25),
        )

        # DDPM 线性噪声调度
        noise_T          = getattr(args, 'noise_T',          40)
        noise_beta_start = getattr(args, 'noise_beta_start', 1e-4)
        noise_beta_end   = getattr(args, 'noise_beta_end',   0.02)
        betas      = torch.linspace(noise_beta_start, noise_beta_end, noise_T)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('ddpm_alphas_bar', alphas_bar)
        self.ddpm_T    = noise_T
        self.lambda_vq = getattr(args, 'lambda_vq', 0.1)
        self.lambda_false            = getattr(args, 'lambda_false',            0.5)
        self.num_false_weight_sample = getattr(args, 'num_false_weight_sample', 10)

        # 初始化TargetAwareFusion (目标感知融合层)
        # NOTE: 当前未使用，实际使用 dismir.read_out() 进行硬选择
        # self.fusion = TargetAwareFusion(hidden_size)

        # Loss模块 (ChamferLoss deprecated, kept for compatibility but not used)
        self.chamfer_loss = ChamferLoss()

        # Training phase control: 'pretrain' or 'finetune'
        self.training_phase = getattr(args, 'training_phase', 'finetune')

        # Pretrain阶段冻结Student相关组件（shared embedding保持可训练）
        if self.training_phase == 'pretrain':
            for name, param in self.dismir.named_parameters():
                if 'embeddings' not in name:
                    param.requires_grad = False

        self.reset_parameters()

    def _apply_ddpm_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        DDPM 前向加噪：x_t = sqrt(ᾱ_t)*x + sqrt(1-ᾱ_t)*ε
        t ~ Uniform[0, T)，每个 batch 元素独立采样
        x 应已 detach（调用方负责）
        """
        B     = x.size(0)
        t     = torch.randint(0, self.ddpm_T, (B,), device=x.device)
        ab    = self.ddpm_alphas_bar[t].unsqueeze(-1)          # [B, 1]
        eps   = torch.randn_like(x)
        return ab.sqrt() * x + (1.0 - ab).sqrt() * eps        # [B, D]

    def forward(self, item_list, label_list, mask, times, device, train=True, future_labels=None):
        """
        前向传播 (Refactored for Dynamic Selection Distillation)

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
        # === Pretrain Phase: Only train Teacher ===
        if train and self.training_phase == 'pretrain':
            return self.forward_teacher_pretrain(item_list, label_list, mask, times, device)

        # === Finetune Phase: Student + Teacher 联合训练 ===

        # 1. Student 前向
        dismir_output = self.dismir(item_list, label_list, mask, times, device, train=train)

        if not train:
            return dismir_output[0], None

        # DisMIR forward 返回: (interests, scores, atten, readout, selection)
        interests, scores, atten, readout, selection = dismir_output

        # 2. Teacher 前向（Stage-2 tokenizer 可训练，始终加噪）
        label_eb   = self.dismir.embeddings(label_list).detach()   # (B, D)
        history_eb = self.dismir.embeddings(item_list).detach()    # (B, L, D)
        history_eb = history_eb * mask.unsqueeze(-1)

        tokenizer_input = self._apply_ddpm_noise(label_eb) if self.training else label_eb
        quantized_tokens, recon_target, vq_loss, _ = self.tokenizer(
            tokenizer_input, history_eb, mask
        )   # quantized_tokens: (B, K, D)

        # 3. Teacher MSE loss（约束 tokenizer 重构目标物品）
        teacher_mse = F.mse_loss(
            F.normalize(recon_target, dim=-1),
            F.normalize(label_eb, dim=-1)
        )

        # 4. DisMIR BPR（使用 DisMIR 原始 hard negative BPR，shared negatives）
        dismir_bpr = self.dismir.compute_bpr_loss_with_hard_negative(
            readout, label_list, self.dismir.hard_neg_candidates
        )

        # 4b. Hard false target BPR
        #     用同一批 quantized_tokens + num_false_weight_sample 组随机权重 → 多个错误配比假目标
        if self.lambda_false > 0:
            B_   = quantized_tokens.shape[0]
            K_   = quantized_tokens.shape[1]
            N_f  = self.num_false_weight_sample  # 默认 10

            # 为每个样本采样 N_f 组 [K, K] 随机权重 → [B, N_f, K, K]
            rand_logits  = torch.rand(B_, N_f, K_, K_, device=device)
            rand_weights = F.softmax(rand_logits, dim=-1)            # [B, N_f, K, K]

            with torch.no_grad():
                # 对 N_f 组权重批量调用 combine_tokens
                tokens_exp  = quantized_tokens.unsqueeze(1).expand(
                                  -1, N_f, -1, -1
                              ).reshape(B_ * N_f, K_, -1)           # [B*N_f, K, D]
                rand_w_flat = rand_weights.reshape(B_ * N_f, K_, K_) # [B*N_f, K, K]
                false_flat  = self.tokenizer.combine_tokens(
                                  tokens_exp, override_weights=rand_w_flat
                              )                                       # [B*N_f, D]
                hard_false_targets = false_flat.reshape(B_, N_f, -1) # [B, N_f, D]

            pos_eb_false  = self.dismir.embeddings(label_list)            # [B, D]，有梯度
            pos_scores_f  = (readout * pos_eb_false).sum(dim=-1)          # [B]
            # 对 N_f 个假目标打分：[B, N_f]
            false_scores  = torch.einsum(
                'bd,bnd->bn', readout, hard_false_targets
            )                                                              # [B, N_f]

            # 取最难（最高分）负样本
            hardest_false, _ = false_scores.max(dim=-1)                   # [B]
            hard_diff    = torch.clamp(pos_scores_f - hardest_false, -20, 20)
            hard_false_loss = -F.logsigmoid(hard_diff)                    # [B]

            # 所有负样本均值
            all_false_diff  = torch.clamp(
                pos_scores_f.unsqueeze(1) - false_scores, -20, 20
            )                                                              # [B, N_f]
            all_false_loss  = -F.logsigmoid(all_false_diff).mean(dim=-1)  # [B]

            false_bpr = (hard_false_loss + all_false_loss).mean()
        else:
            false_bpr = torch.tensor(0.0, device=device)
            hard_false_loss = torch.zeros(1, device=device)
            all_false_loss  = torch.zeros(1, device=device)

        # 5. ChamferLoss：aspect token 集合 ↔ student interest 集合
        chamfer_loss_val = self.chamfer_loss(quantized_tokens, interests)

        # 6. DisMIR 分区损失（可选）
        partition_loss = 0.0
        if hasattr(self.dismir, 'lambda_coef') and self.dismir.lambda_coef > 0:
            deterministic_seed = 42 + item_list.sum().item() % 10000
            partition_loss = self.dismir.compute_partition_loss(
                item_list, mask, seed=deterministic_seed
            )

        # 7. 路由正则化（可选）
        atten_loss = 0.0
        if atten is not None and self.rlambda > 0:
            atten_loss = self.dismir.calculate_atten_loss(atten)

        # 8. 总 Loss
        total_loss = (
            self.lambda_bpr   * dismir_bpr   +
            self.lambda_false * false_bpr    +
            self.lambda_recon * teacher_mse +
            self.lambda_align * chamfer_loss_val +
            self.dismir.lambda_coef * partition_loss +
            self.rlambda      * atten_loss +
            self.lambda_vq    * vq_loss
        )

        loss_dict = {
            'dismir_bpr':      dismir_bpr.item(),
            'false_bpr':       false_bpr.item(),
            'false_bpr_hard':  hard_false_loss.mean().item(),
            'false_bpr_all':   all_false_loss.mean().item(),
            'teacher_mse':     teacher_mse.item(),
            'chamfer_loss':    chamfer_loss_val.item(),
            'vq_loss':         vq_loss.item(),
            'partition_loss':  partition_loss.item() if isinstance(partition_loss, torch.Tensor) else partition_loss,
            'atten_loss':      atten_loss.item() if isinstance(atten_loss, torch.Tensor) else atten_loss,
            'total_loss':      total_loss.item(),
        }

        return interests, total_loss, loss_dict


    def encode_with_teacher(self, item_list, label_list, mask, device):
        """
        使用 Teacher (Tokenizer) 编码用户历史，label 作为 target。

        用于 Pretrain 阶段的评估，生成单个 target-aware token 用于计算 recall。

        Args:
            item_list: 历史物品序列 (batch_size, seq_len)
            label_list: 目标物品 (batch_size,) - 作为 target 输入给 Teacher
            mask: 序列 mask (batch_size, seq_len)
            device: 设备

        Returns:
            recon_target: (batch_size, hidden_size) - Teacher 生成的 target 表示 (单token)
        """
        # 获取 label 和 history 的 embedding（使用统一的 dismir.embeddings）
        label_eb = self.dismir.embeddings(label_list).detach()  # (batch_size, hidden_size)
        history_eb = self.dismir.embeddings(item_list)  # (batch_size, seq_len, hidden_size)
        history_eb = history_eb * mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_size)

        # 调用 tokenizer 生成 tokens、recon_target 和 VQ indices
        quantized_tokens, recon_target, _vq_loss, vq_indices = self.tokenizer(
            label_eb, history_eb, mask
        )

        # 归一化以匹配训练时的MSE loss（F.normalize）
        recon_target = F.normalize(recon_target, dim=-1)

        return recon_target, vq_indices  # (B, D), (B, K)

    def forward_teacher_pretrain(self, item_list, label_list, mask, times, device):
        """
        Stage-1 Teacher-only forward pass (Simplified).
        MSE loss + Partition loss for better Teacher quality.
        """
        # 使用统一的 dismir.embeddings（pretrain阶段不detach history_eb，让tokenizer重构梯度训练embedding）
        label_eb   = self.dismir.embeddings(label_list).detach()  # (B, D)
        history_eb = self.dismir.embeddings(item_list)            # (B, L, D)
        history_eb = history_eb * mask.unsqueeze(-1)

        # [改动] 加噪（防平凡解：tokenizer 不能直接透传 label_eb）
        noisy_label_eb = self._apply_ddpm_noise(label_eb)

        quantized_tokens, recon_target, vq_loss, _ = self.tokenizer(noisy_label_eb, history_eb, mask)

        # Reconstruction/MSE loss
        recon_loss = F.mse_loss(
            F.normalize(recon_target, dim=-1),
            F.normalize(label_eb.detach(), dim=-1)
        )

        # partition_loss is excluded from Stage-1 pretraining.
        # It has no tokenizer parameters in its computation graph, so it cannot
        # improve reconstruction recall; it only destabilises teacher_embeddings
        # and causes recall to collapse after the initial peak.
        # partition_loss is re-introduced in Stage-2 joint training via DisMIR.
        weighted_recon_loss = self.lambda_recon * recon_loss
        weighted_vq_loss    = self.lambda_vq    * vq_loss
        total_loss          = weighted_recon_loss + weighted_vq_loss

        loss_dict = {
            'recon_loss':          recon_loss.item(),
            'weighted_recon_loss': weighted_recon_loss.item(),
            'vq_loss':             vq_loss.item(),
            'weighted_vq_loss':    weighted_vq_loss.item(),
            'total_loss':          total_loss.item(),
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
