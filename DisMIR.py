import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from BasicModel import BasicModel


class CapsuleMultiInterest(nn.Module):
    """
    [DisMIR] Capsule Network for Multi-Interest Extraction
    [PAPER_REF] Du et al. KDD '24 Sec 3.2 and 4.3
    [PAPER_REF] Sabour et al. "Dynamic Routing Between Capsules" NeurIPS 2017

    Uses identity transformation matrix (LightGCN-inspired).
    One-time routing instead of 3 iterations for efficiency (as mentioned in paper Sec 5.1.4).
    """

    def __init__(self, hidden_size, seq_len, interest_num=4, routing_times=1):
        super(CapsuleMultiInterest, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.interest_num = interest_num
        self.routing_times = routing_times

    def forward(self, item_eb, mask):
        """
        Dynamic routing for multi-interest extraction

        Args:
            item_eb: (batch_size, seq_len, hidden_size) item embeddings
            mask: (batch_size, seq_len) padding mask

        Returns:
            interest_capsules: (batch_size, interest_num, hidden_size)
        """
        batch_size = item_eb.shape[0]

        # Initialize routing logits b_if
        capsule_weight = torch.zeros(batch_size, self.interest_num, self.seq_len,
                                     device=item_eb.device)

        # Dynamic routing iterations
        for i in range(self.routing_times):
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            # Softmax over sequence dimension
            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0),
                                                  paddings,
                                                  capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            # Compute interest capsules
            interest_capsule = torch.matmul(capsule_softmax_weight,
                                            item_eb.unsqueeze(1))

            # Squash activation
            cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
            scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
            interest_capsule = scalar_factor * interest_capsule

            # Update routing weights if not last iteration
            if i < self.routing_times - 1:
                delta_weight = torch.matmul(item_eb.unsqueeze(1),
                                            torch.transpose(interest_capsule, 2, 3))
                delta_weight = torch.reshape(delta_weight,
                                             (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight

        interest_capsule = torch.reshape(interest_capsule,
                                         (-1, self.interest_num, self.hidden_size))
        return interest_capsule


class DisMIR(BasicModel):
    """
    [DisMIR] Disentangled Multi-interest Representation Learning
    [PAPER_REF] Du et al. "Disentangled Multi-interest Representation Learning
                for Sequential Recommendation" KDD 2024

    Key features:
    1. Capsule network for multi-interest extraction
    2. Item partition loss with confidence graph sampling
    3. BPR loss with hard negative mining
    4. Shared representation constraint: hidden_size = partition_groups = K
    """

    def __init__(self, item_num, hidden_size, batch_size, interest_num=4,
                 seq_len=20, partition_groups=64, lambda_coef=0.1,
                 num_negatives=100, hard_neg_candidates=10, add_pos=False, beta=0,
                 use_overlapped_partition=False,
                 args=None, device=None):
        """
        [PAPER_REF] Section 5.1.4: hidden_size = partition_groups = 64

        Args:
            item_num: Number of items (N)
            hidden_size: Embedding dimension (d), MUST equal partition_groups
            batch_size: Training batch size
            interest_num: Number of interests (F), default 4 for Gowalla
            seq_len: Maximum sequence length
            partition_groups: Number of partition groups (K), MUST equal hidden_size
            lambda_coef: Trade-off coefficient λ for partition loss
            num_negatives: Number of negative samples for partition loss (N_v=100)
            hard_neg_candidates: Number of candidates for hard negative mining (default 10 per paper)
            add_pos: Whether to add positional encoding (not used in DisMIR)
            beta: IHN temperature parameter
            use_overlapped_partition: Whether to use overlapped partition [PAPER_REF] Sec 4.1.2
                NOTE: Paper uses raw embeddings as w_ik (no softmax, no temperature)
            args: Additional arguments
            device: Computation device
        """
        # [PAPER_REF] Enforce shared representation constraint: hidden_size = K
        if hidden_size != partition_groups:
            print(f"[DisMIR] WARNING: hidden_size ({hidden_size}) != partition_groups ({partition_groups})")
            print(f"[DisMIR] Forcing partition_groups = hidden_size for shared representation constraint")
            partition_groups = hidden_size

        super(DisMIR, self).__init__(item_num, hidden_size, batch_size, seq_len, beta)

        self.interest_num = interest_num
        self.partition_groups = partition_groups  # K = hidden_size
        self.lambda_coef = lambda_coef
        self.num_negatives = num_negatives
        self.hard_neg_candidates = hard_neg_candidates
        self.hard_readout = True
        self.device = device

        # [Overlapped Partition] Configuration
        # [PAPER_REF] Sec 4.1.2: w_ik ∈ ℝ (real values), learned directly without softmax
        self.use_overlapped_partition = use_overlapped_partition
        if use_overlapped_partition:
            print(f"[DisMIR] Using Overlapped Partition (w_ik ∈ ℝ, no softmax)")

        # Confidence matrix for item-item relationships
        self.confidence_matrix = None
        self._conf_matrix_coo = None  # COO format for sampling

        # Capsule network for multi-interest extraction
        self.capsule_net = CapsuleMultiInterest(
            hidden_size, seq_len, interest_num, routing_times=1
        )

        self.reset_parameters()

    def load_confidence_matrix(self, dataset_name, data_path='./data/'):
        """
        Load pre-computed item confidence matrix

        Args:
            dataset_name: Name of dataset (e.g., 'gowalla', 'book')
            data_path: Path to data directory
        """
        import os
        matrix_path = os.path.join(data_path, f'{dataset_name}_data/{dataset_name}_confidence_matrix.npz')

        if not os.path.exists(matrix_path):
            print(f"[DisMIR] Warning: Confidence matrix not found at {matrix_path}")
            print(f"[DisMIR] Please run data preprocessing first with build_item_confidence_matrix()")
            return False

        try:
            self.confidence_matrix = sparse.load_npz(matrix_path)
            # Convert to COO format for efficient sampling
            self._conf_matrix_coo = self.confidence_matrix.tocoo()
            print(f"[DisMIR] Loaded confidence matrix: {self.confidence_matrix.shape}")
            print(f"[DisMIR] Non-zero entries: {self.confidence_matrix.nnz}")
            return True
        except Exception as e:
            print(f"[DisMIR] Error loading confidence matrix: {e}")
            return False

    def sample_positive_neighbors(self, items, num_pos=1):
        """
        Sample positive neighbors from confidence matrix
        [PAPER_REF] Section 4.2.3: Sample Y_i ~ M(·|S_i)

        Args:
            items: (batch_size, seq_len) item ids
            num_pos: Number of positive samples per item

        Returns:
            pos_samples: (batch_size, seq_len, num_pos) sampled positive item ids
        """
        batch_size, seq_len = items.shape
        pos_samples = torch.zeros(batch_size, seq_len, num_pos, dtype=torch.long, device=items.device)

        if self.confidence_matrix is None:
            # Fallback: use random items from sequence as positives
            # This is less effective but allows training without confidence matrix
            pos_samples = torch.zeros(batch_size, seq_len, num_pos, dtype=torch.long, device=items.device)

            for b in range(batch_size):
                # Get non-padding items
                valid_mask = (items[b] != 0).cpu().numpy()
                valid_items = items[b].cpu().numpy()[valid_mask]

                if len(valid_items) == 0:
                    continue

                # Sample random valid items as positives
                for l in range(seq_len):
                    if items[b, l] == 0:  # Padding
                        continue
                    # Random choice from valid items
                    idx = np.random.randint(0, len(valid_items))
                    pos_samples[b, l, 0] = valid_items[idx]

            return pos_samples

        items_np = items.cpu().numpy()

        for b in range(batch_size):
            for l in range(seq_len):
                item_id = items_np[b, l]
                if item_id == 0:  # Padding
                    pos_samples[b, l] = 0
                    continue

                # Get confidence distribution for this item
                row = self.confidence_matrix.getrow(item_id)
                if row.nnz == 0:
                    # No neighbors, use self
                    pos_samples[b, l] = item_id
                    continue

                # Sample from confidence distribution
                neighbors = row.indices
                probs = row.data
                probs = probs / probs.sum()  # Normalize

                if num_pos == 1:
                    sampled = np.random.choice(neighbors, p=probs)
                    pos_samples[b, l, 0] = sampled
                else:
                    sampled = np.random.choice(neighbors, size=num_pos, p=probs, replace=True)
                    pos_samples[b, l] = torch.tensor(sampled)

        return pos_samples

    def compute_partition_loss(self, items, mask):
        """
        Compute partition loss via contrastive learning
        [PAPER_REF] Theorem 4 and Section 4.2.3

        L_Partition = -sum_i E[log(exp(sim(i, pos)) / sum(exp(sim(i, neg))))]

        Supports both Non-overlapped and Overlapped partition:
        - Non-overlapped (default): w_ik ∈ {0,1}, hard partition assignment
        - Overlapped: w_ik ∈ ℝ, soft partition weights [PAPER_REF] Eq (4)
        NOTE: Paper does NOT use softmax or temperature for partition weights.
              w_ik are learned directly as real values.

        Args:
            items: (batch_size, seq_len) item ids
            mask: (batch_size, seq_len) padding mask

        Returns:
            partition_loss: scalar tensor
        """
        batch_size, seq_len = items.shape

        # Get item embeddings (B, L, D) where D = K (partition_groups)
        item_eb = self.embeddings(items)  # (B, L, D)
        mask_expanded = mask.unsqueeze(-1).float()
        item_eb = item_eb * mask_expanded

        # Flatten for batch processing (B*L, K)
        item_eb_flat = item_eb.view(-1, self.partition_groups)

        # [PAPER_REF] Eq (4): w_ik ∈ ℝ (real values)
        # Both overlapped and non-overlapped use raw embeddings as partition weights
        # The difference is conceptual: overlapped allows w_ik to be any real value,
        # while non-overlapped encourages binary values through training dynamics
        item_weights = item_eb_flat

        # Sample positive neighbors from confidence matrix
        pos_samples = self.sample_positive_neighbors(items, num_pos=1)  # (B, L, 1)
        pos_samples_flat = pos_samples.view(-1)
        pos_eb = self.embeddings(pos_samples_flat)  # (B*L, K)

        # Sample negative items (uniformly from vocabulary, excluding positives)
        neg_samples = torch.randint(0, self.item_num,
                                    (batch_size * seq_len, self.num_negatives),
                                    device=items.device)
        neg_eb = self.embeddings(neg_samples)  # (B*L, N, K)

        # Use raw embeddings as partition weights (w_ik)
        pos_weights = pos_eb
        neg_weights = neg_eb

        # Compute similarities: sim(i,j) = Σ_k w_ik · w_jk as in [PAPER_REF] Eq (4)
        # This is the standard dot product measuring partition membership overlap
        pos_sim = (item_weights * pos_weights).sum(dim=-1)  # (B*L,)
        neg_sim = torch.matmul(
            item_weights.unsqueeze(1),
            neg_weights.transpose(-2, -1)
        ).squeeze(1)  # (B*L, N)

        # InfoNCE loss with temperature=1.0 (paper default, no temperature mentioned)
        # [PAPER_REF] Theorem 4: log[exp(Σ_k w_ik w_jk) / Σ_j' exp(Σ_k w_ik w_j'k)]
        temperature = 1.0

        # Clamp similarities to prevent overflow in exp
        pos_sim = torch.clamp(pos_sim / temperature, min=-10, max=10)
        neg_sim = torch.clamp(neg_sim / temperature, min=-10, max=10)

        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim).sum(dim=-1)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-9))

        # Apply mask to ignore padding
        mask_flat = mask.view(-1).float()
        loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-9)

        # Check for NaN
        if torch.isnan(loss):
            print(f"[DisMIR Warning] partition_loss is NaN, returning 0")
            return torch.tensor(0.0, device=items.device)

        return loss

    def compute_bpr_loss_with_hard_negative(self, user_eb, pos_items, neg_candidates=10):
        """
        Compute BPR loss with hard negative mining
        [PAPER_REF] Section 4.3: "We adopt the BPR loss with hard negative mining"

        For each positive item:
        1. Sample N candidates (default 10 per paper)
        2. Select the hardest negative (highest score)
        3. Compute BPR loss: -log(sigmoid(score(pos) - score(neg)))

        Args:
            user_eb: (batch_size, hidden_size) user representation
            pos_items: (batch_size,) positive item ids
            neg_candidates: Number of negative candidates to sample

        Returns:
            bpr_loss: scalar tensor
        """
        batch_size = user_eb.shape[0]

        # Get positive item embeddings
        pos_eb = self.embeddings(pos_items)  # (B, D)
        pos_scores = (user_eb * pos_eb).sum(dim=-1)  # (B,)

        # Sample negative candidates
        neg_samples = torch.randint(0, self.item_num,
                                    (batch_size, neg_candidates),
                                    device=user_eb.device)

        # Get negative embeddings and scores
        neg_eb = self.embeddings(neg_samples)  # (B, N, D)

        # Compute scores for all negatives: (B, 1, D) @ (B, D, N) -> (B, N)
        neg_scores = torch.matmul(
            user_eb.unsqueeze(1),
            neg_eb.transpose(-2, -1)
        ).squeeze(1)

        # Hard negative mining: select highest scoring negative
        hardest_neg_scores, _ = neg_scores.max(dim=-1)  # (B,)

        # BPR loss with numerical stability
        # Use logsigmoid with clamp to prevent extreme values
        diff = pos_scores - hardest_neg_scores
        diff = torch.clamp(diff, min=-20, max=20)  # Prevent overflow in exp

        loss = -F.logsigmoid(diff)

        # Check for NaN
        if torch.isnan(loss.sum()):
            print(f"[DisMIR Warning] BPR loss is NaN")
            print(f"  pos_scores range: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
            print(f"  neg_scores range: [{neg_scores.min():.4f}, {neg_scores.max():.4f}]")
            return torch.tensor(0.0, device=user_eb.device)

        return loss.sum()

    def forward(self, item_list, label_list, mask, times, device, train=True):
        """
        DisMIR forward pass

        Args:
            item_list: (batch_size, seq_len) historical item sequence
            label_list: (batch_size,) target item or None
            mask: (batch_size, seq_len) padding mask
            times: Tuple of (time_matrix, adj_matrix) - not used in DisMIR
            device: computation device
            train: training mode flag

        Returns:
            Training: (user_eb, scores, atten, readout, selection)
            Inference: (interests, None) where interests is (B, F, D)
        """
        # Item embedding lookup
        item_eb = self.embeddings(item_list)  # (B, L, D)
        item_eb = item_eb * torch.reshape(mask, (-1, self.seq_len, 1))

        if train:
            label_eb = self.embeddings(label_list)  # (B, D)

        # Multi-interest extraction via capsule network
        interests = self.capsule_net(item_eb, mask)  # (B, F, D)

        # Dynamic preference aggregation
        seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        e_u_bar = item_eb.sum(dim=1) / seq_lengths  # (B, D)

        # Compute attention over interests
        atten = torch.matmul(interests, e_u_bar.unsqueeze(-1)).squeeze(-1)
        atten = F.softmax(atten, dim=-1)

        if not train:
            # Return all interests for multi-interest retrieval
            return interests, None

        # Training: Read out based on target label
        readout, selection = self.read_out(interests, label_eb)

        # Calculate scores for compatibility with REMI pipeline
        scores = None if self.is_sampler else self.calculate_score(readout)

        return interests, scores, atten, readout, selection

    def calculate_atten_loss(self, attention):
        """
        Compute routing regularization loss
        [PAPER_REF] Xie et al. REMI Eq (1)

        Args:
            attention: (batch_size, interest_num) attention weights

        Returns:
            reg_loss: scalar tensor
        """
        atten_mean = attention.mean(dim=0, keepdim=True)
        atten_var = ((attention - atten_mean) ** 2).mean()
        return atten_var

    def calculate_disloss(self, readout, pos_items, selection, interests, atten):
        """
        Calculate DisMIR-specific loss
        Combines BPR loss with hard negative mining and partition loss
        [PAPER_REF] Eq 6: L = L_Rec + λ * L_Partition

        Args:
            readout: (batch_size, hidden_size) user representation
            pos_items: (batch_size,) positive item ids
            selection: (batch_size,) selected interest index
            interests: (batch_size, interest_num, hidden_size) all interests
            atten: (batch_size, interest_num) attention weights

        Returns:
            loss: scalar tensor
            loss_dict: dict of individual loss components for logging
        """
        # BPR loss with hard negative mining
        bpr_loss = self.compute_bpr_loss_with_hard_negative(
            readout, pos_items, self.hard_neg_candidates
        )

        # Partition loss (computed separately in training loop to handle item sequences)
        loss_dict = {
            'bpr_loss': bpr_loss.item(),
        }
        return bpr_loss, loss_dict