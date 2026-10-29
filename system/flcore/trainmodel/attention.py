import torch
import torch. nn as nn
import torch.nn.functional as F
import math


class QueryEncoder(nn.Module):
    """
    Encodes batch statistics into a query vector. 
    Input: [mean_features, std_features, entropy] -> query embedding
    """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn. Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x:  Batch statistics tensor [batch_size, input_dim] or [input_dim]
        Returns:
            Query embedding [batch_size, output_dim] or [output_dim]
        """
        return self.encoder(x)


class KeyEncoder(nn.Module):
    """
    Encodes peer statistics into a key vector.
    Shared across all peers for efficiency.
    """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args: 
            x:  Peer statistics tensor [num_peers, input_dim] or [input_dim]
        Returns:
            Key embedding [num_peers, output_dim] or [output_dim]
        """
        return self.encoder(x)


class AttentionNet(nn.Module):
    """
    Attention network for selective peer knowledge pooling.
    Computes attention weights over peers based on query-key similarity.
    
    Architecture:
        - Query encoder: Maps batch stats -> query vector
        - Key encoder: Maps peer stats -> key vectors (shared)
        - Scaled dot-product attention for weight computation
    
    Total parameters: ~10K
    """
    def __init__(self, stats_dim=128, hidden_dim=64, attention_dim=64, dropout=0.1, temperature=1.0):
        """
        Args: 
            stats_dim:  Dimension of input statistics (mean + std + entropy + class_dist)
            hidden_dim: Hidden layer dimension
            attention_dim: Query/Key embedding dimension (d_k)
            dropout:  Dropout rate
            temperature: Softmax temperature for attention (higher = softer)
        """
        super().__init__()
        
        self.attention_dim = attention_dim
        self. temperature = temperature
        
        # Query encoder for client's batch statistics
        self. query_encoder = QueryEncoder(
            input_dim=stats_dim,
            hidden_dim=hidden_dim,
            output_dim=attention_dim,
            dropout=dropout
        )
        
        # Key encoder for peer statistics (shared across peers)
        self.key_encoder = KeyEncoder(
            input_dim=stats_dim,
            hidden_dim=hidden_dim,
            output_dim=attention_dim,
            dropout=dropout
        )
        
        # Scaling factor for dot-product attention
        self.scale = math.sqrt(attention_dim)
    
    def forward(self, query_stats, peer_stats_list, peer_ids=None, self_id=None):
        """
        Compute attention weights over peers.
        
        Args:
            query_stats: Current client's batch statistics [stats_dim]
            peer_stats_list: List of peer statistics tensors, each [stats_dim]
                            OR stacked tensor [num_peers, stats_dim]
            peer_ids:  Optional list of peer IDs (for excluding self)
            self_id: Optional self ID (to exclude from attention)
        
        Returns:
            attention_weights:  Normalized attention weights [num_peers]
            peer_indices: Indices of peers (after filtering self)
        """
        # Encode query
        if query_stats.dim() == 1:
            query_stats = query_stats. unsqueeze(0)
        query = self.query_encoder(query_stats)  # [1, attention_dim]
        
        # Stack peer stats if list
        if isinstance(peer_stats_list, list):
            if len(peer_stats_list) == 0:
                return torch.tensor([]), []
            peer_stats = torch.stack(peer_stats_list, dim=0)  # [num_peers, stats_dim]
        else: 
            peer_stats = peer_stats_list
        
        # Filter out self if specified
        valid_indices = list(range(len(peer_stats)))
        if peer_ids is not None and self_id is not None:
            valid_indices = [i for i, pid in enumerate(peer_ids) if pid != self_id]
            if len(valid_indices) == 0:
                return torch. tensor([]), []
            peer_stats = peer_stats[valid_indices]
        
        # Encode keys
        keys = self.key_encoder(peer_stats)  # [num_peers, attention_dim]
        
        # Scaled dot-product attention
        # query:  [1, attention_dim], keys: [num_peers, attention_dim]
        scores = torch.matmul(query, keys.T) / self.scale  # [1, num_peers]
        scores = scores / self.temperature
        
        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1).squeeze(0)  # [num_peers]
        
        return attention_weights, valid_indices
    
    def compute_attention_with_scores(self, query_stats, peer_stats_list):
        """
        Compute attention with raw scores (useful for analysis).
        
        Returns:
            attention_weights:  Normalized weights
            raw_scores:  Unnormalized scores
        """
        if query_stats.dim() == 1:
            query_stats = query_stats.unsqueeze(0)
        query = self.query_encoder(query_stats)
        
        if isinstance(peer_stats_list, list):
            peer_stats = torch.stack(peer_stats_list, dim=0)
        else:
            peer_stats = peer_stats_list
        
        keys = self.key_encoder(peer_stats)
        
        raw_scores = torch. matmul(query, keys.T) / self.scale
        attention_weights = F.softmax(raw_scores / self.temperature, dim=-1).squeeze(0)
        
        return attention_weights, raw_scores. squeeze(0)


class StatisticsExtractor: 
    """
    Utility class to extract statistics from embeddings and logits.
    Used to create query/key inputs for attention network. 
    """
    def __init__(self, num_classes, feature_dim, device='cuda'):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
    
    def extract_batch_stats(self, embeddings, logits, temperature=1.0):
        """
        Extract statistics from a batch of embeddings and logits.
        
        Args: 
            embeddings: Feature embeddings [batch_size, feature_dim]
            logits: Model outputs [batch_size, num_classes]
            temperature: Temperature for entropy computation
        
        Returns:
            stats: Statistics tensor [stats_dim]
                   Contains:  [mean_emb (feature_dim), std_emb (feature_dim), 
                             entropy (1), class_dist (num_classes)]
        """
        # Embedding statistics
        mean_emb = embeddings.mean(dim=0)  # [feature_dim]
        std_emb = embeddings.std(dim=0)    # [feature_dim]
        
        # Prediction entropy
        probs = F.softmax(logits / temperature, dim=-1)  # [batch_size, num_classes]
        entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [batch_size]
        mean_entropy = entropy_per_sample. mean().unsqueeze(0)  # [1]
        
        # Class distribution (histogram of predictions)
        pred_classes = logits.argmax(dim=-1)  # [batch_size]
        class_dist = torch.zeros(self.num_classes, device=embeddings.device)
        for c in range(self. num_classes):
            class_dist[c] = (pred_classes == c).float().sum() / len(pred_classes)
        
        # Concatenate all statistics
        # Reduce embedding stats to fixed size for efficiency
        if self.feature_dim > 64:
            # Use adaptive pooling to reduce dimension
            mean_emb_reduced = F.adaptive_avg_pool1d(
                mean_emb. unsqueeze(0).unsqueeze(0), 32
            ).squeeze()
            std_emb_reduced = F. adaptive_avg_pool1d(
                std_emb. unsqueeze(0).unsqueeze(0), 32
            ).squeeze()
        else:
            mean_emb_reduced = mean_emb[: 32] if len(mean_emb) >= 32 else F.pad(mean_emb, (0, 32 - len(mean_emb)))
            std_emb_reduced = std_emb[:32] if len(std_emb) >= 32 else F.pad(std_emb, (0, 32 - len(std_emb)))
        
        stats = torch.cat([
            mean_emb_reduced,      # 32 dims
            std_emb_reduced,       # 32 dims  
            mean_entropy,          # 1 dim
            class_dist             # num_classes dims
        ], dim=0)
        
        return stats
    
    def get_stats_dim(self):
        """Return the dimension of statistics vector."""
        # 32 (mean) + 32 (std) + 1 (entropy) + num_classes
        return 32 + 32 + 1 + self.num_classes