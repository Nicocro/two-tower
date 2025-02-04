import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from src.cbow import CBOW, CBOWDataset
from src.tokenizer import W2VTokenizer

class QueryTower(nn.Module):
    def __init__(self, embedder: CBOW):
        super().__init__()
        # Get and freeze embeddings from CBOW
        self.embedding = embedder.embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Simple single-layer LSTM
        self.rnn = nn.LSTM(
            input_size=64,  # CBOW embedding dimension
            hidden_size=64,
            batch_first=True
        )
        
        # Final projection layer
        self.projection = nn.Linear(64, 64)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape [batch_size, max_seq_len]
            lengths: Tensor of shape [batch_size] with actual sequence lengths
        Returns:
            Tensor of shape [batch_size, 64]
        """
        embedded = self.embedding(token_ids)
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        _, (hidden, _) = self.rnn(packed)
        return self.projection(hidden[-1])

class PassageTower(nn.Module):
    def __init__(self, embedder: CBOW):
        super().__init__()
        # Get and freeze embeddings from CBOW
        self.embedding = embedder.embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Simple single-layer LSTM
        self.rnn = nn.LSTM(
            input_size=64,  # CBOW embedding dimension
            hidden_size=64,
            batch_first=True
        )
        
        # Final projection layer
        self.projection = nn.Linear(64, 64)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape [batch_size, max_seq_len]
            lengths: Tensor of shape [batch_size] with actual sequence lengths
        Returns:
            Tensor of shape [batch_size, 64]
        """
        embedded = self.embedding(token_ids)
        
        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        _, (hidden, _) = self.rnn(packed)
        return self.projection(hidden[-1])

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, query_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vec: torch.Tensor) -> torch.Tensor:
        # Compute similarities
        pos_sim = F.cosine_similarity(query_vec, pos_vec)
        neg_sim = F.cosine_similarity(query_vec, neg_vec)
        
        # Compute triplet loss with margin
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return loss.mean()