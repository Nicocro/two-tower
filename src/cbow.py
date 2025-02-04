import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from typing import List, Tuple

from src.tokenizer import W2VTokenizer


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words_indices: torch.Tensor) -> torch.Tensor:
        word_embeddings = self.embeddings(context_words_indices) 
        context_embeds_mean = torch.mean(word_embeddings, dim=1)
        out = self.linear(context_embeds_mean) 
        return out
    
class CBOWDataset(Dataset):
    def __init__(self, tokens: List[int], tokenizer: W2VTokenizer, window_size: int, vocab_size: int, num_neg_samples: int = 5):
        self.tokens = tokens
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        self.tokenizer = tokenizer
        self.context_target_pairs = self._create_context_target_pairs()
        

    def _create_context_target_pairs(self) -> List[Tuple[List[int], int, List[int]]]:
        pairs = []
        padding_token = self.tokenizer.special_tokens.get("<PAD>", 0)

        for i in range(len(self.tokens)):
            left_context = self.tokens[max(0, i - self.window_size): i]
            right_context = self.tokens[i + 1: i + 1 + self.window_size]

            # Ensure fixed size by padding if necessary
            left_context = [padding_token] * (self.window_size - len(left_context)) + left_context
            right_context = right_context + [padding_token] * (self.window_size - len(right_context))

            context = left_context + right_context
            target = self.tokens[i]
            neg_samples = self._negative_sampling(target)
            
            pairs.append((context, target, neg_samples))

        return pairs
    
    def _negative_sampling(self, target: int)  -> List[int]:
      return [n for n in torch.randint(0, self.vocab_size, (self.num_neg_samples,)).tolist()]

    def __len__(self) -> int:
        return len(self.context_target_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target, neg_samples = self.context_target_pairs[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long)
        )
    
def cbow_loss(output: torch.Tensor, target: torch.Tensor, neg_samples: torch.Tensor) -> torch.Tensor:
    batch_size = output.shape[0]
    
    # Positive sample loss - gather target logits
    target_logits = output[torch.arange(batch_size), target]
    pos_loss = F.logsigmoid(target_logits)
    
    # Negative samples loss - reshape for batch processing
    neg_logits = output[torch.arange(batch_size).unsqueeze(1), neg_samples]
    neg_loss = F.logsigmoid(-neg_logits).sum(dim=1)

    return -(pos_loss + neg_loss).mean()