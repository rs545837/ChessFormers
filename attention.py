import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        
        # key, query, value projections for all heads
        # dimensions: [n_embed, n_head, n_embed // n_head]
        self.key = nn.Parameter(torch.zeros(config.n_embed, config.n_head, config.n_embed // config.n_head))
        self.query = nn.Parameter(torch.zeros(config.n_embed, config.n_head, config.n_embed // config.n_head))
        self.value = nn.Parameter(torch.zeros(config.n_embed, config.n_head, config.n_embed // config.n_head))
        
        # regularization: dropout layers for attention and residual connections
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection layer
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # calculate query, key, values for all heads in batch
        # dimensions: [batch_size, n_head, seq_len, n_embed // n_head]
        k = self.key(x).view(B, self.n_head, T, C // self.n_head)
        q = self.query(x).view(B, self.n_head, T, C // self.n_head)
        v = self.value(x).view(B, self.n_head, T, C // self.n_head)
        
        # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # scale attention scores by square root of head size
        att = torch.bmm(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask out future positions to preserve causal property
        # dimensions: [1, 1, seq_len, seq_len]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e10)
        # softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        # apply dropout to attention probabilities
        att = self.attn_drop(att)
        
        # compute attended values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = torch.bmm(att, v)
        # reshape attended values back to original shape: (B, T, C)
        y = y.view(B, T, C)
        # apply final output projection and residual dropout
        y = self.resid_drop(self.proj(y))
        
        return y