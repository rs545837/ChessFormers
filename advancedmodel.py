import torch
import torch.nn as nn
import attention

class GPTJConfig(GPTConfig):
    n_layer = 28
    n_head = 16
    n_embed = 4096
    rotary_dim = 64

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = torch.einsum("i,j->ij", seq.float(), self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

class GPTJ(nn.Module):
    def __init__(self, config):
        super().__init__()
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = RotaryEmbedding(config.rotary_dim)
        self.drop = nn.Dropout(config.embed_pdrop)

        # transformer network
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.model_config = config

        print('Number of parameters: {}'.format(sum(p.numel() for p in self.parameters())))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb(t)

        x = self.drop(token_embeddings)
        x = x + position_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets.view(-1),
                ignore_index=0
            )

        return logits, loss
