import torch
import torch.nn as nn
import torch.nn.functional as F




class MaskedSelfAttention(nn.Module):

    def __init__(self, 
                 emb_dim,
                 num_heads,
                 block_size,
                 attention_dropout,
                 residual_dropout):
        super(MaskedSelfAttention, self).__init__()
        assert emb_dim % num_heads == 0, 'emb_dim should be divisible by num_heads'
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout

        # keys, queries and values for all heads
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

        # regularization
        self.att_dropout = nn.Dropout(attention_dropout)  # attention dropout
        self.resid_dropout = nn.Dropout(residual_dropout)  # residual dropout

        # mask to prevent attending to future positions
        mask = torch.tril(torch.ones(block_size, block_size))

        # output projection
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.register_buffer('mask', mask.view(1, 1, block_size, block_size))



    def forward(self, x):
        B, T, C = x.shape   # batch size, sequence length, embedding dimensionality (emb_dim)

        # calculate query, key, values
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attention is masked to prevent attending to future positions
        # self-attend (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scale_factor = torch.sqrt(k.size(-1))   # scale to preserve variance to one
        attn = (q @ k.transpose(-2, -1)) / scale_factor 
        attn = attn.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # dropout
        attn = self.att_dropout(attn)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y
    


class Block(nn.Module):

    def __init__(self,
                 emb_dim,
                 num_heads,
                 block_size,
                 attention_dropout,
                 residual_dropout,):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = MaskedSelfAttention(emb_dim, num_heads, block_size, attention_dropout, residual_dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(residual_dropout)
        )

    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    


class GPT(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 num_heads,
                 num_layers,
                 block_size,
                 emb_dropout,
                 attention_dropout,
                 residual_dropout):
        
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.emb_dropout = emb_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, emb_dim))
        self.drop = nn.Dropout(emb_dropout)
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads, block_size, attention_dropout, residual_dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    
    def forward(self, idx, embeddings=None):
        token_emb = self.token_emb(idx)
        if embeddings is not None:
            token_emb = torch.cat([token_emb, embeddings], dim=1)

        t = token_emb.shape[1]
        assert t <= self.block_size, 'Cannot forward, model block size is exhausted.'
        position_emb = self.pos_emb[:, :t, :]
        x = self.drop(token_emb + position_emb)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

        