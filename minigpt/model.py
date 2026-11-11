from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_embd, self.n_layer, self.n_head) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop, self.resid_drop = nn.Dropout(cfg.dropout), nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask[None, None], persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.heads, self.dim)
        q, k, v = [z.view(shape).transpose(1, 2) for z in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) * (self.dim ** -0.5)
        scores = scores.masked_fill(~self.mask[:, :, :t, :t], torch.finfo(scores.dtype).min)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4*cfg.n_embd), nn.GELU(), nn.Linear(4*cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.n_embd), FeedForward(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2: raise ValueError("idx must have shape (batch, sequence)")
        b, t = idx.shape
        if t > self.config.block_size: raise ValueError("Input exceeds configured context length")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        logits = self.lm_head(self.final_ln(self.blocks(x)))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        if temperature <= 0 or max_new_tokens < 0: raise ValueError("Invalid generation settings")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size:])
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        finally: self.train(was_training)
        return idx
