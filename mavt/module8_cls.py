"""
Module 8: Classifier 

Two versions for classifier 
1. Simple one 
- Design: 2 Linear layers (one for learning from z_understand 
  which can be reused when training different data structures, one for output layer)
2. Fast former one 
- Design: Fast former for extracting z and one linear for z_understand,
  concate them and use a linear layer for output
"""

import torch 
import torch.nn as nn
from .types import LatentOutput

class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channel: int) -> None:
        super().__init__()      

        self.num_classes = num_classes
        self.in_channel = in_channel

        self.linear = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.BatchNorm1d(in_channel // 2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(in_channel // 2, num_classes)

    def forward(self, latent_output: LatentOutput):
        x = self.linear(latent_output.z_understand)
        return self.classifier(x)
    
class StableSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask=None):
        x_max = torch.max(x, dim=-1, keepdim=True).values
        x_exp = torch.exp(x - x_max)

        if attention_mask is not None: 
            x_exp = x_exp * attention_mask

        return x_exp / torch.sum(x_exp, dim=-1, keepdim=True) + 1e-9
    
class FastAttention(nn.Module):
    def __init__(
        self, latent_dim: int,
        num_heads: int=8
    ):
        super().__init__()

        assert latent_dim % num_heads == 0

        self.latent_dim = latent_dim 
        self.num_heads = num_heads
        self.scale = (self.latent_dim // self.num_heads) ** (-0.5)
        self.to_qkv = nn.Linear(latent_dim, 3 * latent_dim)
        self.q_linear_attention = nn.Linear(latent_dim // num_heads, 1)
        self.k_linear_attention = nn.Linear(latent_dim // num_heads, 1)
        self.out = nn.Linear(latent_dim, latent_dim)
        self.stable_softmax = StableSoftmax()

    def split_heads(self, x, num_heads):
        batch_size, sequence_length, dim = x.size()
        head_dim = dim // num_heads
        return x.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
    
    def merge_heads(self, x):
        batch_size, num_heads, sequence_length, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, sequence_length, num_heads * head_dim)

    def forward(self, x, attention_mask=None):
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.split_heads(q, self.num_heads) # (B, H, N, Hd)
        k = self.split_heads(k, self.num_heads) # (B, H, N, Hd)
        v = self.split_heads(v, self.num_heads) # (B, H, N, Hd)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) # (B, 1, N)

        q_logits = self.q_linear_attention(q) * self.scale # (B, H, N, 1)
        alpha = self.stable_softmax(q_logits.squeeze(-1), attention_mask) # (B, H, N)
        global_q = torch.einsum("bhn,bhnd->bhd", alpha, q) # (B, H, Hd)
        global_q = global_q.unsqueeze(-2)
        k = k * global_q # (B, H, N, Hd)
        k_logits = self.k_linear_attention(k) * self.scale 
        beta = self.stable_softmax(k_logits.squeeze(-1), attention_mask) # (B, H, N)
        global_k = torch.einsum("bhn,bhnd->bhd", beta, k) # (B, H, Hd)

        out = global_k.unsqueeze(-2) * v 
        out = self.merge_heads(out) # (B, N, latent_dim)
        out = self.out(out)

        return out 
    
class FastFormerBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = FastAttention(latent_dim=latent_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
    
class FastFormerClassifier(nn.Module):
    def __init__(
        self, num_classes: int, in_channel: int, out_channel: int,
        latent_dim: int, num_blocks: int, num_heads: int
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channel = in_channel
        self.latent_dim = latent_dim

        self.fastformer_blocks = nn.ModuleList([
            FastFormerBlock(latent_dim=latent_dim, num_heads=num_heads)
            for i in range(num_heads)
        ])
        self.linear = nn.Linear(in_channel, out_channel)
        self.classifier = nn.Linear(out_channel + latent_dim, num_classes)

    def forward(self, latent_output: LatentOutput, attention_mask=None):
        z = latent_output.z
        for block in self.fastformer_blocks:
            z = block(z) # (B, N, latent_dim)
        z = z.mean(dim=1) # (B, latent_dim)
        z_understand = self.linear(latent_output.z_understand) # (B, out_channel)
        x = torch.concat([z, z_understand], dim=-1) 
        return self.classifier(x)







        




