import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attn(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, need_weights=False):
        x, weights = self.attn.forward(x, x, x, need_weights=need_weights)

        return x, weights


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio=4, attn_dropout=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = Attn(emb_dim, num_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
        )

        self.c_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 6),
        )

    def forward(self, x, t, y, need_weights=False):
        c = (t + y).unsqueeze(1)
        c1, c2, c3, c4, c5, c6 = self.c_proj(c).chunk(6, dim=-1)

        if need_weights == False:
            x = x + c3 * self.attn(c1 * self.ln1(x) + c2, need_weights=False)[0]
            x = x + c6 * self.mlp(c4 * self.ln2(x) + c5)
            return x, None

        if need_weights == True:
            attn, weights = self.attn(c1 * self.ln1(x) + c2, need_weights=True)
            x = x + c3 * attn
            x = x + c6 * self.mlp(c4 * self.ln2(x) + c5)
            return x, weights


class DiT(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio, num_layers):
        super().__init__()
        self.conv_in = nn.Conv2d(4, emb_dim, 1)
        self.blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.conv_out = nn.Conv2d(emb_dim, 4, 1)

        self.y_emb = nn.Embedding(11, emb_dim)
        self.t_emb = nn.Linear(1, emb_dim)
        self.pos_emb = nn.Parameter(
            torch.randn((1, 16, emb_dim)), requires_grad=True
        )

    def forward(self, x, t=None, y=None, need_weights=False):
        B, C, H, W = x.shape
        if t is None:
            t = torch.zeros((B, 1), device=x.device, dtype=x.dtype)
        if y is None:
            y = torch.zeros((B), device=x.device, dtype=torch.long)

        t = t.view(B, 1)

        y = self.y_emb(y)
        t = self.t_emb(t)
        x = self.conv_in(x)
        x = rearrange(x, "b c h w -> b (h w) c") + self.pos_emb
        for block in self.blocks:
            x, _ = block(x, t, y, need_weights=need_weights)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = DiT(emb_dim=16, num_heads=16, mlp_ratio=4, num_layers=6)
    summary(model, input_size=(256, 4, 4, 4))
