# pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, embed_time, arg=None, dim=None, device="cuda"):
        super(TimeEmbedding, self).__init__()
        self.embed_time = embed_time
        self.device = device
        self.arg = arg
        if arg == "periodic" or arg == "linear":
            if dim is not None:
                self.w = nn.Linear(1, dim * embed_time)
            else:
                self.w = nn.Linear(1, embed_time)
        elif arg == "periodic_linear":
            self.w1 = nn.Linear(1, embed_time // 2)
            self.w2 = nn.Linear(1, embed_time // 2)
        elif arg == "identity":
            print("identity")
            pass
        else:
            self.w1 = nn.Linear(1, embed_time - 1)
            self.w2 = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        if self.arg == "linear":
            return self.w(tt)
        elif self.arg == "periodic":
            return torch.sin(self.w(tt))
        elif self.arg == "identity":
            return tt.repeat(1, 1, self.embed_time)
        else:
            out2 = torch.sin(self.w1(tt))
            out1 = self.w2(tt)
            return torch.cat([out1, out2], -1)


class UnTAN(nn.Module):
    def __init__(
        self,
        input_dim,
        nhidden,
        embed_time,
        num_heads,
        intensity=None,
        union_tp=None,
        dropout=0.0,
        no_mix=True,
    ):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.intensity = intensity
        input_size = 2 * input_dim if intensity and not no_mix else input_dim
        self.linears = nn.ModuleList([
            nn.Linear(embed_time, embed_time, bias=False),
            nn.Linear(embed_time, embed_time, bias=False),
            nn.Linear(input_size * num_heads, nhidden, bias=False)
        ])
        self.time_emb = TimeEmbedding(embed_time, arg='periodic')
        self.dropout = nn.Dropout(p=dropout)
        self.union_tp = union_tp
        self.no_mix = no_mix

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        normalizer = torch.logsumexp(scores, dim=-2)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        if self.intensity:
            assert self.union_tp is not None
            un_key = self.linears[1](self.time_emb(self.union_tp)).view(
                1, self.h, -1, self.embed_time_k)
            un_scores = torch.matmul(query, un_key.transpose(-2, -1)) / math.sqrt(d_k)
            # normalizer = torch.logsumexp(un_scores.unsqueeze(-1), dim=-2)
            # intensity = torch.exp(torch.logsumexp(scores, dim=-2) - normalizer)
            normalizer = torch.max(un_scores.unsqueeze(-1), dim=-2)[0]
            intensity = torch.exp(torch.max(scores, dim=-2)[0] - normalizer)
        else:
            intensity = torch.exp(torch.logsumexp(scores, dim=-2) - normalizer)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), intensity

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        "Compute Multi Time Attention"
        key = self.time_emb(key)
        query = self.time_emb(query)
        batch = value.size(0)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(
            x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))]
        x, intensity = self.attention(query, key, value, mask, self.dropout)
        if self.intensity:
            intensity = intensity.transpose(1, 2).contiguous() \
                .view(batch, -1, self.h * self.dim)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * self.dim)
        if self.intensity and self.no_mix:
            return torch.stack((x, intensity), -1)
        elif self.intensity:
            return self.linears[-1](torch.cat((x, intensity), -1))
        return self.linears[-1](x)
