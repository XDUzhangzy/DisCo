"""
Core DisCo-iFormer model definition.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from dataset import standardize_eeg


class SeriesDecomposition(nn.Module):
    def __init__(self, thresholds, n_channels=None, channel_wise=False):
        super().__init__()
        self.thresholds = thresholds
        self.num_thresholds = len(thresholds)
        self.sampling_rate = 250
        self.n_channels = n_channels
        self.channel_wise = channel_wise
        if channel_wise:
            if n_channels is None:
                raise ValueError("n_channels must be specified for channel-wise decomposition.")
            self.weights = nn.Parameter(torch.randn(self.n_channels, self.num_thresholds))
        else:
            self.weights = nn.Parameter(torch.randn(1, self.num_thresholds))

    def forward(self, data):
        batch_size, channels, timepoints = data.shape
        fft_data = torch.fft.rfft(data, dim=-1)
        freqs = torch.fft.rfftfreq(timepoints, 1 / self.sampling_rate).to(data.device)
        trends = []

        for threshold in self.thresholds:
            low_freq_fft = fft_data.clone()
            low_freq_fft[..., freqs > threshold] = 0
            trends.append(torch.fft.irfft(low_freq_fft, n=timepoints, dim=-1))

        trends = torch.stack(trends, dim=-1)
        weights = torch.softmax(self.weights, dim=-1)

        if self.channel_wise:
            weights = weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, timepoints, -1)
        else:
            weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(batch_size, channels, timepoints, -1)

        trend = torch.sum(trends * weights, dim=-1)
        periodic = data - trend
        return trend, periodic


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_length=128):
        super().__init__()
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()

        embedding = torch.zeros(max_length, embedding_dim).float()
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("embedding", embedding.unsqueeze(0))

    def forward(self, x):
        return self.embedding[:, : x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, embedding_dim=64, num_tokens=32, dropout=0.5, use_cls_token=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.use_cls_token = use_cls_token
        self.value_embedding = nn.Sequential(
            nn.Linear(250, embedding_dim, bias=False),
            nn.BatchNorm1d(num_tokens),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.position_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, embedding_dim))
        else:
            self.position_embedding = nn.Parameter(torch.randn(1, num_tokens, embedding_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.value_embedding(x)
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_dim, embedding_dim)
        self.queries = nn.Linear(embedding_dim, embedding_dim)
        self.values = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill_(~mask, fill_value)

        scaling = (self.embedding_dim / self.num_heads) ** 0.5
        attention = F.softmax(energy / scaling, dim=-1)
        self.last_attention = attention.detach()
        attention = self.attention_dropout(attention)
        output = torch.einsum("bhal, bhlv -> bhav", attention, values)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.output_projection(output)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        residual = x
        x = self.fn(x, **kwargs)
        return x + residual


class FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_dim, expansion=4, dropout=0.5):
        super().__init__(
            nn.Linear(embedding_dim, expansion * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embedding_dim, embedding_dim),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_dim, num_heads=1, attention_dropout=0.5, expansion=4, mlp_dropout=0.5):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    MultiHeadAttention(embedding_dim, num_heads, attention_dropout),
                    nn.Dropout(attention_dropout),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    FeedForwardBlock(embedding_dim, expansion=expansion, dropout=mlp_dropout),
                    nn.Dropout(mlp_dropout),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, embedding_dim):
        super().__init__(*[TransformerEncoderBlock(embedding_dim) for _ in range(depth)])


class RepresentationHead(nn.Module):
    def __init__(self, embedding_dim, inner_channels, use_cls_token=False):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.embedding_dim = embedding_dim
        self.inner_channels = inner_channels
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inner_channels * embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        channels = x.shape[1]
        if self.use_cls_token:
            x = x[:, :1, :]
        else:
            x = x[:, -channels:, :]
        return self.projection(x)


class ClassifierHead(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class DisCoIFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.embedding_dim = config.emb_dim
        self.depth = config.depth
        self.num_classes = config.n_class
        self.n_channels = config.n_channels
        self.inner_channels = config.inner_channels
        self.dropout = config.dropout

        self.series_decomposition = SeriesDecomposition(
            config.thresholds,
            n_channels=self.inner_channels,
            channel_wise=True,
        )
        if self.n_channels != self.inner_channels:
            self.channel_mapper = nn.Conv1d(self.n_channels, self.inner_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.channel_mapper = nn.Identity()

        self.data_embedding = DataEmbedding(
            embedding_dim=self.embedding_dim,
            num_tokens=self.inner_channels,
            dropout=self.dropout,
            use_cls_token=self.use_cls_token,
        )
        self.transformer_encoder = TransformerEncoder(self.depth, self.embedding_dim)
        self.representation_head = RepresentationHead(
            self.embedding_dim,
            self.inner_channels,
            use_cls_token=self.use_cls_token,
        )
        self.classifier_head = ClassifierHead(self.num_classes)
        self.adapter = None

    def forward(self, x):
        x = x.squeeze(1)
        x = self.channel_mapper(x)
        trend, _periodic = self.series_decomposition(x)
        x = standardize_eeg(trend, channels=True)
        x = self.data_embedding(x)
        x = self.transformer_encoder(x)
        features = self.representation_head(x)
        if self.adapter is not None:
            features = features + self.adapter(x)
        logits = self.classifier_head(features)
        return logits, features
