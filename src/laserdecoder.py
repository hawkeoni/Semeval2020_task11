#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import math

import torch
import torch.nn as nn


def gen_wavelength_embedding(d_model: int, max_length: int) -> torch.Tensor:
    """
    Taken from https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """
    pos_embedding = torch.zeros(max_length, d_model)
    position = torch.arange(0, max_length).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, nheads: int, bias: bool = False, dropout: float = 0.2
    ):
        super().__init__()
        assert d_model % nheads == 0, "Number of heads should divide d_model"
        self.d_model = d_model
        self.nheads = nheads
        self.bias = bias
        self.d_k = d_model // nheads
        self.Q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.K_linear = nn.Linear(d_model, d_model, bias=bias)
        self.V_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Q - float tensors of shape [batch, seq_len1, d_model].
        K, V - float tensors of shape [batch, seq_len2, d_model].
        mask should be 0, where padding is.
        """
        batch_size = query.size(0)
        q_proj = (
            self.Q_linear(query)
            .view(batch_size, -1, self.nheads, self.d_k)
            .permute(0, 2, 1, 3)
        )
        k_proj = (
            self.K_linear(key)
            .view(batch_size, -1, self.nheads, self.d_k)
            .permute(0, 2, 3, 1)
        )
        v_proj = (
            self.V_linear(value)
            .view(batch_size, -1, self.nheads, self.d_k)
            .permute(0, 2, 1, 3)
        )
        weights = torch.matmul(q_proj, k_proj)  # batch, nheads, seq_len1, seq_len2
        weights = weights / (self.d_k ** 0.5)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e12)
        weights = torch.softmax(weights, dim=3)
        #  weights - batch, nheads, seq_len1, seq_len2
        #  V_proj - batch, nhead, seq_len2, d_k
        output = torch.matmul(weights, v_proj)  # batch, nheads, seq_len1, d_k
        output = output.transpose(1, 2)
        output = output.contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.out(output)
        output = self.dropout(output)
        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x - float tensor of shape [batch, seq_len, d_model].
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class LaserDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nheads: int,
        ff_dim: int,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Decoder with no encoder-decoder attention.
        """
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.ff_dim = ff_dim
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, nheads, bias, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, ff_dim)

    def forward(self, trg: torch.Tensor, trg_mask: torch.Tensor = None):
        """
        Masks should be 0, where padding is.
        """
        trg = trg + self.self_attention(trg, trg, trg, trg_mask)
        trg = self.norm1(trg)
        trg = trg + self.feedforward(trg)
        trg = self.norm2(trg)
        return trg


class LaserDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_layers: int,
        ff_dim: int,
        num_heads: int,
        num_classes: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_classes
        self.num_classes = num_classes
        self.projection = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.register_buffer("pos_embedding", gen_wavelength_embedding(hidden_dim, 512))
        self.layers = nn.ModuleList(
            [
                LaserDecoderLayer(hidden_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor, trg_mask: torch.Tensor = None
    ):
        """
        src - tensor of shape [batch, src_len, d_model].
        trg - tensor of shape [batch, trg_len].
        masks - tensors of shape [batch, src/trg_len].
        """
        if trg_mask is None:
            trg_mask = trg.new_ones(trg.size(0), 1, 1, trg.size(1))
            trg_mask = (
                trg_mask.float()
                * torch.tril(trg_mask.new_ones(trg.size(1), trg.size(1))).float()
            )
        y = (
            self.embedding(trg) * (self.hidden_dim ** 0.5)
            + self.pos_embedding[:, : trg.size(1)]
        )
        y = torch.cat((y, src), dim=2)
        y = self.projection(y)
        for layer in self.layers:
            y = layer(y, trg_mask)
        return self.out(y)
