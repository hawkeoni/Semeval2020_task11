#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
from allennlp.common import Registrable


class SpanClassifier(nn.Module, Registrable):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout

    def get_output_dim(self):
        return self.hidden_dim * 2

    def get_input_dim(self):
        return self.hidden_dim

    def forward(self, encoded: torch.Tensor, **kwargs):
        raise NotImplemented(
            "Not implemented forward for SpanClassifier abstract class."
        )


@SpanClassifier.register("RBert")
class RBertEncoder(SpanClassifier):
    def __init__(
        self, hidden_dim: int, dropout: float = 0.1, activation_first: bool = True
    ):
        super().__init__(hidden_dim, dropout)
        if activation_first:
            self.h0 = nn.Sequential(
                nn.Dropout(dropout), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.h1 = nn.Sequential(
                nn.Dropout(dropout), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.h0 = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
            )
            self.h1 = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
            )

    def forward(self, encoded: torch.Tensor, pos: List[Tuple[int, int]]):
        """
        :param encoded: - [batch_size, seq_len, hidden_dim]
        :param pos: - [batch_size, 2]
        :return: - [batch_size, hidden_dim]
        """
        h0 = encoded[:, 0]  # batch_size, hidden_dim
        h1 = []
        for i, (start, end) in enumerate(pos):
            h1.append(torch.mean(encoded[i, start:end], dim=0))
        h1 = torch.stack(h1, dim=0)
        h0 = self.h0(h0)
        h1 = self.h1(h1)
        h = torch.cat((h0, h1), dim=1)  # batch_size, hidden_dim * 2
        return h


@SpanClassifier.register("RbertAttention")
class RBertEncoderAttention(RBertEncoder):
    def __init__(
        self, hidden_dim: int, dropout: float = 0.1, activation_first: bool = True
    ):
        super().__init__(hidden_dim, dropout, activation_first)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, encoded: torch.Tensor, pos: List[Tuple[int, int]]):
        """
        :param encoded: - [batch_size, seq_len, hidden_dim]
        :param pos: - [batch_size, 2]
        :return: - [batch_size, hidden_dim]
        """
        h0 = encoded[:, 0]  # batch_size, hidden_dim
        h1 = []
        for i, (start, end) in enumerate(pos):
            span = encoded[i, start:end]  # span_len, hidden_dim
            span_weights = torch.softmax(self.attention(span), dim=0)  # span_len, 1
            span = span * span_weights  # span_len, hidden_dim
            span = torch.sum(span, dim=0)  # hidden_dim
            h1.append(span)
        h1 = torch.stack(h1, dim=0)
        h0 = self.h0(h0)
        h1 = self.h1(h1)
        h = torch.cat((h0, h1), dim=1)  # batch_size, hidden_dim * 2
        return h


@SpanClassifier.register("Kitaev")
class KitaevEncoder(SpanClassifier):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def get_output_dim(self):
        return self.hidden_dim

    def forward(self, encoded: torch.Tensor, pos: List[Tuple[int, int]]):
        """
        :param encoded: - [batch_size, seq_len, hidden_dim]
        :param pos: - [batch_size, 2]
        :return: - [batch_size, hidden_dim]
        """
        encoded = self.dropout(encoded)
        h = []
        for k, (i, j) in enumerate(pos):
            yi = encoded[k, i]  # hidden_dim
            yj = encoded[k, j - 1]  # hidden_dim
            span_features = torch.cat((yj[0::2] - yi[0::2], yi[1::2] - yj[1::2]), dim=0)
            h.append(span_features)
        h = torch.stack(h, dim=0)
        return h


@SpanClassifier.register("CLS")
class CLSEncoder(SpanClassifier):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoded: torch.Tensor, *args, **kwargs):
        return self.feedforward(encoded[:, 0])

    def get_output_dim(self):
        return self.hidden_dim
