#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from allennlp.training.metrics import Metric


class MultilabelMicroF1(Metric):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __call__(self, pred_labels, true_labels):
        self.tp += ((pred_labels == 1) * (true_labels == 1)).sum().item()
        self.fp += ((pred_labels == 1) * (true_labels == 0)).sum().item()
        self.fn += ((pred_labels == 0) * (true_labels == 1)).sum().item()

    def get_metric(self, reset: bool = False):
        if self.tp + self.fp > 0:
            prec = self.tp / (self.tp + self.fp)
        else:
            prec = 0
        if self.tp + self.fn > 0:
            rec = self.tp / (self.tp + self.fn)
        else:
            rec = 0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0
        if reset:
            self.reset()
        return {"prec": prec, "rec": rec, "f1": f1}


class Accuracy(Metric):
    def __init__(self):
        self.correct_pos = 0
        self.correct = 0
        self.total = 0
        self.total_pos = 0

    def reset(self):
        self.correct_pos = 0
        self.correct = 0
        self.total = 0
        self.total_pos = 0

    def __call__(self, pred_labels, true_labels):
        self.correct_pos += ((pred_labels > 0) * (true_labels > 0)).sum().item()
        self.correct += (pred_labels == true_labels).sum().item()
        self.total += pred_labels.numel()
        self.total_pos += true_labels.sum().item()

    def get_metric(self, reset: bool):
        if self.total > 0:
            acc = self.correct / self.total
        else:
            acc = 0.0
        if self.total_pos > 0:
            pos_acc = self.correct_pos / self.total_pos
        else:
            pos_acc = 0
        if reset:
            self.reset()
        return {"acc": acc, "pos_acc": pos_acc}
