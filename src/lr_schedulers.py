#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from torch.optim.lr_scheduler import LambdaLR
from allennlp.training.learning_rate_schedulers import LearningRateScheduler


class LRPolicy:
    def __init__(self, warmup_steps, total_steps):
        self.cur_step = 0
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step=None):
        if step is None:
            step = self.cur_step
            self.cur_step += 1
        if step < self.warmup_steps:
            return float(step) / max(1, self.warmup_steps)
        return max(
            0.0,
            (float(self.total_steps) - step)
            / max(1.0, (float(self.total_steps) - self.warmup_steps)),
        )


def get_linear_warmup_linear_decrease(
    optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1
):
    return LambdaLR(optimizer, LRPolicy(warmup_steps, total_steps), last_epoch)


@LearningRateScheduler.register("WarmupLinearLR")
class WarmupLinearScheduler(LearningRateScheduler):
    def __init__(
        self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1
    ):
        self.scheduler = get_linear_warmup_linear_decrease(
            optimizer, warmup_steps, total_steps, last_epoch
        )

    def step(self, metric, epoch):
        return

    def step_batch(self, batch_num_total):
        self.scheduler.step()
