#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import sys
import random
from collections import defaultdict


def main(train_file: str, k: int):
    samples = defaultdict(list)
    lines = []
    for line in open(train_file, encoding="utf8"):
        lines.append(line)
    n = len(lines)
    fold_size = n // k
    for i in range(k):
        test_lines = lines[i * fold_size : (i + 1) * fold_size]
        train_lines = lines[: i * fold_size] + lines[(i + 1) * fold_size :]
        assert len(train_lines) + len(test_lines) == len(lines)
        train_fold_file = open(f"train_fold_ti_{i}.txt", "w", encoding="utf8")
        for line in train_lines:
            train_fold_file.write(line)
        train_fold_file.close()
        test_fold_file = open(f"dev_fold_ti_{i}.txt", "w", encoding="utf8")
        for line in test_lines:
            test_fold_file.write(line)
        test_fold_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Required argv: train_filename, number of folds.")
    train_file = sys.argv[1]
    k = int(sys.argv[2])
    main(train_file, k)
