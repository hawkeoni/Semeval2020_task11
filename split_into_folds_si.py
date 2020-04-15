#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import sys
import random
from collections import defaultdict


def main(train_file: str, k: int):
    samples = defaultdict(list)
    cur_article = None
    for line in open(train_file, encoding="utf8"):
        if line.startswith("-" * 25):
            cur_article = line.strip().strip("-")
        else:
            if cur_article is None:
                print(line)
                raise BaseException("Current article is none!")
            samples[cur_article].append(line)
    article_ids = list(samples.keys())
    random.shuffle(article_ids)
    n = len(article_ids)
    fold_size = n // k
    for i in range(k):
        test_articles = article_ids[i * fold_size : (i + 1) * fold_size]
        train_articles = (
            article_ids[: i * fold_size] + article_ids[(i + 1) * fold_size :]
        )
        assert len(train_articles) + len(test_articles) == len(article_ids)
        train_fold_file = open(f"train_fold_{i}.txt", "w", encoding="utf8")
        for article_id in train_articles:
            train_fold_file.write("-" * 25 + article_id + "-" * 25 + "\n")
            for sample in samples[article_id]:
                train_fold_file.write(sample)
        train_fold_file.close()
        test_fold_file = open(f"dev_fold_{i}.txt", "w", encoding="utf8")
        for article_id in test_articles:
            test_fold_file.write("-" * 25 + article_id + "-" * 25 + "\n")
            for sample in samples[article_id]:
                test_fold_file.write(sample)
        test_fold_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Required argv: train_filename, number of folds.")
    train_file = sys.argv[1]
    k = int(sys.argv[2])
    main(train_file, k)
