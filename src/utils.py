#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from typing import List

import torch
from tqdm import tqdm
from allennlp.models import Model
from allennlp.data import Token, Instance


def get_token_length(token: Token):
    text = token.text
    if text == "#":
        return 1
    else:
        return len(text) - text.count("#")


def generate_spans(
    tokens: List[Token],
    tags: List[int],
    sentence_offset: int = 0,
    article_id: str = "0",
    return_spans: bool = False,
):
    assert len(tokens) == len(tags)
    n = len(tokens)
    i = 0
    start = None
    end = None
    spans = []
    while i < n:
        if tags[i]:
            if start is None:
                start = tokens[i].idx
                end = start + get_token_length(tokens[i])
            else:
                end = tokens[i].idx + get_token_length(tokens[i])
            i += 1
        elif start is not None:
            spans.append((start + sentence_offset, end + sentence_offset))
            start, end = None, None
            i += 1
        else:
            i += 1
    if start is not None:
        spans.append((start + sentence_offset, end + sentence_offset))
    if return_spans:
        return spans
    result = []
    for span in spans:
        result.append(f"{article_id}\t{span[0]}\t{span[1]}\n")
    return "".join(result)


def generate_one_span(
    tokens: List[Token], tags: List[int], sentence_offset: int, article_id: str
):
    assert len(tokens) == len(tags)
    start = None
    end = None
    for i, tag in enumerate(tags):
        if tag and start is None:
            start = tokens[i].idx
        elif start is not None:
            end = tokens[i].idx
    if start is not None and end is None:
        end = tokens[-1].idx + len(tokens[-1].text)
    return f"{article_id}\t{start + sentence_offset}\t{end + sentence_offset}\n"


def evaluate_si(
    model: Model, dataset: List[Instance], filename: str, singlespan: bool = False
):
    f = open(filename, "w", encoding="utf8")
    model.eval()
    # to flush metrics
    model.forward_on_instance(dataset[0])
    model.get_metrics(True)
    batch_size = 50
    with torch.no_grad():
        for i in tqdm(range(len(dataset) // batch_size + 1)):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            if batch == []:
                break
            predictions = model.forward_on_instances(batch)
            tags = [p["tags"] for p in predictions]
            for instance, tag in zip(batch, tags):
                if sum(tag) == 0:
                    continue
                article_id, sentence_offset = (
                    instance["metadata"]["id"],
                    int(instance["metadata"]["sentence_pos"]),
                )
                tokens = instance["sentence"].tokens[1:-1]
                tag = tag[1:-1]
                if not singlespan:
                    f.write(generate_spans(tokens, tag, sentence_offset, article_id))
                else:
                    f.write(generate_one_span(tokens, tag, sentence_offset, article_id))
    f.close()
    return model.get_metrics(True)


def evaluate_ti(model: Model, dataset: List[Instance], filename: str):
    f = open(filename, "w", encoding="utf8")
    model.eval()
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(len(dataset) // batch_size + 1)):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            if batch == []:
                break
            predictions = model.forward_on_instances(batch)
            for p in predictions:
                f.write(p["final_output"])
    f.close()
    return model.get_metrics(True)
