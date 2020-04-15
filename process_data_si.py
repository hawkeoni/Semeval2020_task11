#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import os
import re
import argparse
from typing import List, Tuple


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping spans for SI task."""
    spans = sorted(spans)
    stack = [spans[0]]
    i = 1
    while i < len(spans):
        if stack[-1][1] >= spans[i][0]:
            top = stack.pop()
            stack.append((top[0], spans[i][1]))
        else:
            stack.append(spans[i])
        i += 1
    return stack


def get_spans_si(spanfile: str) -> List[Tuple[int, int]]:
    """Read file with spans for SI task."""
    spans = []
    with open(spanfile, encoding="utf8") as f:
        for line in f:
            article_id, start, end = map(int, line.split("\t"))
            spans.append((start, end))
    i = 0
    if spans == []:
        return []
    spans = merge_spans(spans)
    while i < len(spans) - 1:
        assert (
            spans[i][1] < spans[i + 1][0]
        ), f"{i}-th span, {spans[i]}, {spans[i + 1]} inf {spanfile}"
        i += 1
    return spans


def balance_sentence(sentence: str) -> Tuple[str, bool]:
    """
    Returns balanced sentence and False if its span stopped,
    True if it continues.
    """
    continues = False
    left, right = False, False
    for i in range(len(sentence) - 1):
        cur_slice = sentence[i : i + 2]
        if cur_slice == "[[":
            left = True
        if cur_slice == "]]":
            if left:
                break
            else:
                sentence = "[[" + sentence
    n = len(sentence)

    for i in range(len(sentence) - 1):
        cur_slice = sentence[n - i - 2 : n - i]
        if cur_slice == "]]":
            right = True
        if cur_slice == "[[":
            if right:
                break
            else:
                sentence = sentence + "]]"
                continues = True
    return sentence, continues


def insert_spans(article_text: str, spans: List[Tuple[int, int]]) -> List[str]:
    sentence_starts = [0]
    for i, c in enumerate(article_text):
        if c == "\n":
            sentence_starts.append(i + 1)
    new_text = []
    text_pos = 0
    for span in spans:
        start, end = span
        new_text.append(article_text[text_pos:start])
        new_text.append("[[")
        new_text.append(article_text[start:end])
        text_pos = end
        new_text.append("]]")
    if text_pos < len(article_text):
        new_text.append(article_text[text_pos:])
    new_text = "".join(new_text)
    continues = False
    sentences = new_text.split("\n")
    assert len(sentences) == len(sentence_starts)
    sentences = zip(sentences, sentence_starts)
    sentences = list(filter(lambda x: x[0] != "", sentences))
    sentences, sentence_starts = zip(*sentences)
    new_sentences = []
    for i, sentence in enumerate(sentences):
        if continues:
            sentence, continues = balance_sentence("[[" + sentence)
        else:
            sentence, continues = balance_sentence(sentence)
        new_sentences.append(sentence + "\t" + str(sentence_starts[i]))
    return new_sentences


def generate_dataset(
    article_folder: str,
    labels_si_folder: str = None,
    dataset_filename: str = "parsed.txt",
):
    f = open(dataset_filename, "w", encoding="utf8")
    articles = filter(lambda x: not x.startswith("."), os.listdir(article_folder))
    for article in articles:
        article_path = os.path.join(article_folder, article)
        article_id = re.match(r"article(\d+)\.txt", article).groups()[0]
        f.write("-" * 25)
        f.write(article_id)
        f.write("-" * 25)
        f.write("\n")
        article_text = open(article_path, encoding="utf8").read()
        spans = []
        if labels_si_folder:
            span_path = os.path.join(
                labels_si_folder, f"article{article_id}.task1-SI.labels"
            )
            spans = get_spans_si(span_path)
        for sentence in insert_spans(article_text, spans):
            f.write(sentence)
            f.write("\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--articles-folder",
        type=str,
        required=True,
        help="Path to folder with articles. Usually path/to/datasets/train-articles.",
    )
    parser.add_argument(
        "--articles-labels-si-folder",
        type=str,
        default=None,
        help="Path to folder with SI positions."
        "Usually path/to/datasets/train-labels-task1-span-identification"
        "Or None for test set.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="parsed.txt",
        help="Parsed dataset filename.",
    )
    args = parser.parse_args()
    generate_dataset(
        args.articles_folder, args.articles_labels_si_folder, args.output_filename
    )
