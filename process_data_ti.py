#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import os
import re
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict


def parse_ti_file(filename: str) -> Dict[str, List[Tuple[str, int, int]]]:
    spans = defaultdict(list)
    for line in open(filename, "r", encoding="utf8"):
        article_id, prop_type, start, end = line.strip().split("\t")
        spans[article_id].append((prop_type, int(start), int(end)))
    return spans


def get_full_span(text: str, start: int, end: int) -> str:
    l = len(text)
    # find start
    i = start + 1
    while i > 0 and text[i] != "\n":
        i -= 1
    new_start = i + 1
    # find end
    i = end
    while i < l and text[i] != "\n":
        i += 1
    new_end = i
    return "".join(
        (
            text[new_start:start],
            " [unused1] ",
            text[start:end],
            " [unused2] ",
            text[end:new_end],
        )
    )


def generate_dataset(
    article_folder: str,
    labels_ti_file: str = None,
    dataset_filename: str = "parsed.txt",
):
    f = open(dataset_filename, "w", encoding="utf8")
    spans = parse_ti_file(labels_ti_file)
    for article_id, article_spans in spans.items():
        article_spans = sorted(article_spans, key=lambda x: (x[1], x[2]))
        article_path = os.path.join(article_folder, f"article{article_id}.txt")
        article_text = open(article_path, encoding="utf8").read()
        prop_type, start, end = article_spans[0]
        prop_types = [prop_type]
        i = 1
        while i < len(article_spans):
            if article_spans[i][1] == start and article_spans[i][2] == end:
                prop_types.append(article_spans[i][0])
            else:
                text_piece = get_full_span(article_text, start, end)
                text_piece = re.sub(r"\s+", " ", text_piece)
                f.write(text_piece)
                f.write("\t")
                f.write(";".join(prop_types))
                f.write("\t")
                f.write(article_id)
                f.write("\t")
                f.write(str(start))
                f.write("\t")
                f.write(str(end))
                f.write("\n")
                prop_type, start, end = article_spans[i]
                prop_types = [prop_type]
            i += 1
        text_piece = get_full_span(article_text, start, end)
        text_piece = re.sub(r"\s+", " ", text_piece)
        f.write(text_piece)
        f.write("\t")
        f.write(";".join(prop_types))
        f.write("\t")
        f.write(article_id)
        f.write("\t")
        f.write(str(start))
        f.write("\t")
        f.write(str(end))
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
        "--articles-labels-ti-folder",
        type=str,
        default=None,
        help="Path to folder with TI spans."
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
        args.articles_folder, args.articles_labels_ti_folder, args.output_filename
    )
