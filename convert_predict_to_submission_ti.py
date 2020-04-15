#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import argparse
import json
from typing import Dict, Tuple, List
from collections import defaultdict


def read_pred_file(filename: str) -> Dict[Tuple[str, str, str], List[str]]:
    d = defaultdict(list)
    for line in open(filename, "r", encoding="utf8"):
        line = line.strip()
        article_id, prop_type, start, end = line.split("\t")
        d[(article_id, start, end)].append(prop_type)
    return d


def read_pred_file_json(filename: str) -> Dict[Tuple[str, str, str], List[str]]:
    d = defaultdict(list)
    for json_line in open(filename, "r", encoding="utf8"):
        json_dict = json.loads(json_line)
        final_output = json_dict["final_output"]
        for line in final_output.strip().split("\n"):
            article_id, prop_type, start, end = line.split("\t")
            d[(article_id, start, end)].append(prop_type)
    return d


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-p",
        "--prediction",
        type=str,
        required=True,
        help="File with model predictions.",
    )
    argparser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Original file with correct order.",
    )
    argparser.add_argument(
        "-o", "--output", type=str, required=True, help="File to write submission to."
    )
    argparser.add_argument(
        "-j",
        "--json",
        action="store_true",
        default=False,
        help="Bool flag, set to True"
        "if predicitons are in"
        "json format from"
        "allennlp predict.",
    )
    args = argparser.parse_args()
    print(args)
    pred_file = args.prediction
    orig_file = args.source
    outfile = args.output
    parse_json = args.json
    if parse_json:
        d = read_pred_file_json(pred_file)
    else:
        d = read_pred_file(pred_file)
    fout = open(outfile, "w", encoding="utf8")
    for line in open(orig_file, "r", encoding="utf8"):
        article_id, _, start, end = line.strip().split("\t")
        prop_type = d[(article_id, start, end)].pop(0)
        fout.write("\t".join((article_id, prop_type, start, end)))
        fout.write("\n")
