#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
import sys
import json

fin = open(sys.argv[1], encoding="utf8")
fout = open(sys.argv[2], "w", encoding="utf8")

for line in fin:
    d = json.loads(line)
    if d["spans"] != "":
        fout.write(d["spans"])
