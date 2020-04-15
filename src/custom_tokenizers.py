#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from typing import List

import spacy
from nltk.tokenize import word_tokenize
from allennlp.data.tokenizers import Token, Tokenizer
from tokenizers import BertWordPieceTokenizer


@Tokenizer.register("BertTokenizer")
class BertTokenizer:
    def __init__(self, pretrained_name: str = "bert-base-cased-vocab.txt"):
        self.tokenizer = BertWordPieceTokenizer(pretrained_name, lowercase=False)

    def tokenize(self, s: str, offset: int = 0) -> List[Token]:
        output = self.tokenizer.encode(s)
        result = []
        n = len(output.tokens)
        for i, (bpe, pos, token_id) in enumerate(
            zip(output.tokens, output.offsets, output.ids)
        ):
            if i == 0 or i == n - 1:
                continue
            result.append(Token(bpe, idx=offset + pos[0], text_id=token_id))
        return result


@Tokenizer.register("NltkTokenizer")
class NltkTokenizer(Tokenizer):
    def tokenize(self, s: str, offset: int = 0) -> List[Token]:
        tokens = word_tokenize(s)
        result = []
        start = 0
        for token in tokens:
            pos = s.find(token, start)
            result.append(Token(token, idx=pos + offset))
            start = pos + len(token)
        return result


@Tokenizer.register("SpacyTokenizer")
class SpacyTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = spacy.load("en_core_web_sm")

    def tokenize(self, s: str, offset: int = 0) -> List[Token]:
        tokens = [token.text for token in self.tokenizer(s)]
        result = []
        start = 0
        for token in tokens:
            pos = s.find(token, start)
            result.append(Token(token, idx=pos + offset))
            start = pos + len(token)
        return result
