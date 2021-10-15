#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from typing import Iterator, List, Dict

from allennlp.data import Instance
from allennlp.data.fields import (
    TextField,
    SequenceLabelField,
    MetadataField,
    MultiLabelField,
)
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from src.custom_tokenizers import SpacyTokenizer


def split_sentence(sentence: str):
    start = 0
    i = 0
    parts, tags = [], []
    while i < len(sentence) - 1:
        if sentence[i] == "[" and sentence[i + 1] == "[":
            parts.append(sentence[start:i])
            tags.append(0)
            start = i + 2
            while sentence[i] != "]" or sentence[i + 1] != "]":
                i += 1
            parts.append(sentence[start:i])
            tags.append(1)
            start = i + 2
            i += 2
        else:
            i += 1
    parts.append(sentence[start:])
    tags.append(0)
    return parts, tags


@DatasetReader.register("TaskSIReader")
class TaskSIReader(DatasetReader):
    def __init__(
        self,
        tagging_scheme: str = "binary",
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_bert: bool = False,
    ):
        super().__init__()
        self.tagging_scheme = tagging_scheme
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
            "chars": TokenCharactersIndexer(),
        }
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.use_bert = use_bert
        if use_bert:
            print("REMEMBER ABOUT HARDCODED [CLS] AND [SEP] token_ID!")

    def text_to_instance(
        self,
        tokens: List[Token],
        tags: List[str] = None,
        line: str = None,
        curid: str = None,
        sentence_pos: str = None,
    ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        fields["metadata"] = MetadataField(
            {"line": line, "id": curid, "sentence_pos": sentence_pos, "tokens": tokens}
        )
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        f = open(file_path, encoding="utf8")
        curid = None
        for line in f:
            line = line.strip()
            if line.startswith("-" * 25):
                curid = line.strip("-")
                continue
            line, sentence_pos = line.split("\t")
            if not self.use_bert:
                sentence, tags = [Token(START_SYMBOL)], [0]
            else:
                sentence, tags = [Token("[CLS]", text_id=101, idx=0)], [0]
            parts, part_tags = split_sentence(line)
            offset = 0
            for k, (part, tag) in enumerate(zip(parts, part_tags)):
                tokens = self.tokenizer.tokenize(part, offset)
                offset += len(part)
                sentence += tokens
                tags += [tag] * len(tokens)
            if not self.use_bert:
                sentence.append(Token(END_SYMBOL))
            else:
                sentence.append(Token("[SEP]", text_id=10, idx=sentence[-1].idx))
            tags.append(0)
            yield self.text_to_instance(sentence, tags, line, curid, sentence_pos)


@DatasetReader.register("TaskTIReader")
class TaskTIReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        include_sentinels: bool = True,
        sentinel_str: str = "^",
    ):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
            "chars": TokenCharactersIndexer(),
        }
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.include_sentinels = include_sentinels
        self.sentinel_token = self.tokenizer.tokenize(sentinel_str)[0]
        print("Remember about hardcoded [CLS] & [SEP] token ids.")

    def text_to_instance(
        self,
        tokens: List[Token],
        prop_types: List[str],
        start: int,
        end: int,
        article_id: str,
        line: str,
        meta_start: str,
        meta_end: str,
        num_predictions: str,
    ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        fields["metadata"] = MetadataField(
            {
                "article_id": article_id,
                "line": line,
                "start_token": tokens[start].text,
                "end_token": tokens[end].text,
                "output_str": f"{article_id}\t%s\t{meta_start}\t{meta_end}\n",
                "start": start,
                "end": end,
                "num_pred": num_predictions,
                "prop_type": prop_types,
            }
        )
        if prop_types is not None:
            label_field = MultiLabelField(prop_types, num_labels=14)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        f = open(file_path, encoding="utf8")
        for line in f:
            line = line.strip()
            text, prop_types, article_id, meta_start, meta_end = line.strip().split(
                "\t"
            )
            if "?" in prop_types:
                num_predictions = prop_types.count(";") + 1
                prop_types = None
            else:
                num_predictions = prop_types.count(";") + 1
                prop_types = prop_types.split(";")
            left, right = text.split("[unused1]")
            middle, right = right.split("[unused2]")
            tokens = [Token("[CLS]", text_id=101)]
            tokens += self.tokenizer.tokenize(left)
            if self.include_sentinels:
                tokens.append(self.sentinel_token)
            # I want start end to be indexed as [start: end], so start should be first word after unused1
            # and end should be unused2
            start = len(tokens)
            tokens += self.tokenizer.tokenize(middle)
            end = len(tokens)
            if self.include_sentinels:
                tokens.append(self.sentinel_token)
            tokens += self.tokenizer.tokenize(right)
            tokens.append(Token("[SEP]", text_id=102))
            yield self.text_to_instance(
                tokens,
                prop_types,
                start,
                end,
                article_id,
                line,
                meta_start,
                meta_end,
                num_predictions,
            )
