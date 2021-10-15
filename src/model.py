#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (С) ABBYY (BIT Software), 1993 - 2020. All rights reserved.
from typing import Dict, Any, List

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure
from allennlp.modules import FeedForward
from allennlp.modules import ConditionalRandomField

from src.laserdecoder import LaserDecoder
from src.utils import generate_spans
from src.metric import MultilabelMicroF1, Accuracy
from src.modules import SpanClassifier


@Model.register("UniversalTagger")
class UniversalTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        emb_to_enc_proj: FeedForward = None,
        feedforward: FeedForward = None,
        dropout: float = 0.0,
        num_tags: int = 2,
        use_crf: bool = False,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.emb_to_enc_proj = None
        if emb_to_enc_proj is not None:
            self.emb_to_enc_proj = emb_to_enc_proj
        self.encoder = encoder
        assert (
            embedder.get_output_dim() == encoder.get_input_dim()
            or emb_to_enc_proj is not None
            and emb_to_enc_proj.get_output_dim() == encoder.get_input_dim()
        )
        self.feedforward = None
        pre_output_dim = encoder.get_output_dim()
        if feedforward is not None:
            assert feedforward.get_input_dim() == encoder.get_output_dim()
            self.feedforward = feedforward
            pre_output_dim = self.feedforward.get_output_dim()

        self.hidden2tag = torch.nn.Linear(
            in_features=pre_output_dim, out_features=num_tags
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(1)
        self.use_crf = use_crf
        if use_crf:
            self.crf = ConditionalRandomField(
                num_tags, include_start_end_transitions=True
            )

    def forward(
        self, sentence: Dict[str, torch.Tensor], labels: torch.Tensor = None, **kwargs
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)
        embeddings = self.embedder(sentence)
        embeddings = self.dropout(embeddings)
        if self.emb_to_enc_proj is not None:
            embeddings = self.emb_to_enc_proj(embeddings)
        encoder_out = self.encoder(embeddings, mask)
        if self.feedforward is not None:
            encoder_out = self.feedforward(encoder_out)
        output = {}
        tag_logits = self.hidden2tag(encoder_out)
        if self.use_crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            best_paths = [x[0] for x in best_paths]
            output["best_paths"] = best_paths
            if labels is not None:
                output["loss"] = -self.crf(tag_logits, labels, mask)
                class_probabilities = tag_logits * 0.0
                for i, instance_tags in enumerate(best_paths):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
                self.accuracy(class_probabilities, labels, mask)
                self.f1(class_probabilities, labels, mask)
        else:
            output["tag_logits"] = tag_logits
            if labels is not None:
                self.accuracy(tag_logits, labels, mask)
                self.f1(tag_logits, labels, mask)
                output["loss"] = sequence_cross_entropy_with_logits(
                    tag_logits, labels, mask
                )
        output["mask"] = mask
        output["metadata"] = kwargs.get("metadata")
        return output

    def decode(self, output_dict):
        new_output = {}
        mask = output_dict["mask"]
        lengths = torch.sum(mask, dim=1).tolist()  # batch
        words_all = []
        if self.use_crf:
            tag_logits = output_dict["best_paths"]
        else:
            tag_logits = output_dict["tag_logits"]  # batch, seq_len, num_tags
            tag_logits = torch.argmax(tag_logits, dim=2).tolist()  # batch, seq_len
        for word_idxs, length in zip(tag_logits, lengths):
            words = []
            for idx, _ in zip(word_idxs, range(length)):
                words.append(idx)
            words_all.append(words)
        new_output["tags"] = words_all
        new_output["spans"] = []
        metadata = output_dict["metadata"]
        for i in range(len(tag_logits)):
            tokens = metadata[i]["tokens"]
            tags = new_output["tags"][i]
            assert len(tokens) == len(tags)
            sentence_offset = int(metadata[i]["sentence_pos"])
            article_id = metadata[i]["id"]
            new_output["spans"].append(
                generate_spans(tokens, tags, sentence_offset, article_id)
            )
        return new_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = {"accuracy": self.accuracy.get_metric(reset)}
        p, r, f1 = self.f1.get_metric(reset)
        d["prec"] = p
        d["rec"] = r
        d["f1"] = f1
        return d


@Model.register("LaserTagger")
class LaserTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        dropout: float = 0.2,
        decoder_hidden_dim: int = 128,
        decoder_ff_dim: int = 128,
        decoder_num_layers: int = 1,
        decoder_num_heads: int = 4,
        teacher_forcing: float = 1.0,
        num_teacher_forcing_steps: int = None,
        num_tags: int = 2,
        label_smoothing: float = None,
    ):
        super().__init__(vocab)
        # teacher forcing is how often we choose to force the correct answer.
        self.embedder = embedder
        self.encoder = encoder
        self.laserdecoder = LaserDecoder(
            hidden_dim=decoder_hidden_dim,
            encoder_dim=encoder.get_output_dim(),
            num_layers=decoder_num_layers,
            ff_dim=decoder_ff_dim,
            num_heads=decoder_num_heads,
            num_classes=num_tags,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(1)
        self.teacher_forcing = teacher_forcing
        self.num_tf_steps = num_teacher_forcing_steps
        self.cur_tf_steps = 0
        self.label_smoothing = label_smoothing

    def forward(
        self, sentence: Dict[str, torch.Tensor], labels: torch.Tensor = None, **kwargs
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)
        batch_size, seq_len = mask.shape
        embeddings = self.embedder(sentence)
        embeddings = self.dropout(embeddings)
        encoder_out = self.encoder(embeddings, mask)
        # encoder_out - batch_size, seq_len, encoder_output_dim
        # tag_logits - batch_size, seq_len, num_classes
        output = dict()
        output["mask"] = mask
        if labels is not None and self.training and self.teacher_forcing == 1.0:
            # labels - batch_size, seq_len
            train_labels = torch.cat((labels.new_zeros(batch_size, 1), labels), dim=1)[
                :, :-1
            ]
            tag_logits = self.laserdecoder(
                encoder_out, train_labels
            )  # batch_size, seq_len, num_labels
        else:
            inputs = embeddings.new_zeros((batch_size, 1), dtype=torch.long)
            for i in range(seq_len):
                tag_logits = self.laserdecoder(
                    encoder_out[:, : inputs.size(1)], inputs
                )  # batch_size, cur_len
                next_token = torch.argmax(tag_logits[:, -1], dim=1).unsqueeze(
                    1
                )  # batch_size, 1
                if (
                    labels is not None
                    and self.training
                    and torch.rand(1).item() < self.teacher_forcing
                ):
                    next_token = labels[:, i].unsqueeze(1)
                inputs = torch.cat(
                    (inputs, next_token), dim=1
                )  # batch_size, cur_len + 1
        output["tag_logits"] = tag_logits
        if labels is not None:
            # print(tag_logits.shape, labels.shape, mask.shape)
            self.accuracy(tag_logits, labels, mask)
            self.f1(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask, label_smoothing=self.label_smoothing
            )
        output["metadata"] = kwargs.get("metadata")
        if self.num_tf_steps is not None and self.training:
            self.cur_tf_steps += 1
            self.teacher_forcing = max(
                0.0, (self.num_tf_steps - self.cur_tf_steps) / self.num_tf_steps
            )
        return output

    def decode(self, output_dict):
        new_output = {}
        tag_logits = output_dict["tag_logits"]  # batch, seq_len, num_tags
        mask = output_dict["mask"]
        tag_logits = torch.argmax(tag_logits, dim=2).tolist()  # batch, seq_len
        lengths = torch.sum(mask, dim=1).tolist()  # batch
        words_all = []
        for word_idxs, length in zip(tag_logits, lengths):
            words = []
            for idx, _ in zip(word_idxs, range(length)):
                words.append(idx)
            words_all.append(words)
        new_output["tags"] = words_all
        new_output["spans"] = []
        metadata = output_dict["metadata"]
        for i in range(len(tag_logits)):
            tokens = metadata[i]["tokens"]
            tags = new_output["tags"][i]
            assert len(tokens) == len(tags)
            sentence_offset = int(metadata[i]["sentence_pos"])
            article_id = metadata[i]["id"]
            new_output["spans"].append(
                generate_spans(tokens, tags, sentence_offset, article_id)
            )
        return new_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = {"accuracy": self.accuracy.get_metric(reset)}
        d.update(self.f1.get_metric(reset))
        return d


@Model.register("TaskTIClassifier")
class TaskTIModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        feature_encoder: SpanClassifier,
        num_classes: int = 14,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.feature_encoder = feature_encoder
        self.hidden2tag = torch.nn.Linear(feature_encoder.get_output_dim(), num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.acc = Accuracy()
        self.f1 = MultilabelMicroF1()
        self.idx2label = vocab.get_index_to_token_vocabulary("labels")

    @staticmethod
    def get_labels(logits: torch.Tensor, num_pred: List[int]) -> torch.LongTensor:
        # logits - batch_size, num_classes
        predicted_labels = torch.zeros_like(
            logits, dtype=torch.long
        )  # [batch_size, num_classes]
        ones = torch.ones_like(logits[0], dtype=torch.long)  # [num_classes]
        for i, row in enumerate(logits):
            # row - [num_classes]
            indice = torch.topk(row, num_pred[i], dim=0).indices  # num_pred
            predicted_labels[i].scatter_(0, indice, ones)
        return predicted_labels

    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        labels: torch.LongTensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        output = {}
        embedded = self.embedder(sentence)
        # embedded - batch_size, seq_len, hidden_dim
        pos = [
            (instance_metadata["start"], instance_metadata["end"])
            for instance_metadata in metadata
        ]
        num_pred = [instance_metadata["num_pred"] for instance_metadata in metadata]
        encoded_features = self.feature_encoder(embedded, pos)
        logits = self.hidden2tag(encoded_features)
        # logits - [batch_size, num_classes]
        output["logits"] = logits
        pred_labels = self.get_labels(logits, num_pred)
        output["pred_labels"] = pred_labels
        if labels is not None:
            loss = self.criterion(logits, labels.float())
            output["loss"] = loss
            self.acc(pred_labels, labels)
            self.f1(pred_labels, labels)
        output["metadata"] = metadata
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = self.f1.get_metric(reset)
        d.update(self.acc.get_metric(reset))
        return d

    def decode(self, output: Dict[str, Any]):
        pred_labels = output["pred_labels"].tolist()  # batch, num_labels
        output_strings = []
        for i, class_labels in enumerate(pred_labels):
            class_string = ""
            for j, class_value in enumerate(class_labels):
                if class_value == 1:
                    class_string += (
                        output["metadata"][i]["output_str"] % self.idx2label[j]
                    )
            output_strings.append(class_string)
        output["final_output"] = output_strings
        return output
