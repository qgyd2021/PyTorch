#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

"""
import argparse
import json
import os
import random
from pathlib import Path
import platform
from typing import List

import librosa
import numpy as np
import torch
from torch.nn import functional
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from toolbox.librosa import filters
from toolbox.torch.modules.loss import FocalLoss
from toolbox.torch.training.metrics.categorical_accuracy import CategoricalAccuracy
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subset", default="train.jsonl", type=str)
    parser.add_argument("--valid_subset", default="valid.jsonl", type=str)

    parser.add_argument("--vocabulary_dir", default="vocabulary", type=str)

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--keep_most_recent_by_count", default=10, type=int)
    parser.add_argument("--patience", default=-1, type=int)
    parser.add_argument("--serialization_dir", default="serialization_dir", type=str)

    parser.add_argument("--lr_scheduler_step_size", default=50, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=0.5, type=float)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")

    args = parser.parse_args()
    return args


class AudioClassificationJsonDataset(Dataset):
    def __init__(self,
                 audio_key: str = "filename",
                 label_key: str = "label"
                 ):
        self.audio_key = audio_key
        self.label_key = label_key

        self.examples: List[dict] = list()

    def read(self, filename: str):
        examples = list()
        with open(filename, "r", encoding="utf-8") as f:
            for row in f:
                row = json.loads(row)

                row_ = {
                    "filename": row[self.audio_key],
                    "label": row[self.label_key],
                }
                examples.append(row_)

        self.examples = examples

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def __len__(self):
        return len(self.examples)


class MelSpectrumFeature(object):
    def __init__(self,
                 max_wave_value: float = 1.0,
                 sample_rate: int = 8000,
                 n_fft: int = 512,
                 n_mels: int = 80,
                 fmin: int = 0,
                 fmax: int = 4000,
                 hop_size: int = 128,
                 win_size: int = 512,
                 center: bool = False,
                 ):
        # 32768.0
        self.max_wave_value = max_wave_value
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_size = hop_size
        self.win_size = win_size
        self.center = center

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: shape=[batch_size, length]
        :return: spectrum, shape=[batch_size, time_steps, n_mels]
        """
        inputs: torch.Tensor = inputs / self.max_wave_value

        if torch.min(inputs) < -1.:
            raise AssertionError()
        if torch.max(inputs) > 1.:
            raise AssertionError()

        device = inputs.device

        # linux 上安装 librosa 比较麻烦.
        mel = filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # shape=[n_mels, spec_dim]
        mel = torch.from_numpy(mel).float().to(device)

        inputs = torch.nn.functional.pad(
            input=inputs,
            pad=(int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode='reflect'
        )

        spec = torch.stft(
            inputs,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=torch.hann_window(self.win_size).to(device),
            center=self.center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )

        # shape=[batch_size, spec_dim, time_steps]
        spec = torch.abs(spec)
        # shape=[batch_size, n_mels, time_steps]
        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1)

        # shape=[batch_size, time_steps, n_mels]
        spec = torch.transpose(spec, dim0=1, dim1=2)
        return spec

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class CollateFunction(object):
    def __init__(self,
                 vocabulary: Vocabulary,
                 feature_extractor: MelSpectrumFeature,
                 resample_rate: int = 8000,
                 label_namespace: str = "labels"
                 ):
        self.vocabulary = vocabulary
        self.feature_extractor = feature_extractor
        self.resample_rate = resample_rate
        self.label_namespace = label_namespace

    def read_wav(self, filename: str):
        """signal has values between 0 and 1"""
        signal, _ = librosa.load(filename, sr=self.resample_rate)
        return signal

    def __call__(self, batch: List[dict]):
        waves = list()
        label_list = list()
        for example in batch:
            filename = example["filename"]
            signal = self.read_wav(filename)
            waves.append(signal)

            label = example["label"]
            label_idx = self.vocabulary.get_token_index(token=label, namespace=self.label_namespace)
            label_list.append(label_idx)

        waves = np.array(waves, dtype=np.float32)
        waves = torch.from_numpy(waves)
        inputs: torch.Tensor = self.feature_extractor(waves)

        targets = np.array(label_list, dtype=np.int32)
        targets: torch.LongTensor = torch.from_numpy(targets)
        return inputs, targets


class VoicemailModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels

        # cnn
        self.layer_norm = nn.LayerNorm(normalized_shape=80, eps=1e-05)
        self.conv1d_0 = nn.Conv1d(80, 64, kernel_size=(5,), stride=(1,), padding='same')
        self.relu_0 = nn.ReLU()
        self.dropout_0 = nn.Dropout(p=0.1, inplace=False)

        # cnn
        self.conv1d_1 = nn.Conv1d(64, 64, kernel_size=(5,), stride=(1,), padding='same')
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1, inplace=False)

        # feedforward
        self.linear_2 = nn.Linear(64, 64)
        self.dropout_2 = nn.Dropout(p=0.1, inplace=False)
        self.layer_norm_2_0 = nn.LayerNorm(64)

        # attention
        self.query_3 = nn.Linear(64, 64, bias=True)
        self.key_3 = nn.Linear(64, 64, bias=False)
        self.value_3 = nn.Linear(64, 64, bias=False)
        self.dropout_3 = nn.Dropout(p=0.1, inplace=False)
        self.linear_3 = nn.Linear(64, 64)
        self.layer_norm_2_1 = nn.LayerNorm(64)

        # classifier
        self.classification_layer = torch.nn.Linear(64, self.num_labels)

    def forward(self, inputs: torch.Tensor):
        # shape=[batch_size, seq_length, dim]
        x = inputs

        # norm
        x = self.layer_norm(x)

        # shape=[batch_size, dim, seq_length]
        x = torch.transpose(x, dim0=-1, dim1=-2)

        # cnn1
        x = self.conv1d_0(x)
        x = self.relu_0(x)
        x = self.dropout_0(x)

        # cnn2
        x = self.conv1d_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)

        # shape=[batch_size, dim, seq_length] -> shape=[batch_size, seq_length, dim]
        x = torch.transpose(x, dim0=-1, dim1=-2)

        # feedforward
        # shape=[batch_size, seq_length, dim]
        x = self.linear_2(x)
        x = self.dropout_2(x)
        x = self.layer_norm_2_0(x)

        # attention
        # shape=[batch_size, seq_length, dim]
        query = self.query_3(x)
        similarities = self.key_3(query)
        attention = functional.softmax(similarities, dim=-1)
        attention = self.dropout_3(attention)
        x = self.value_3(attention)
        x = self.layer_norm_2_0(x)

        # seq2vec
        # shape=[batch_size, seq_length, dim]
        x = torch.mean(x, dim=1)

        # classifier
        logits = self.classification_layer(x)
        return logits


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    n_gpu = torch.cuda.device_count()

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    categorical_accuracy = CategoricalAccuracy(top_k=1)
    loss_fn = FocalLoss(vocabulary.get_vocab_size(namespace="labels"))

    train_dataset = AudioClassificationJsonDataset()
    train_dataset.read(args.train_subset)
    valid_dataset = AudioClassificationJsonDataset()
    valid_dataset.read(args.valid_subset)

    collate_fn = CollateFunction(
        vocabulary=vocabulary,
        feature_extractor=MelSpectrumFeature(
            sample_rate=8000,
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    model = VoicemailModel(num_labels=vocabulary.get_vocab_size(namespace="labels"))
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(
        optimizer=optimizer,
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_gamma
    )

    # global params
    best_epoch: int = -1
    best_validation_accuracy: float = -1
    best_validation_loss: float = float("inf")

    model_state_filename_list = list()
    training_state_filename_list = list()

    for epoch_idx in range(args.epochs):

        # train
        categorical_accuracy.reset()
        training_total_loss: float = 0.0
        training_total_steps: int = 0
        progress_bar = tqdm(train_dataloader, desc='Epoch={} (train)'.format(epoch_idx), leave=True)
        for step, batch in enumerate(progress_bar):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets: torch.LongTensor = targets.to(device).long()

            logits = model.forward(inputs)

            loss: torch.Tensor = loss_fn.forward(logits, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predictions = torch.softmax(logits, dim=-1)
            categorical_accuracy(predictions, gold_labels=targets)

            training_total_loss += loss.item()
            training_total_steps += 1

            # progress_bar
            progress_bar_postfix = {
                "loss": training_total_loss / training_total_steps,
                "accuracy": categorical_accuracy.get_metric()
            }
            progress_bar.set_postfix(**progress_bar_postfix)
        training_accuracy: float = categorical_accuracy.get_metric()
        training_loss: float = training_total_loss / training_total_steps

        # validation
        categorical_accuracy.reset()
        validation_total_loss: float = 0.0
        validation_total_steps: int = 0
        progress_bar = tqdm(valid_dataloader, desc='Epoch={} (valid)'.format(epoch_idx), leave=True)
        with torch.no_grad():
            for step, batch in enumerate(progress_bar):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets: torch.LongTensor = targets.to(device).long()

                logits = model.forward(inputs)

                loss: torch.Tensor = loss_fn.forward(logits, targets)

                predictions = torch.softmax(logits, dim=-1)
                categorical_accuracy(predictions, gold_labels=targets)

                validation_total_loss += loss.item()
                validation_total_steps += 1

                # progress_bar
                progress_bar_postfix = {
                    "loss": validation_total_loss / validation_total_steps,
                    "accuracy": categorical_accuracy.get_metric()
                }
                progress_bar.set_postfix(**progress_bar_postfix)
        validation_accuracy = categorical_accuracy.get_metric()
        validation_loss: float = validation_total_loss / validation_total_steps

        # lr scheduler
        lr_scheduler.step()

        # serialization
        serialization_dir = Path(args.serialization_dir)
        serialization_dir.mkdir(exist_ok=True)

        # keep most recent by count
        if len(model_state_filename_list) >= args.keep_most_recent_by_count > 0:
            model_state_filename_ = model_state_filename_list.pop(0)
            os.remove(model_state_filename_)
            training_state_filename_ = training_state_filename_list.pop(0)
            os.remove(training_state_filename_)

        model_state_filename = serialization_dir / "model_state_{}.th".format(epoch_idx)
        training_state_filename = serialization_dir / "training_state_{}.th".format(epoch_idx)
        model_state_filename_list.append(model_state_filename.as_posix())
        with open(model_state_filename.as_posix(), "wb") as f:
            torch.save(model.state_dict(), f)
        training_state_filename_list.append(training_state_filename.as_posix())
        with open(training_state_filename.as_posix(), "wb") as f:
            torch.save(optimizer.state_dict(), f)

        # early stop
        if validation_accuracy > best_validation_accuracy:
            with open(serialization_dir / "best.th", "wb") as f:
                torch.save(model.state_dict(), f)

            best_epoch = epoch_idx
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss

        if epoch_idx - best_epoch >= args.patience > 0:
            # early stop
            break

        # metrics epoch
        metrics = {
            "best_epoch": best_epoch,

            "epoch": epoch_idx,

            "training_accuracy": training_accuracy,
            "training_loss": training_loss,

            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
        }
        metrics_epoch_filename = serialization_dir / "metrics_epoch_{}.json".format(epoch_idx)
        with open(metrics_epoch_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

    return


if __name__ == '__main__':
    main()
