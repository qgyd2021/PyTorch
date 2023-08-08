#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

import librosa
import numpy as np
import torch
from torch.nn import functional
import torch.nn as nn

from project_settings import project_path
from toolbox.librosa import filters
from toolbox.torch.utils.data.vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default=(project_path / "examples/exercises/early_media_voice_classification/data_dir/254/254_segmented/bell/254_bell_0_4.wav").as_posix(),
        type=str
    )

    parser.add_argument('--vocabulary_dir', default='vocabulary', type=str)
    parser.add_argument('--weights_file', default='serialization_dir/best.th', type=str)

    parser.add_argument('--trace_model_filename', default='trace_model.zip', type=str)
    parser.add_argument('--trace_quantized_model_filename', default='trace_quantized_model.zip', type=str)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=3407, type=str, help="https://arxiv.org/abs/2109.08203")

    args = parser.parse_args()
    return args


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
        # self.batch_norm = nn.BatchNorm1d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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

    vocabulary = Vocabulary.from_files(args.vocabulary_dir)

    collate_fn = CollateFunction(
        vocabulary=vocabulary,
        feature_extractor=MelSpectrumFeature(
            sample_rate=8000,
        )
    )

    model = VoicemailModel(num_labels=vocabulary.get_vocab_size(namespace="labels"))

    with open(args.weights_file, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    example = {
        "filename": args.filename,
        "label": "voice"
    }

    inputs, targets = collate_fn.__call__([example])

    with torch.no_grad():
        logits = model.forward(inputs)
        probs = torch.softmax(logits, dim=-1)
        label_idx = torch.argmax(probs, dim=-1)
        label_idx = label_idx.numpy().tolist()
        probs = probs.numpy().tolist()
        for l_idx, pbs in zip(label_idx, probs):
            label_str = vocabulary.get_token_from_index(index=l_idx, namespace="labels")
            prob = pbs[l_idx]
            print("label: {}, prob: {}".format(label_str, round(prob, 4)))

    # export torch script
    trace_model = torch.jit.trace(func=model, example_inputs=inputs, strict=False)
    trace_model.save(args.trace_model_filename)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    trace_quantized_model = torch.jit.trace(func=quantized_model, example_inputs=inputs, strict=False)
    trace_quantized_model.save(args.trace_quantized_model_filename)

    return


if __name__ == '__main__':
    main()
