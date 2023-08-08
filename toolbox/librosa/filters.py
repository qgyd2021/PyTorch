#!/usr/bin/python3
# -*- coding: utf-8 -*-
import warnings

import numpy as np

from toolbox.librosa.core.convert import fft_frequencies, mel_frequencies
from toolbox.librosa import util


def mel(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm="slaney",
    dtype=np.float32,
):
    if fmax is None:
        fmax = float(sr) / 2

    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)

    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = util.normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


if __name__ == '__main__':
    pass
