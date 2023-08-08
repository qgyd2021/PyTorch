#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

from toolbox.librosa.util.exceptions import ParameterError


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):

    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    mag = np.abs(S).astype(np.float)

    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def tiny(x):
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


if __name__ == '__main__':
    pass
