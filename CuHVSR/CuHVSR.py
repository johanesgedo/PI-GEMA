"""
All Rights Reserved

Copyright (c) 2026 Johanes Gedo Sea

THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL.
UNAUTHORIZED COPYING, TRANSFERRING OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM IS STRICTLY PROHIBITED.

The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them
for any purpose other than the purpose for which they were provided to you.

The software is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
the warranties of merchantability, fitness for a particular purpose and non infringement.
In no event shall the authors or copyright holders be liable for any claim, damages or other liability,
whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software
or the use or other dealings in the software.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

CUDA-based Horizontal-to-Vertical Spectral Ratio
"""


from __future__ import annotations

import math
# import cupy as cp
# import numpy as np
import pandas as pd
import json
import traceback
import warnings
import typing as _t
import logging
import inspect
import re
import builtins
import pathlib
import logging
import itertools
import io
import obspy
import click
import contextlib
import time
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Cursor
from multiprocessing import get_context
from cupyx.scipy.signal import butter, sosfiltfilt
from cupyx.scipy.signal.windows import tukey
from scipy.signal import find_peaks
from scipy import stats
from typing import Iterable, Tuple, Any, Optional, Union, Sequence, List, Dict
from abc import ABC
from termcolor import colored
from contextlib import contextmanager
from copy import deepcopy
from IPython.display import display

try:
    import cupy as cp
    xp = cp
    USE_CUPY = True
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.cuda.Device(0).use()
    except Exception:
        pass
except Exception:
    import numpy as np
    xp = np
    cp = None
    USE_CUPY = False

import numpy as np



DEFAULT_EPS = 1e-30
EPS = 1e-12

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["Time_Series_CUDA"]

Number = Union[int, float]


__all__ = ["HvsrCurve"]

__all__ = ["SeismicRecording3C"]

__all__ = ["InstrumentTransferFunction",
           "Differentiate", "Integrate",
           "Remove_Instrument_Response"]

__all__ = ["Psd", "USE_CUPY", "xp"]

__all__ = ["HvsrAzimuthal", "USE_CUPY", "xp"]

__all__ = ["HvsrDiffuseField"]

__all__ = ["HvsrGeopsy", "USE_CUPY", "xp"]

__all__ = ["MonteCarlo", "HvsrSpatial", "USE_CUPY", "xp"]

__all__ = ["HvsrTraditional", "USE_CUPY", "xp"]

__all__ = [
    "write_hvsr_object_to_file",
    "read_hvsr_object_from_file",
    "write_settings_object_to_file",
    "read_settings_object_from_file",
]

__all__ = ["hvsr_preprocess", "psd_preprocess", "preprocess", "PREPROCESSING_METHODS"]

__all__ = [
    "prepare_fft_settings",
    "traditional_hvsr_processing",
    "traditional_single_azimuth_hvsr_processing",
    "traditional_rotdpp_hvsr_processing",
    "rpsd",
    "diffuse_field_hvsr_processing",
    "process",
    "PROCESSING_METHODS",
    "USE_CUPY",
    "xp",
]


__all__ = [
    "sta_lta_window_rejection",
    "maximum_value_window_rejection",
    "frequency_domain_window_rejection",
    "manual_window_rejection",
    "frequency_domain_window_rejection",
    "USE_CUPY",
    "xp",
]



###############################
# Metadata
###############################
__version__ = "1.0.0.0.0"


###############################
# Constants
###############################
DISTRIBUTION_MAP = {
    "log-normal": "lognormal",
    "lognormal": "lognormal",
    "normal": "normal",
}


#################################################
# Regex (Regular Expressions for text parsing)
#################################################
# NEWLINE = r"[\r\n?|\n]"

#=======================
# DataWrangler | saf
#=======================
saf_npts_expr = r"NDAT = (\d+)[\r\n?|\n]"
saf_fs_expr = r"SAMP_FREQ = (\d+)[\r\n?|\n]"
saf_sample_expr = r"-?\d+"
saf_row_expr = r"^(-?\d+)\s(-?\d+)\s(-?\d+)[\r\n?|\n]"
saf_v_ch_expr = r"CH(\d)_ID = V"
saf_n_ch_expr = r"CH(\d)_ID = N"
saf_e_ch_expr = r"CH(\d)_ID = E"
saf_north_rot_expr = r"NORTH_ROT = (\d+)"
saf_version_expr = r"SESAME ASCII data format \(saf\) v. (\d)"

saf_npts_exec = re.compile(saf_npts_expr)
saf_fs_exec = re.compile(saf_fs_expr)
saf_row_exec = re.compile(saf_row_expr, flags=re.MULTILINE)
saf_v_ch_exec = re.compile(saf_v_ch_expr)
saf_n_ch_exec = re.compile(saf_n_ch_expr)
saf_e_ch_exec = re.compile(saf_e_ch_expr)
saf_north_rot_exec = re.compile(saf_north_rot_expr)
saf_version_exec = re.compile(saf_version_expr)


#============================
# DataWrangler | minishark
#============================
mshark_npts_expr = r"#Sample number:\t(\d+)[\r\n?|\n]"
mshark_fs_expr = r"#Sample rate \(sps\):\t(\d+)[\r\n?|\n]"
mshark_gain_expr = r"#Gain:\t(\d+)[\r\n?|\n]"
mshark_conversion_expr = r"#Conversion factor:\t(\d+)[\r\n?|\n]"
mshark_sample_expr = r"-?\d+"
mshark_row_expr = r"(-?\d+)\t(-?\d+)\t(-?\d+)[\r\n?|\n]"

mshark_npts_exec = re.compile(mshark_npts_expr)
mshark_fs_exec = re.compile(mshark_fs_expr)
mshark_gain_exec = re.compile(mshark_gain_expr)
mshark_conversion_exec = re.compile(mshark_conversion_expr)
mshark_row_exec = re.compile(mshark_row_expr)

#=========================
# DataWrangler | peer
#=========================
peer_direction_expr = r", (UP|VER|\d|\d\d|\d\d\d|[FGDCESHB][HLGMN][ENZ])[\r\n?|\n]"
peer_npts_expr = r"NPTS=\s*(\d+),"
peer_dt_expr = r"DT=\s*(\d*\.\d+)\s"
peer_sample_expr = r"(-?\d*\.\d+[eE][+-]?\d*)"

peer_direction_exec = re.compile(peer_direction_expr)
peer_npts_exec = re.compile(peer_npts_expr)
peer_dt_exec = re.compile(peer_dt_expr)
peer_sample_exec = re.compile(peer_sample_expr)

#=============
# ObjectIO
#=============
azimuth_expr = r"azimuth (\d+\.\d+) deg | hvsr curve \d+"
azimuth_exec = re.compile(azimuth_expr)

#===============
# HvsrGeopsy
#===============
geopsy_line_expr = r"(\d+\.\d+)\t(\d+\.\d+)\t(\d+\.\d+)\t\d+\.\d+[\r\n?|\n]"
geopsy_line_exec = re.compile(geopsy_line_expr)



#################################
# Time Series
#################################
def Fill_Gaps_Linear(
    xp_arr,
    max_gap: int = 10
):
    is_cupy = isinstance(xp_arr, getattr(cp, "ndarray", type(None)))

    if is_cupy:
        arr = cp.asarray(xp_arr)
        arr_work = arr.astype(cp.float64, copy=True)
    else:
        arr = np.asarray(xp_arr)
        arr_work = arr.astype(np.float64, copy=True)

    n = arr_work.size
    if n == 0:
        return (cp.asarray(arr_work).astype(cp.float32) if is_cupy else arr_work.astype(np.float32))

    nan_mask = (cp.isnan(arr_work) if is_cupy else np.isnan(arr_work))
    if not bool((nan_mask.any()).item() if is_cupy else nan_mask.any()):
        return (arr_work.astype(cp.float32) if is_cupy else arr_work.astype(np.float32))

    if is_cupy:
        nan_pos = cp.asnumpy(cp.where(nan_mask)[0])
        valid_idx = cp.asnumpy(cp.where(~nan_mask & cp.isfinite(arr_work))[0])
        valid_vals = cp.asnumpy(cp.asarray(arr_work)[~nan_mask & cp.isfinite(arr_work)])
    else:
        nan_pos = np.where(nan_mask)[0]
        valid_idx = np.where(~nan_mask & np.isfinite(arr_work))[0]
        valid_vals = arr_work[~nan_mask & np.isfinite(arr_work)]

    if valid_idx.size < 2:
        if is_cupy:
            out = cp.nan_to_num(arr_work, nan=0.0, posinf=0.0, neginf=0.0)
            return out.astype(cp.float32)
        else:
            out = np.nan_to_num(arr_work, nan=0.0, posinf=0.0, neginf=0.0)
            return out.astype(np.float32)

    if nan_pos.size == 0:
        return (arr_work.astype(cp.float32) if is_cupy else arr_work.astype(np.float32))
    gaps = np.split(nan_pos, np.where(np.diff(nan_pos) != 1)[0] + 1)

    fill_positions = np.zeros(n, dtype=bool)
    edge_fill_pos = []

    arr_work_host = cp.asnumpy(arr_work) if is_cupy else np.asarray(arr_work)

    for g in gaps:
        g0 = int(g[0]);
        g1 = int(g[-1]);
        length = g1 - g0 + 1
        if length <= int(max_gap):
            left = g0 - 1
            right = g1 + 1
            left_ok = (left >= 0) and (not np.isnan(arr_work_host[left])) and np.isfinite(arr_work_host[left])
            right_ok = (right < n) and (not np.isnan(arr_work_host[right])) and np.isfinite(arr_work_host[right])
            if left_ok and right_ok:
                fill_positions[g0:(g1 + 1)] = True
            elif left_ok and not right_ok:
                edge_fill_pos.append((np.arange(g0, g1 + 1), float(arr_work_host[left])))
            elif right_ok and not left_ok:
                edge_fill_pos.append((np.arange(g0, g1 + 1), float(arr_work_host[right])))
            else:
                pass
        else:
            pass

    if is_cupy:
        out_host = cp.asnumpy(arr_work)
    else:
        out_host = np.asarray(arr_work)

    if fill_positions.any():
        target_idx = np.where(fill_positions)[0]
        if is_cupy and hasattr(cp, "interp"):
            device_target = cp.asarray(target_idx)
            device_knots_x = cp.asarray(valid_idx)
            device_knots_y = cp.asarray(valid_vals)
            interp_vals = cp.interp(device_target, device_knots_x, device_knots_y)
            interp_vals_host = cp.asnumpy(interp_vals)
        else:
            interp_vals_host = np.interp(target_idx, valid_idx, valid_vals)
        out_host[target_idx] = interp_vals_host

    for pos_array, fill_value in edge_fill_pos:
        out_host[pos_array] = fill_value

    if is_cupy:
        return cp.asarray(out_host).astype(cp.float32)
    else:
        return out_host.astype(np.float32)


class Time_Series_CUDA:

    @classmethod
    def from_timeseries(
        cls,
        other: "Time_Series_CUDA"
    ) -> "Time_Series_CUDA":
        if not isinstance(other, Time_Series_CUDA):
            raise TypeError("Input must be a Time_Series_CUDA object")
        return cls(other.amplitude.copy(), other.dt_in_seconds)

    def __init__(
        self,
        amplitude: Iterable,
        dt_in_seconds: Number,
        fill_max_gap: int = 10
    ):
        try:
            arr = cp.asarray(amplitude, dtype=cp.float32)
        except Exception as exc:
            raise TypeError("'amplitude' must be convertible to numeric array") from exc

        if arr.ndim != 1:
            raise TypeError("'amplitude' must be 1-D")

        # if cp.isnan(arr).any():
        #   arr = Fill_Gaps_Linear(arr, max_gap=fill_max_gap)

        if bool(cp.isnan(arr).any().item()):
            arr = Fill_Gaps_Linear(arr, max_gap=fill_max_gap)

        self.amplitude: cp.ndarray = arr
        try:
            self.dt_in_seconds = float(dt_in_seconds)
        except Exception as exc:
            raise TypeError("'dt_in_seconds' must be numeric") from exc

    def fill_gaps(self, max_gap: int = 20) -> None:
        # if cp.isnan(self.amplitude).any():
        #    self.amplitude = Fill_Gaps_Linear(self.amplitude, max_gap=max_gap)

        if bool(cp.isnan(self.amplitude).any().item()):
            self.amplitude = Fill_Gaps_Linear(self.amplitude, max_gap=max_gap)

    #-------------
    # Helpers
    #-------------
    @property
    def n_samples(self) -> int:
        return int(self.amplitude.shape[0])

    @property
    def fs(self) -> float:
        return 1.0 / self.dt_in_seconds

    @property
    def fnyq(self) -> float:
        return 0.5 * self.fs

    def time(self) -> cp.ndarray:
        return cp.arange(self.n_samples, dtype=cp.float32) * cp.float32(self.dt_in_seconds)

    def is_similar(
        self,
        other,
        check_amplitude: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        if not isinstance(other, Time_Series_CUDA):
            return False

        if self.dt_in_seconds != other.dt_in_seconds:
            return False
        if self.n_samples != other.n_samples:
            return False

        if check_amplitude:
            try:
                res = cp.allclose(self.amplitude, other.amplitude, rtol, atol)
                return bool(res)
            except Exception:
                return np.allclose(cp.asnumpy(self.amplitude), cp.asnumpy(other.amplitude), rtol=rtol, atol=atol)
        return True

    def __eq__(
        self,
        other
    ) -> bool:
        if not isinstance(other, Time_Series_CUDA):
            return False
        if self.dt_in_seconds != other.dt_in_seconds:
            return False
        if self.n_samples != other.n_samples:
            return False
        try:
            return bool(cp.allclose(self.amplitude, other.amplitude))
        except Exception:
            return bool(np.allclose(cp.asnumpy(self.amplitude), cp.asnumpy(other.amplitude)))

    def trim(
        self,
        start_time: Number,
        end_time: Number
    ) -> None:

        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except Exception as exc:
            raise IndexError("start_time and end_time must be numeric") from exc

        if start_time < 0 or end_time < 0:
            raise IndexError("Start or end time cannot be negative.")
        if start_time >= end_time:
            raise IndexError("Start time must be less than end time.")

        max_time = (self.n_samples - 1) * self.dt_in_seconds
        if end_time > max_time:
            raise IndexError("End time exceeds time series duration.")

        tvec = self.time()

        start_idx_cp = cp.argmin(cp.abs(tvec - start_time))
        end_idx_cp = cp.argmin(cp.abs(tvec - end_time))

        start_index = int(start_idx_cp.item())
        end_index = int(end_idx_cp.item())

        if start_index > end_index:
            raise IndexError("computed start index is after end index; check start/end times.")

        self.amplitude = self.amplitude[start_index: end_index + 1]

    def detrend(
        self,
        type: str = "linear"
    ) -> None:

        if type == "constant":
            self.amplitude = self.amplitude - cp.mean(self.amplitude)
            return

        if type == "linear":
            n = self.n_samples

            x = cp.linspace(0.0, 1.0, n, dtype=cp.float32)
            A = cp.column_stack((x, cp.ones(n, dtype=cp.float32)))

            try:
                coeffs, *_ = cp.linalg.lstsq(A, self.amplitude, rcond=None)
                trend = A @ coeffs
                self.amplitude = self.amplitude - trend
            except Exception:
                xm = cp.mean(x)
                ym = cp.mean(self.amplitude)
                numer = cp.sum((x - xm) * (self.amplitude - ym))
                denom = cp.sum((x - xm) ** 2)
                slope = numer / denom
                intercept = ym - slope * xm
                trend = slope * x + intercept
                self.amplitude = self.amplitude - trend
            return

        raise NotImplementedError("Only 'linear' and 'constant' detrend supported on GPU.")


    def split(
        self,
        window_length_in_seconds: Union[float, int, Sequence[Union[float, int, str]]],
        *,
        allow_truncate: bool = True,
        pad_mode: Optional[str] = None,
        pad_value: float = 0.0,
    ) -> List["Time_Series_CUDA"]:

        #--------------------------------------------------------------
        # 1) Normalize input and extract numeric length if necessary
        #--------------------------------------------------------------
        if isinstance(window_length_in_seconds, (list, tuple)):
            if len(window_length_in_seconds) == 0:
                raise ValueError("window_length_in_seconds sequence is empty.")

            numeric_val = None
            for elem in window_length_in_seconds:
                if isinstance(elem, (int, float)):
                    numeric_val = float(elem)
                    break
                if isinstance(elem, str):
                    try:
                        numeric_val = float(elem)
                        break
                    except Exception:
                        continue
            if numeric_val is None:
                warnings.warn("window_length_in_seconds sequence contains no numeric element. "
                              "If you meant to pass a taper (e.g. 'tukey'), use a separate parameter.")
                raise TypeError("window_length_in_seconds sequence must contain a numeric element.")
            if len(window_length_in_seconds) > 1:
                warnings.warn("window_length_in_seconds is a sequence; using the first numeric element found.")
            window_length_in_seconds = numeric_val

        elif isinstance(window_length_in_seconds, str):
            raise TypeError(
                "window_length_in_seconds must be numeric. Received a string (e.g. 'tukey'). "
                "If you intended to specify a taper/window type, supply it using a separate parameter."
            )

        try:
            window_length_in_seconds = float(window_length_in_seconds)
        except Exception:
            raise TypeError("window_length_in_seconds must be numeric (or a sequence containing a numeric element).")

        #------------------------------------------
        # 2) Basic validation of dt and samples
        #------------------------------------------
        if not hasattr(self, "dt_in_seconds") or self.dt_in_seconds is None:
            raise ValueError("Time_Series has no dt_in_seconds defined.")
        dt = float(self.dt_in_seconds)
        if dt <= 0.0:
            raise ValueError("dt_in_seconds must be positive.")

        n_samples = int(self.n_samples)
        if n_samples <= 0:
            raise ValueError("Time series has no samples (n_samples <= 0).")

        #-------------------------------------------
        # 3) Compute number of samples per window
        #-------------------------------------------
        samples_per_window = int(math.floor(window_length_in_seconds / dt)) + 1

        if samples_per_window <= 1:
            raise ValueError("Window length too small relative to dt_in_seconds (samples_per_window <= 1).")

        amp = self.amplitude
        is_cupy_array = False
        if USE_CUPY:
            try:
                is_cupy_array = isinstance(amp, cp.ndarray)
            except Exception:
                is_cupy_array = False

        xp_pad = cp if is_cupy_array else np

        if samples_per_window > n_samples:
            msg = (f"Requested window length {window_length_in_seconds}s -> samples_per_window={samples_per_window} "
                   f"but record has only n_samples={n_samples} (duration={n_samples * dt}s).")
            if pad_mode is not None:
                try:
                    pad_width = samples_per_window - n_samples
                    if is_cupy_array:
                        padded = xp_pad.pad(amp, (0, pad_width), mode=pad_mode, constant_values=pad_value)
                    else:
                        padded = xp_pad.pad(amp, (0, pad_width), mode=pad_mode, constant_values=pad_value)
                    amp_used = padded
                    n_samples = amp_used.size
                except Exception as e:
                    warnings.warn(f"Padding failed ({e}). Falling back to truncate behavior.", UserWarning)
                    amp_used = amp
                    samples_per_window = n_samples
            elif allow_truncate:
                warnings.warn(msg + " Truncating window to full record length (returning single window).", UserWarning)
                samples_per_window = n_samples
                amp_used = amp
            else:
                raise ValueError("Window too long for time series (samples_per_window > n_samples).")
        else:
            amp_used = amp

        #------------------------------------------
        # 4) Compute step and number of windows
        #------------------------------------------
        step = samples_per_window - 1
        if step <= 0:
            raise ValueError("Computed step <= 0; check window_length_in_seconds and dt_in_seconds.")

        n_windows = (int(n_samples) - 1) // int(step)

        if n_windows < 1:
            if samples_per_window >= n_samples:
                seg = amp_used[0:int(samples_per_window)]
                return [Time_Series_CUDA(seg, dt)]
            else:
                raise ValueError("Window too long for time series; no windows fit.")

        #---------------------------
        # 5) Slice into windows
        #---------------------------
        windows: List[Time_Series_CUDA] = []
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + samples_per_window
            if end_idx > int(n_samples):
                end_idx = int(n_samples)
            seg = amp_used[start_idx:end_idx]
            windows.append(Time_Series_CUDA(seg, dt))

        return windows


    def window(
        self,
        type: str = "tukey",
        width: float = 0.1
    ) -> None:
        if type != "tukey":
            raise NotImplementedError("Only 'tukey' window supperted on CUDA.")
        w = tukey(self.n_samples, alpha=float(width))
        self.amplitude = self.amplitude * w


    def butterworth_filter(
        self,
        fcs_in_hz: Tuple[Union[None, float],
        Union[None, float]],
        order: int = 5
    ) -> None:
        fc_low, fc_high = fcs_in_hz
        if fc_low is None and fc_high is None:
            warnings.warn("No corner frequencies provided; no filtering performed.")
            return

        if fc_low is None and fc_high is not None:
            btype = "lowpass"
            wn = float(fc_high)
        elif fc_low is not None and fc_high is None:
            btype = "highpass"
            wn = float(fc_low)
        elif fc_low is not None and fc_high is not None:
            btype = "bandpass"
            wn = [float(fc_low), float(fc_high)]
        else:
            raise ValueError("Invalid corner frequencies.")
        sos = butter(order, wn, btype=btype, fs=self.fs, output="sos")
        self.amplitude = sosfiltfilt(sos, self.amplitude)

    #---------------------
    # Representations
    #---------------------
    def __str__(self) -> str:
        return f"CUDA Time Series with {self.n_samples} samples."

    def __repr__(self) -> str:
        return f"Time_Series_CUDA(amplitude=<cupy.ndarray shape={tuple(self.amplitude.shape)} dtype={self.amplitude.dtype}>, dt_in_seconds={self.dt_in_seconds})"



####################################
# Statistics
####################################
def Distribution_Factory(
    distribution: str,
    calculation: str = "mean"
):
    if distribution is None:
        raise NotImplementedError("distribution must be provided.")

    key = DISTRIBUTION_MAP.get(distribution.lower(), None)
    if key is None:
        raise NotImplementedError(f"distribution type {distribution} not recognized.")

    # Pre/post processors using xp
    if key == "normal":
        if calculation == "mean":
            pre = lambda values: values
            post = lambda values: values
        elif calculation == "std":
            pre = lambda values: values
            post = lambda values: values
        else:
            raise NotImplementedError(f"calculation {calculation} not supported.")
    elif key == "lognormal":
        if calculation == "mean":
            pre = lambda values: xp.log(values)
            post = lambda values: xp.exp(values)
        elif calculation == "std":
            pre = lambda values: xp.log(values)
            # For std, the original returned the transformed std unchanged for lognormal
            post = lambda values: values
        else:
            raise NotImplementedError(f"calculation {calculation} not supported.")
    else:
        raise NotImplementedError(f"distribution type {distribution} not recognized.")

    return pre, post


def Generalized_Weighted_Mean(
    distribution: str,
    values,
    weights=None,
    mean_kwargs: _t.Dict = None
):
    if mean_kwargs is None:
        mean_kwargs = {}

    xp_local = xp
    dist_lower = str(distribution).lower()

    arr = xp_local.asarray(values)

    if dist_lower.startswith("log"):
        valid_mask = xp_local.isfinite(arr) & (arr > 0)
    else:
        valid_mask = xp_local.isfinite(arr)

    count_valid = xp_local.sum(valid_mask, **mean_kwargs)

    try:
        n_valid = int(count_valid)
    except Exception:
        n_valid = int(xp_local.asnumpy(count_valid))

    if n_valid == 0:
        return xp_local.nan

    if weights is None:
        weights_arr = xp_local.ones_like(arr, dtype=xp_local.float64)
    else:
        weights_arr = xp_local.asarray(weights, dtype=xp_local.float64)
        try:
            if weights_arr.shape != arr.shape:
                weights_arr = xp_local.broadcast_to(weights_arr, arr.shape)
        except Exception:
            raise ValueError("weights cannot be broadcast to values shape")

    weights_masked = xp_local.where(valid_mask, weights_arr, 0.0)

    vals_pre = xp_local.zeros_like(arr, dtype=xp_local.float64)
    if dist_lower.startswith("log"):
        vals_pre = xp_local.where(valid_mask, xp_local.log(arr), 0.0)
    else:
        pre_fxn, _ = Distribution_Factory(distribution=distribution, calculation="mean")
        try:
            full_pre = pre_fxn(arr)
            vals_pre = xp_local.where(valid_mask, xp_local.asarray(full_pre, dtype=xp_local.float64), 0.0)
        except Exception:
            vals_pre = xp_local.zeros_like(arr, dtype=xp_local.float64)
            valid_idx = valid_mask
            vals_pre_subset = pre_fxn(arr[valid_idx])
            vals_pre[valid_idx] = xp_local.asarray(vals_pre_subset, dtype=xp_local.float64)

    num = xp_local.sum(weights_masked * vals_pre, **mean_kwargs)
    den = xp_local.sum(weights_masked, **mean_kwargs)

    try:
        den_scalar = float(den)
    except Exception:
        den_scalar = float(xp_local.asnumpy(den))

    if den_scalar == 0.0 or not np.isfinite(den_scalar):
        return xp_local.nan

    mean_pre = num / den
    _, post_fxn = Distribution_Factory(distribution=distribution, calculation="mean")
    try:
        result = post_fxn(mean_pre)
    except Exception:
        mean_host = xp_local.asnumpy(mean_pre) if USE_CUPY else np.asarray(mean_pre)
        host_result = post_fxn(mean_host)
        result = xp_local.asarray(host_result)
    return result


def Weighted_Standard_Deviation(
    distribution: str,
    values,
    weights=None,
    std_kwargs: _t.Dict = None,
    denominator: str = "nist"
):
    if std_kwargs is None:
        std_kwargs = {}

    xp_local = xp

    arr = xp_local.asarray(values)
    pre_fxn, post_fxn = Distribution_Factory(distribution=distribution, calculation="std")

    if str(distribution).lower().startswith("log"):
        valid_mask = xp_local.isfinite(arr) & (arr > 0)
    else:
        valid_mask = xp_local.isfinite(arr)

    count_valid = xp_local.sum(valid_mask, **std_kwargs)
    try:
        count_valid_scalar = int(count_valid)
    except Exception:
        count_valid_scalar = int(xp_local.asnumpy(count_valid))

    if count_valid_scalar <= 1:
        return xp_local.nan

    if weights is None:
        weights_arr = xp_local.ones_like(arr, dtype=xp_local.float64)
    else:
        weights_arr = xp_local.asarray(weights, dtype=xp_local.float64)
        try:
            if weights_arr.shape != arr.shape:
                weights_arr = xp_local.broadcast_to(weights_arr, arr.shape)
        except Exception:
            raise ValueError("weights cannot be broadcast to values shape")

    weights_masked = xp_local.where(valid_mask, weights_arr, 0.0)

    try:
        vals_pre = pre_fxn(arr)
    except Exception:
        vals_pre = xp_local.zeros_like(arr, dtype=xp_local.float64)
        vals_pre[valid_mask] = pre_fxn(arr[valid_mask])

    sum_w = xp_local.sum(weights_masked, **std_kwargs)
    try:
        sum_w_scalar = float(sum_w)
    except Exception:
        sum_w_scalar = float(xp_local.asnumpy(sum_w))

    if sum_w_scalar == 0.0 or not np.isfinite(sum_w_scalar):
        return xp_local.nan

    sum_wx = xp_local.sum(weights_masked * xp_local.where(valid_mask, vals_pre, 0.0), **std_kwargs)
    mean_pre = sum_wx / sum_w

    diffsq = xp_local.where(valid_mask, (vals_pre - mean_pre) ** 2, 0.0)
    numerator = xp_local.sum(weights_masked * diffsq, **std_kwargs)

    if denominator == "nist":
        non_nan_count = xp_local.sum(valid_mask, **std_kwargs)
        try:
            n_scalar = float(non_nan_count)
        except Exception:
            n_scalar = float(xp_local.asnumpy(non_nan_count))
        if n_scalar <= 1.0:
            return xp_local.nan
        den = (1.0 - (1.0 / n_scalar)) * sum_w
    elif denominator == "cheng":
        den = 1.0 - xp_local.sum((weights_masked) ** 2, **std_kwargs)
    else:
        raise NotImplementedError("denominator method not recognized")

    try:
        den_scalar = float(den)
    except Exception:
        den_scalar = float(xp_local.asnumpy(den))

    if (not np.isfinite(den_scalar)) or (den_scalar <= 0.0):
        return xp_local.nan

    std_raw = xp_local.sqrt(numerator / den)
    try:
        result = post_fxn(std_raw)
    except Exception:
        std_host = xp_local.asnumpy(std_raw) if USE_CUPY else np.asarray(std_raw)
        host_result = post_fxn(std_host)
        result = xp_local.asarray(host_result)
    return result


def Standard_Score_Shift(
    n: float,
    distribution: str,
    mean,
    std
):
    xp_local = xp
    key = DISTRIBUTION_MAP.get(distribution, distribution).lower()

    mean_arr = xp_local.asarray(mean)
    std_arr = xp_local.asarray(std)

    try:
        mean_b, std_b = xp_local.broadcast_to(mean_arr, xp_local.broadcast_shapes(mean_arr.shape, std_arr.shape)), \
                        xp_local.broadcast_to(std_arr, xp_local.broadcast_shapes(mean_arr.shape, std_arr.shape))
    except Exception:
        mean_b = xp_local.asarray(mean_arr)
        std_b = xp_local.asarray(std_arr)

    if key == "normal":
        out = mean_arr + (n * std_arr)

    elif key == "lognormal":
        valid_mean = xp_local.isfinite(mean_arr) & (mean_arr > 0)
        valid_std = xp_local.isfinite(std_arr)
        valid = valid_mean & valid_std

        safe_log_mean = xp_local.where(valid, xp_local.log(mean_arr), 0.0)
        out_pre = safe_log_mean + (n * std_arr)
        out = xp_local.where(valid, xp_local.exp(out_pre), xp_local.nan)

    else:
        raise NotImplementedError(f"distribution type {distribution} not recognized.")

    try:
        is_scalar_like = (getattr(mean_arr, "size", 1) == 1) and (getattr(std_arr, "size", 1) == 1)
    except Exception:
        is_scalar_like = False

    if is_scalar_like:
        try:
            return float(out.item())
        except Exception:
            return float(xp_local.asnumpy(out).item())
    return out


def Flatten_List(unflattened_list: _t.Iterable[_t.Iterable[_t.Any]]):
    flattened_list = []
    for _list in unflattened_list:
        flattened_list.extend(_list)
    return flattened_list


__all__ = [
    "Distribution_Factory",
    "Generalized_Weighted_Mean",
    "Weighted_Standard_Deviation",
    "Standard_Score_Shift",
    "Flatten_List",
    "USE_CUPY",
]



####################################
# Smoothing
####################################
__all__ = [
    "konno_and_ohmachi",
    "parzen",
    "savitzky_and_golay",
    "linear_rectangular",
    "log_rectangular",
    "linear_triangular",
    "log_triangular",
    "SMOOTHING_OPERATORS",
    "USE_CUPY",
]


#================
# Helpers
#================
def as_1d_float_array(
    a,
    dtype=None
):
    if dtype is None:
        dtype = xp.float32 if USE_CUPY else xp.float64
    arr = xp.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D")
    return arr.astype(dtype)


def safe_divide_number_by_denominator(num, den):
    den = xp.asarray(den)
    zero_mask = den == 0
    den_safe = den.copy()
    if den_safe.ndim == 0:
        den_safe = den_safe.reshape((1,))
        zero_mask = zero_mask.reshape((1,))
    den_safe = den_safe.copy()
    den_safe[zero_mask] = 1.0
    result = num / den_safe[None, :]
    if xp.any(zero_mask):
        result[:, zero_mask] = 0.0
    return result


def Matrix_Multiplication(a, b):
    try:
        return a @ b
    except Exception as ex1:
        if USE_CUPY:
            try:
                a32 = a.astype(xp.float32)
                b32 = b.astype(xp.float32)
                return a32 @ b32
            except Exception:
                a_host = cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
                b_host = cp.asnumpy(b) if isinstance(b, cp.ndarray) else np.asarray(b)
                res_host = a_host @ b_host
                try:
                    return cp.asarray(res_host)
                except Exception:
                    return res_host
        else:
            raise ex1


#====================
# Konno and Ohmachi
#====================
def konno_and_ohmachi(
    frequencies,
    spectrum,
    fcs,
    bandwidth=40.0
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec_dtype = xp.float32 if USE_CUPY else xp.float64
    spec = xp.asarray(spectrum, dtype=spec_dtype)
    if spec.ndim != 2:
        raise ValueError("spectrum must be 2D (nspectrum, nfrequency)")
    nspec, nfreq = spec.shape
    if frequencies.size != nfreq:
        raise ValueError("frequencies length must match spectrum's frequency axis")

    n = 3.0
    upper_limit = xp.power(10.0, +n / bandwidth)
    lower_limit = xp.power(10.0, -n / bandwidth)

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_on_fc = f / Fc

    mask_invalid = (frequencies[:, None] < 1e-6) | (f_on_fc > upper_limit) | (f_on_fc < lower_limit)

    ctx = xp.errstate(divide="ignore", invalid="ignore") if hasattr(xp, "errstate") else nullcontext()
    with ctx:
        x = bandwidth * xp.log10(f_on_fc)

    sinx_over_x = xp.where(x == 0, 1.0, xp.sin(x) / x)
    window = sinx_over_x ** 4

    eq_mask = xp.isclose(f, Fc, atol=1e-12)
    window = xp.where(eq_mask, 1.0, window)

    window = xp.where(mask_invalid, 0.0, window)

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)

    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


#=============
# Parzen
#=============
def parzen(
    frequencies,
    spectrum,
    fcs,
    bandwidth=0.5
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec_dtype = xp.float32 if USE_CUPY else xp.float64
    spec = xp.asarray(spectrum, dtype=spec_dtype)
    if spec.ndim != 2:
        raise ValueError("spectrum must be 2-D (nspec, nfrequency)")
    nspec, nfreq = spec.shape
    if frequencies.size != nfreq:
        raise ValueError("frequencies length must match spectrum's frequency axis")

    a = (math.pi * 280.0) / (2.0 * 151.0)
    upper_limit = xp.sqrt(6.0) * a / bandwidth
    lower_limit = -upper_limit

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_minus_fc = f - Fc

    mask_invalid = (frequencies[:, None] < 1e-6) | (f_minus_fc > upper_limit) | (f_minus_fc < lower_limit)

    x = a * f_minus_fc / float(bandwidth)
    ctx = xp.errstate(divide="ignore", invalid="ignore") if hasattr(xp, "errstate") else nullcontext()
    with ctx:
        sinx_over_x = xp.where(x == 0, 1.0, xp.sin(x) / x)
    window = sinx_over_x ** 4

    eq_mask = xp.isclose(f, Fc, atol=1e-12)
    window = xp.where(eq_mask, 1.0, window)

    window = xp.where(mask_invalid, 0.0, window)

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)

    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


#==========================
# Savitzky and Golay
#==========================
def savitzky_and_golay(
    frequencies,
    spectrum,
    fcs,
    bandwidth=9,
    tol=None
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    spec = xp.asarray(spectrum, dtype=freq_dtype)
    fcs = xp.asarray(fcs, dtype=freq_dtype)

    m = int(bandwidth)
    if m % 2 != 1:
        raise ValueError("bandwidth for savitzky_and_golay must be an odd integer.")

    nterms = ((m - 1) // 2) + 1
    coefficients = xp.empty((nterms,), dtype=freq_dtype)
    for idx, i in enumerate(range(-(nterms - 1), 1)):
        coefficients[idx] = ((3 * m * m - 7 - 20 * abs(i * i)) / 4.0)
    normalization_coefficient = (m * (m * m - 4) / 3.0)

    diff = xp.diff(frequencies)
    if tol is None:
        if frequencies.dtype == xp.float32:
            rtol, atol = 1e-4, 1e-6
        else:
            rtol, atol = 1e-6, 1e-8
    else:
        rtol, atol = tol

    if not xp.allclose(diff, diff[0], rtol=rtol, atol=atol):
        raise ValueError("For savitzky_and_golay frequency samples of input data must be linearly spaced.")

    df = float(xp.mean(diff))
    if df == 0.0:
        raise ValueError("Frequency increment is zero (invalid frequency axis).")

    nfcs = xp.rint((fcs - float(xp.min(frequencies))) / df).astype(xp.int64)

    return _savitzky_and_golay_gpu(spec, nfcs, coefficients, normalization_coefficient)


#====================================
# Rectangular/triangular windows
#====================================
def linear_rectangular(
    frequencies,
    spectrum,
    fcs,
    bandwidth=0.5
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec = xp.asarray(spectrum, dtype=freq_dtype)

    nspec, nfreq = spec.shape
    if frequencies.size != nfreq:
        raise ValueError("frequencies length must match spectrum's frequency axis")

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_minus_fc = f - Fc

    mask_invalid = (frequencies[:, None] < 1e-6) | (xp.abs(f_minus_fc) > (bandwidth / 2.0))
    window = xp.where(mask_invalid, 0.0, 1.0)

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)
    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


def log_rectangular(
    frequencies,
    spectrum,
    fcs,
    bandwidth=0.05
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec = xp.asarray(spectrum, dtype=freq_dtype)

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_on_fc = f / Fc

    lower_limit = xp.power(10.0, -bandwidth / 2.0)
    upper_limit = xp.power(10.0, +bandwidth / 2.0)

    mask_invalid = (frequencies[:, None] < 1e-6) | (f_on_fc < lower_limit) | (f_on_fc > upper_limit)
    window = xp.where(mask_invalid, 0.0, 1.0)

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)
    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


def linear_triangular(
    frequencies,
    spectrum,
    fcs,
    bandwidth=0.5
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec = xp.asarray(spectrum, dtype=freq_dtype)

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_minus_fc = f - Fc

    mask_invalid = (frequencies[:, None] < 1e-6) | (xp.abs(f_minus_fc) > (bandwidth / 2.0))
    window = xp.where(mask_invalid, 0.0, 1.0 - xp.abs(f_minus_fc) * (2.0 / bandwidth))

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)
    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


def log_triangular(
    frequencies,
    spectrum,
    fcs,
    bandwidth=0.05
):
    freq_dtype = xp.float32 if USE_CUPY else xp.float64
    frequencies = as_1d_float_array(frequencies, dtype=freq_dtype)
    fcs = as_1d_float_array(fcs, dtype=freq_dtype)
    spec = xp.asarray(spectrum, dtype=freq_dtype)

    f = frequencies[:, None]
    Fc = fcs[None, :]
    f_on_fc = f / Fc

    lower_limit = xp.power(10.0, -bandwidth / 2.0)
    upper_limit = xp.power(10.0, +bandwidth / 2.0)

    mask_invalid = (frequencies[:, None] < 1e-6) | (f_on_fc < lower_limit) | (f_on_fc > upper_limit)
    window = xp.where(mask_invalid, 0.0, 1.0 - xp.abs(xp.log10(f_on_fc)) * (2.0 / bandwidth))

    if window.dtype != spec.dtype:
        window = window.astype(spec.dtype)

    sumproduct = Matrix_Multiplication(spec, window)
    sumwindow = xp.sum(window, axis=0)
    smoothed = safe_divide_number_by_denominator(sumproduct, sumwindow)
    out_dtype = xp.float64 if not USE_CUPY else xp.float32
    return smoothed.astype(out_dtype)


# Mapping
SMOOTHING_OPERATORS = {
    "konno_and_ohmachi": konno_and_ohmachi,
    "parzen": parzen,
    "savitzky_and_golay": savitzky_and_golay,
    "linear_rectangular": linear_rectangular,
    "log_rectangular": log_rectangular,
    "linear_triangular": linear_triangular,
    "log_triangular": log_triangular,
}



####################################
# Settings
####################################
__all__ = [
    "HvsrPreProcessingSettings",
    "PsdPreProcessingSettings",
    "PsdProcessingSettings",
    "HvsrTraditionalProcessingSettings",
    "HvsrTraditionalSingleAzimuthProcessingSettings",
    "HvsrTraditionalRotDppProcessingSettings",
    "HvsrAzimuthalProcessingSettings",
    "HvsrDiffuseFieldProcessingSettings",
]


#=============================
# Helper for serialization
#=============================
def _sanitize_numeric_for_json(obj):
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        return [_sanitize_numeric_for_json(v) for v in obj]

    if isinstance(obj, (np.generic,)):
        py = obj.item()
        return _sanitize_numeric_for_json(py)

    if isinstance(obj, float):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, str)):
        return obj

    try:
        return str(obj)
    except Exception:
        return None


def _to_serializable(entry):
    if isinstance(entry, dict):
        return {str(k): _to_serializable(v) for k, v in entry.items()}

    if cp is not None and isinstance(entry, cp.ndarray):
        try:
            arr = cp.asnumpy(entry)
            return _sanitize_numeric_for_json(arr.tolist())
        except Exception:
            try:
                lst = entry.tolist()
                return _sanitize_numeric_for_json(lst)
            except Exception:
                return str(entry)

    if isinstance(entry, np.ndarray):
        try:
            return _sanitize_numeric_for_json(entry.tolist())
        except Exception:
            lst = np.asarray(entry).tolist()
            return _sanitize_numeric_for_json(lst)

    if hasattr(entry, "tolist") and not isinstance(entry, (str, bytes)):
        try:
            lst = entry.tolist()
            return _to_serializable(lst)
        except Exception:
            try:
                arr = np.asarray(entry)
                return _sanitize_numeric_for_json(arr.tolist())
            except Exception:
                pass

    if isinstance(entry, (list, tuple)):
        return [_to_serializable(x) for x in entry]

    if isinstance(entry, (np.generic,)):
        try:
            py = entry.item()
            return _to_serializable(py)
        except Exception:
            return str(entry)

    if isinstance(entry, float):
        return None if not np.isfinite(entry) else float(entry)
    if isinstance(entry, (int, bool, str)) or entry is None:
        return entry

    try:
        return str(entry)
    except Exception:
        return None


def geomspace(
    start,
    stop,
    num=50,
    dtype=None
):
    try:
        s = float(start)
        t = float(stop)
    except Exception:
        raise ValueError("start and stop must be numeric")

    if s <= 0 or t <= 0:
        raise ValueError("geomspace start and stop must be strictly positive")

    try:
        if hasattr(xp, "geomspace"):
            try:
                return xp.geomspace(start, stop, num=num, dtype=dtype)
            except TypeError:
                return xp.geomspace(start, stop, num=num)
    except Exception:
        pass

    np_arr = np.geomspace(s, t, num=num)
    try:
        return xp.array(np_arr, dtype=dtype) if dtype is not None else xp.array(np_arr)
    except Exception:
        return np_arr


class Settings(ABC):
    def __init__(self, hvsrpy_version=__version__):
        self.attrs = ["hvsrpy_version"]
        self.hvsrpy_version = hvsrpy_version

    @property
    def attr_dict(self):
        attr_dict = {}
        for name in self.attrs:
            attr = getattr(self, name)
            attr_dict[name] = _to_serializable(attr)
        return attr_dict

    def save(self, fname):
        with open(fname, "w") as f:
            json.dump(self.attr_dict, f)

    def load(self, fname):
        with open(fname, "r") as f:
            attr_dict = json.load(f)

        for key, value in attr_dict.items():
            orig = getattr(self, key, None)
            if orig is not None and hasattr(orig, "shape"):
                try:
                    setattr(self, key, xp.asarray(value))
                except Exception:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

    def psummary(self):
        for key, value in self.attr_dict.items():
            if isinstance(value, dict):
                print(f"{key: <40} :")
                for key2, value2 in value.items():
                    if len(str(value2)) > 40:
                        value2 = f"{str(value2)[0:20]} ... {str(value2)[-20:]}"
                    print(f"     {key2: <35} : {value2}")
            else:
                print(f"{key: <40} : {value}")

    def __eq__(self, other):
        if not isinstance(other, Settings):
            return False
        return self.attr_dict == other.attr_dict

    def __str__(self):
        return f"{type(self).__name__} with {len(self.attrs)} attributes."

    def __repr__(self):
        kwargs = ", ".join([f"{k}={v}" for k, v in self.attr_dict.items()])
        return f"{type(self).__name__}({kwargs})"


#====================================
# PreProcessingSettings classes
#====================================
class PreProcessingSettings(Settings):

    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["orient_to_degrees_from_north",
                           "filter_corner_frequencies_in_hz",
                           "window_length_in_seconds",
                           "detrend",
                           "ignore_dissimilar_time_step_warning",
                           ])
        self.orient_to_degrees_from_north = orient_to_degrees_from_north
        self.filter_corner_frequencies_in_hz = filter_corner_frequencies_in_hz
        self.window_length_in_seconds = window_length_in_seconds
        self.detrend = detrend
        self.ignore_dissimilar_time_step_warning = ignore_dissimilar_time_step_warning


class HvsrPreProcessingSettings(PreProcessingSettings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 preprocessing_method="hvsr",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         orient_to_degrees_from_north=orient_to_degrees_from_north,
                         filter_corner_frequencies_in_hz=filter_corner_frequencies_in_hz,
                         window_length_in_seconds=window_length_in_seconds,
                         detrend=detrend,
                         ignore_dissimilar_time_step_warning=ignore_dissimilar_time_step_warning)
        self.attrs.extend(["preprocessing_method"])
        self.preprocessing_method = preprocessing_method


class PsdPreProcessingSettings(PreProcessingSettings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 orient_to_degrees_from_north=0.,
                 filter_corner_frequencies_in_hz=[None, None],
                 window_length_in_seconds=60.,
                 detrend="linear",
                 ignore_dissimilar_time_step_warning=False,
                 window_type_and_width=["tukey", 0.1],
                 fft_settings=None,
                 instrument_transfer_function=None,
                 differentiate=False,
                 preprocessing_method="psd",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         orient_to_degrees_from_north=orient_to_degrees_from_north,
                         filter_corner_frequencies_in_hz=filter_corner_frequencies_in_hz,
                         window_length_in_seconds=window_length_in_seconds,
                         detrend=detrend,
                         ignore_dissimilar_time_step_warning=ignore_dissimilar_time_step_warning)
        self.attrs.extend(["window_type_and_width",
                           "fft_settings",
                           "instrument_transfer_function",
                           "differentiate",
                           "preprocessing_method",
                           ])
        self.window_type_and_width = window_type_and_width
        self.fft_settings = fft_settings
        self.instrument_transfer_function = instrument_transfer_function
        self.differentiate = differentiate
        self.preprocessing_method = preprocessing_method


#==================================
# Processing settings classes
#==================================
class PsdProcessingSettings(Settings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="keeping_majority_time_step",
                 processing_method="psd",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["window_type_and_width",
                           "smoothing",
                           "fft_settings",
                           "handle_dissimilar_time_steps_by",
                           "processing_method"
                           ])
        self.window_type_and_width = window_type_and_width
        self.smoothing = dict(smoothing)
        try:
            cf = self.smoothing.get("center_frequencies_in_hz", None)
            if cf is not None and not hasattr(cf, "shape"):
                self.smoothing["center_frequencies_in_hz"] = xp.asarray(cf)
        except Exception:
            pass
        self.fft_settings = fft_settings
        self.handle_dissimilar_time_steps_by = handle_dissimilar_time_steps_by
        self.processing_method = processing_method


class HvsrProcessingSettings(Settings):
    def __init__(self,
                 hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version)
        self.attrs.extend(["window_type_and_width",
                           "smoothing",
                           "fft_settings",
                           "handle_dissimilar_time_steps_by",
                           ])
        self.window_type_and_width = window_type_and_width
        self.smoothing = dict(smoothing)
        try:
            cf = self.smoothing.get("center_frequencies_in_hz", None)
            if cf is not None and not hasattr(cf, "shape"):
                self.smoothing["center_frequencies_in_hz"] = xp.asarray(cf)
        except Exception:
            pass
        self.fft_settings = fft_settings
        self.handle_dissimilar_time_steps_by = handle_dissimilar_time_steps_by


class HvsrTraditionalProcessingSettingsBase(HvsrProcessingSettings):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 fft_settings=None,
                 processing_method="traditional",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings)
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method


class HvsrTraditionalProcessingSettings(HvsrTraditionalProcessingSettingsBase):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="traditional",
                 method_to_combine_horizontals="geometric_mean",
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals"])
        self.method_to_combine_horizontals = method_to_combine_horizontals


class HvsrTraditionalSingleAzimuthProcessingSettings(HvsrTraditionalProcessingSettingsBase):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 fft_settings=None,
                 processing_method="traditional",
                 method_to_combine_horizontals="single_azimuth",
                 azimuth_in_degrees=20.,
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method)
        self.attrs.extend(["method_to_combine_horizontals",
                           "azimuth_in_degrees",
                           ])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.azimuth_in_degrees = azimuth_in_degrees


class HvsrTraditionalRotDppProcessingSettings(HvsrTraditionalProcessingSettingsBase):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="traditional",
                 method_to_combine_horizontals="rotdpp",
                 ppth_percentile_for_rotdpp_computation=50.,
                 azimuths_in_degrees=xp.arange(0, 180, 5)
                 ):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         fft_settings=fft_settings,
                         processing_method=processing_method,
                         )
        self.attrs.extend(["method_to_combine_horizontals",
                           "ppth_percentile_for_rotdpp_computation",
                           "azimuths_in_degrees"])
        self.method_to_combine_horizontals = method_to_combine_horizontals
        self.ppth_percentile_for_rotdpp_computation = ppth_percentile_for_rotdpp_computation
        try:
            self.azimuths_in_degrees = xp.asarray(azimuths_in_degrees)
        except Exception:
            self.azimuths_in_degrees = azimuths_in_degrees


class HvsrAzimuthalProcessingSettings(HvsrProcessingSettings):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="frequency_domain_resampling",
                 processing_method="azimuthal",
                 azimuths_in_degrees=xp.arange(0, 180, 5)):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         fft_settings=fft_settings,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         )
        self.attrs.extend(["processing_method",
                           "azimuths_in_degrees"])
        self.processing_method = processing_method
        try:
            self.azimuths_in_degrees = xp.asarray(azimuths_in_degrees)
        except Exception:
            self.azimuths_in_degrees = azimuths_in_degrees


class HvsrDiffuseFieldProcessingSettings(HvsrProcessingSettings):
    def __init__(self, hvsrpy_version=__version__,
                 window_type_and_width=["tukey", 0.1],
                 smoothing=dict(operator="konno_and_ohmachi",
                                bandwidth=40,
                                center_frequencies_in_hz=geomspace(0.1, 50, 200)),
                 fft_settings=None,
                 handle_dissimilar_time_steps_by="keeping_majority_time_step",
                 processing_method="diffuse_field"):
        super().__init__(hvsrpy_version=hvsrpy_version,
                         window_type_and_width=window_type_and_width,
                         smoothing=smoothing,
                         fft_settings=fft_settings,
                         handle_dissimilar_time_steps_by=handle_dissimilar_time_steps_by,
                         )
        self.attrs.extend(["processing_method"])
        self.processing_method = processing_method



####################################
# HVSR Curve
####################################
class HvsrCurve:

    @staticmethod
    def _to_host_array(arr):
        if USE_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)


    def __init__(
        self,
        frequency,
        amplitude,
        meta: Optional[dict] = None
    ):

        self.frequency = self._check_input(frequency, "frequency")
        self.amplitude = self._check_input(amplitude, "amplitude")
        if self.frequency.size != self.amplitude.size:
            msg = f"Length of amplitude {self.amplitude.size} and length of frequency {self.frequency.size} must agree."
            raise ValueError(msg)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()
        self._search_range_in_hz = None
        self._find_peaks_kwargs = None
        self.peak_frequency: Optional[float] = None
        self.peak_amplitude: Optional[float] = None
        self.update_peaks_bounded()


    @staticmethod
    def _check_input(
        value,
        name: str
    ):
        #--------------------------------------------------------------------------------------------------
        # 1) If value is already a backend array (xp.ndarray), avoid implicit conversion where possible.
        #--------------------------------------------------------------------------------------------------
        is_xp_array = False
        try:
            is_xp_array = isinstance(value, xp.ndarray)
        except Exception:
            is_xp_array = False

        if USE_CUPY and is_xp_array:
            try:
                arr = xp.asarray(value, dtype=xp.float64, copy=False)
            except Exception:
                try:
                    host_arr = np.asarray(value.get(), dtype=np.double)
                    arr = xp.asarray(host_arr, dtype=xp.float64)
                except Exception as e:
                    raise TypeError(f"{name} must be castable to array of doubles.") from e
        else:
            try:
                host_arr = np.asarray(value, dtype=np.double)
            except Exception:
                raise TypeError(f"{name} must be castable to array of doubles, not {type(value)}.")
            try:
                arr = xp.asarray(host_arr, dtype=xp.float64)
            except Exception as e:
                try:
                    arr = xp.asarray(np.asarray(host_arr, dtype=np.double), dtype=xp.float64)
                except Exception as e2:
                    raise TypeError(f"{name} must be castable to array of doubles.") from e2

        #---------------------------------------
        # 2) Detect NaN and negative values.
        #---------------------------------------
        try:
            nan_count = int(xp.count_nonzero(xp.isnan(arr)).item()) if USE_CUPY else int(xp.count_nonzero(xp.isnan(arr)))
            neg_count = int(xp.count_nonzero(arr < 0).item()) if USE_CUPY else int(xp.count_nonzero(arr < 0))
        except Exception:
            need_get = USE_CUPY and hasattr(arr, "get")
            host_view = np.asarray(arr.get() if need_get else arr, dtype=np.double)
            if np.isnan(host_view).all():
                raise ValueError(f"{name} may not contain nan (all values are NaN).")
            if np.isnan(host_view).any():
                idx = np.arange(host_view.size)
                finite_mask = np.isfinite(host_view)
                if not finite_mask.any():
                    raise ValueError(f"{name} may not contain nan (no finite values to impute).")
                interp_values = host_view.copy()
                interp_values[~finite_mask] = np.interp(
                    idx[~finite_mask], idx[finite_mask], host_view[finite_mask]
                )
                if np.isnan(interp_values).any():
                    good_idx = np.where(finite_mask)[0]
                    first_good, last_good = good_idx[0], good_idx[-1]
                    interp_values[:first_good] = host_view[first_good]
                    interp_values[last_good + 1 :] = host_view[last_good]

                try:
                    arr = xp.asarray(interp_values, dtype=xp.float64)
                except Exception:
                    arr = xp.asarray(np.asarray(interp_values, dtype=np.double), dtype=xp.float64)

                try:
                    neg_count = int(xp.count_nonzero(arr < 0).item()) if USE_CUPY else int(xp.count_nonzero(arr < 0))
                except Exception:
                    host_view2 = np.asarray(arr.get() if (USE_CUPY and hasattr(arr, "get")) else arr, dtype=np.double)
                    if (host_view2 < 0).any():
                        raise ValueError(f"{name} must be >= 0.")

            else:
                if (host_view < 0).any():
                    raise ValueError(f"{name} must be >= 0.")
                return arr

        else:
            if nan_count > 0:
                need_get = USE_CUPY and hasattr(arr, "get")
                host_view = np.asarray(arr.get() if need_get else arr, dtype=np.double)
                if np.isnan(host_view).all():
                    raise ValueError(f"{name} may not contain nan (all values are NaN).")
                if np.isnan(host_view).any():
                    idx = np.arange(host_view.size)
                    finite_mask = np.isfinite(host_view)
                    if not finite_mask.any():
                        raise ValueError(f"{name} may not contain NaN (no finite values to impute).")
                    interp_values = host_view.copy()
                    interp_values[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], host_view[finite_mask])
                    if np.isnan(interp_values).any():
                        good_idx = np.where(finite_mask)[0]
                        first_good, last_good = good_idx[0], good_idx[-1]
                        interp_values[:first_good] = host_view[first_good]
                        interp_values[last_good + 1 :] = host_view[last_good]
                    try:
                        arr = xp.asarray(interp_values, dtype=xp.float64)
                    except Exception:
                        arr = xp.asarray(np.asarray(interp_values, dtype=np.double), dtype=xp.float64)
                    try:
                        neg_count = int(xp.count_nonzero(arr < 0).item()) if USE_CUPY else int(xp.count_nonzero(arr < 0))
                    except Exception:
                        host_view2 = np.asarray(arr.get() if (USE_CUPY and hasattr(arr, "get")) else arr, dtype=np.double)
                        if (host_view2 < 0).any():
                            raise ValueError(f"{name} must be >= 0.")
                else:
                    pass

            if neg_count > 0:
                raise ValueError(f"{name} must be >= 0.")
        return arr


    @staticmethod
    def _find_peak_unbounded(
        frequency,
        amplitude,
        find_peaks_kwargs: Optional[dict] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        if find_peaks_kwargs is None:
            find_peaks_kwargs = {}
        freq_host = HvsrCurve._to_host_array(frequency).astype(float)
        amp_host = HvsrCurve._to_host_array(amplitude).astype(float)
        if not np.any(np.isfinite(amp_host)):
            return (None, None)
        if not np.any(amp_host > 0):
            return (None, None)
        amp_clean = np.copy(amp_host)
        nan_mask = ~np.isfinite(amp_clean)
        if nan_mask.any():
            amp_clean[nan_mask] = -np.inf
        potential_peak_indices, _ = find_peaks(amp_clean, **find_peaks_kwargs)
        if len(potential_peak_indices) == 0:
            return (None, None)
        potential_peak_amplitudes = amp_host[potential_peak_indices]
        valid_idx_mask = np.isfinite(potential_peak_amplitudes)
        if not np.any(valid_idx_mask):
            return (None, None)
        sub_idx = int(np.nanargmax(potential_peak_amplitudes))
        peak_idx = int(potential_peak_indices[sub_idx])
        peak_freq = float(freq_host[peak_idx])
        peak_amp = float(amp_host[peak_idx])
        return (peak_freq, peak_amp)


    @staticmethod
    def _search_range_to_index_range(
        frequency,
        search_range_in_hz: Tuple[Optional[float], Optional[float]]
    ) -> Tuple[int, int]:
        f_low, f_high = search_range_in_hz
        freq_arr = xp.asarray(frequency)
        if f_low is None:
            f_low_idx = 0
        else:
            f_low_idx = int(xp.argmin(xp.abs(freq_arr - float(f_low))).item())
        if f_high is None:
            f_high_idx = int(freq_arr.size)
        else:
            f_high_idx = int(xp.argmin(xp.abs(freq_arr - float(f_high))).item())
        return (f_low_idx, f_high_idx)


    @staticmethod
    def _find_peak_bounded(
        frequency,
        amplitude,
        search_range_in_hz=(None, None),
        find_peaks_kwargs: Optional[dict] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        f_low_idx, f_high_idx = HvsrCurve._search_range_to_index_range(frequency, search_range_in_hz)
        freq_slice = xp.asarray(frequency)[f_low_idx:f_high_idx]
        amp_slice = xp.asarray(amplitude)[f_low_idx:f_high_idx]
        return HvsrCurve._find_peak_unbounded(freq_slice, amp_slice, find_peaks_kwargs=find_peaks_kwargs)


    def update_peaks_bounded(
        self,
        search_range_in_hz=(None, None),
        find_peaks_kwargs: Optional[dict] = None
    ) -> None:
        if (search_range_in_hz == self._search_range_in_hz) and (find_peaks_kwargs == self._find_peaks_kwargs):
            return
        self._search_range_in_hz = tuple(search_range_in_hz)
        self.meta["search_range_in_hz"] = self._search_range_in_hz
        if find_peaks_kwargs is None:
            self._find_peaks_kwargs = {}
            self.meta["find_peaks_kwargs"] = None
        else:
            self._find_peaks_kwargs = dict(find_peaks_kwargs)
            self.meta["find_peaks_kwargs"] = dict(find_peaks_kwargs)
        frq, amp = self._find_peak_bounded(self.frequency, self.amplitude,
                                           search_range_in_hz=search_range_in_hz,
                                           find_peaks_kwargs=find_peaks_kwargs)
        if frq is None or amp is None or (not np.isfinite(frq)) or (not np.isfinite(amp)):
            logger.info("No peak found in HVSR curve within search range.")
            self.peak_frequency = None
            self.peak_amplitude = None
            return
        self.peak_frequency = float(frq)
        self.peak_amplitude = float(amp)


    def is_similar(
        self,
        other: "HvsrCurve",
        atol: float = 1e-9,
        rtol: float = 0.0
    ) -> bool:

        if not isinstance(other, HvsrCurve):
            return False

        if self.frequency.size != other.frequency.size:
            return False

        f1 = np.asarray(self._to_host_array(self.frequency))
        f2 = np.asarray(self._to_host_array(other.frequency))
        if not np.allclose(f1, f2, atol=atol, rtol=rtol):
            return False

        return True


    def __eq__(
        self,
        other: object
    ) -> bool:

        if not isinstance(other, HvsrCurve):
            return False

        if not self.is_similar(other):
            return False

        a1 = np.asarray(self._to_host_array(self.amplitude))
        a2 = np.asarray(self._to_host_array(other.amplitude))
        if not np.allclose(a1, a2):
            return False

        pf1 = None if self.peak_frequency is None else float(self.peak_frequency)
        pf2 = None if other.peak_frequency is None else float(other.peak_frequency)
        pa1 = None if self.peak_amplitude is None else float(self.peak_amplitude)
        pa2 = None if other.peak_amplitude is None else float(other.peak_amplitude)

        if pf1 is None and pf2 is None:
            pass
        elif pf1 is None or pf2 is None:
            return False
        else:
            if not np.isclose(pf1, pf2):
                return False

        if pa1 is None and pa2 is None:
            pass
        elif pa1 is None or pa2 is None:
            return False
        else:
            if not np.isclose(pa1, pa2):
                return False

        return True



####################################
# Sesame (Various HVSR Utilities)
####################################
def _to_numpy(x):
    if USE_CUPY:
        try:
            return cp.asnumpy(x)
        except Exception:
            return np.asarray(x)
    else:
        return np.asarray(x)

def _safe_nanmin(xp_arr):
    try:
        return xp.nanmin(xp_arr)
    except Exception:
        return np.nanmin(_to_numpy(xp_arr))

def _safe_nanmax(xp_arr):
    try:
        return xp.nanmax(xp_arr)
    except Exception:
        return np.nanmax(_to_numpy(xp_arr))

def _safe_nanargmax(xp_arr):
    try:
        return int(xp.nanargmax(xp_arr))
    except Exception:
        arr = _to_numpy(xp_arr)
        try:
            return int(np.nanargmax(arr))
        except Exception:
            return -1

def _safe_isfinite_scalar(x):
    try:
        return bool(xp.isfinite(x))
    except Exception:
        try:
            return np.isfinite(_to_numpy(x)).item()
        except Exception:
            return False

def pass_fail(value):
    try:
        v = float(value)
    except Exception:
        v = float(np.asarray(value))
    return colored("Pass", "green") if v > 0 else colored("Fail", "red")


def peak_index(curve):
    arr = xp.asarray(curve)
    try:
        if USE_CUPY and isinstance(arr, cp.ndarray):
            curve_host = cp.asnumpy(arr)
        else:
            curve_host = np.asarray(arr)
        peak_idx, _ = HvsrCurve._find_peak_unbounded(np.arange(len(curve_host)), curve_host)
        if peak_idx is not None:
            return int(peak_idx)
    except Exception:
        pass
    idx = _safe_nanargmax(arr)
    if idx < 0:
        raise ValueError("Unable to determine peak index: curve contains no finite values.")
    return int(idx)


def trim_curve(
    search_range_in_hz,
    frequency,
    mean_curve,
    std_curve,
    verbose=0
):
    freq_x = xp.asarray(frequency)
    mean_x = xp.asarray(mean_curve)
    std_x = xp.asarray(std_curve)
    if freq_x.size == 0:
        if verbose > 0:
            print("Warning: frequency array is empty; returning empty arrays.")
        return freq_x, mean_x, std_x
    try:
        freq_min = float(_safe_nanmin(freq_x))
        freq_max = float(_safe_nanmax(freq_x))
    except Exception:
        freq_min = float(np.nan)
        freq_max = float(np.nan)
    low_limit = search_range_in_hz[0] if search_range_in_hz[0] is not None else freq_min
    upp_limit = search_range_in_hz[1] if search_range_in_hz[1] is not None else freq_max
    mask = (freq_x >= low_limit) & (freq_x <= upp_limit)
    if not xp.any(mask):
        if verbose > 0:
            print("Warning: No data within the specified range; returning full arrays.")
        return freq_x, mean_x, std_x
    trimmed_freq = freq_x[mask]
    trimmed_mean = mean_x[mask]
    trimmed_std = std_x[mask]
    if verbose > 0:
        try:
            print(f"Considering only frequencies bertween {float(low_limit):.03f} and {float(upp_limit):.03f} Hz.")
        except Exception:
            print("Considering specified frequency range (could not format values).")

    return trimmed_freq, trimmed_mean, trimmed_std


def reliability(
    windowlength,
    passing_window_count,
    frequency,
    mean_curve,
    std_curve,
    search_range_in_hz=(None, None),
    verbose=1
):
    freq = xp.asarray(frequency)
    mean_c = xp.asarray(mean_curve)
    std_c = xp.asarray(std_curve)
    if freq.size == 0 or mean_c.size == 0:
        if verbose > 0:
            print("Warning: empty input arrays; reliability cannot be assessed.")
        return np.array([0, 0, 0], dtype=np.int32)
    try:
        defaults = [float(_safe_nanmin(freq)), float(_safe_nanmax(freq))]
    except Exception:
        defaults = [float(np.nan), float(np.nan)]
    limits = []
    limits_were_both_none = True
    for limit, default in zip(search_range_in_hz, defaults):
        if limit is None:
            limits.append(default)
        else:
            limits.append(float(limit))
            limits_were_both_none = False
    search_range_in_hz = tuple(limits)
    if not limits_were_both_none:
        freq, mean_c, std_c = trim_curve(search_range_in_hz, freq, mean_c, std_c, verbose=verbose)
        if freq.size == 0:
            if verbose > 0:
                print("Warning: trimmed frequency array empty; reliability cannot be assessed.")
            return np.array([0, 0, 0], dtype=np.int32)
    mc_peak_index = peak_index(mean_c)
    mc_peak_frq = float(_to_numpy(freq)[mc_peak_index])
    if verbose > 0:
        print(colored("Assessing SESAME (2004) reliability criteria ...", attrs=["bold"]))
    criteria = xp.zeros(3, dtype=xp.int32)
    if _safe_isfinite_scalar(mc_peak_frq) and mc_peak_frq > 10.0 / windowlength:
        criteria[0] = 1
    if verbose > 0:
        print(f" Criteria i): {pass_fail(criteria[0])}")
    nc = windowlength * passing_window_count * mc_peak_frq
    if _safe_isfinite_scalar(nc) and nc > 200:
        criteria[1] = 1
    if verbose > 0:
        print(f" Criteria ii): {pass_fail(criteria[1])}")
    positive_mask = mean_c > 0
    upper_curve = xp.full_like(mean_c, xp.nan, dtype=mean_c.dtype)
    if xp.any(positive_mask):
        mc_pos = mean_c[positive_mask]
        std_pos = std_c[positive_mask]
        with np.errstate(invalid="ignore"):
            upper_curve[positive_mask] = xp.exp(xp.log(mc_pos) + std_pos)
    sigma_a = xp.full_like(mean_c, xp.nan, dtype=mean_c.dtype)
    sigma_a[positive_mask] = upper_curve[positive_mask] / mean_c[positive_mask]
    cond_mask = xp.logical_and(freq > 0.5 * mc_peak_frq, freq < 2.0 * mc_peak_frq)
    if xp.any(cond_mask):
        try:
            sigma_a_max = float(_safe_nanmax(xp.where(cond_mask, sigma_a, xp.nan)))
        except Exception:
            sigma_a_max = float(np.nan)
    else:
        sigma_a_max = float(np.nan)
    if _safe_isfinite_scalar(mc_peak_frq):
        if mc_peak_frq > 0.5:
            if np.isfinite(sigma_a_max) and sigma_a_max < 2:
                criteria[2] = 1
        else:
            if np.isfinite(sigma_a_max) and sigma_a_max < 3:
                criteria[2] = 1
    if verbose > 0:
        print(f" Criteria iii): {pass_fail(criteria[2])}")
    if verbose > 1:
        try:
            sam = float(sigma_a_max)
        except Exception:
            sam = float(np.nan)
        print(f" sigma_a(f)={sam:.03f} {is_isnot(criteria[2])} < {2 if mc_peak_frq > 0.5 else 3}")
    overall = colored("PASSES", "green") if int(xp.sum(criteria)) == 3 else colored("FAILS", "red")
    if verbose > 0:
        print(f" The chosen peak {overall} the peak reliability criteria, with {int(xp.sum(criteria))} of 3.")
    return _to_numpy(criteria).astype(np.int32)


def clarity(
    frequency,
    mean_curve,
    std_curve,
    fn_std,
    search_range_in_hz=(None, None),
    verbose=1
):
    freq = xp.asarray(frequency)
    mean_c = xp.asarray(mean_curve)
    std_c = xp.asarray(std_curve)
    if freq.size == 0 or mean_c.size == 0:
        if verbose > 0:
            print("Warning: empty inputs to clarity(); returning all zeros.")
        return np.zeros(6, dtype=np.int32)
    if verbose > 0:
        print(colored("Assessing SESAME (2004) clarity criteria ...", attrs=["bold"]))
    criteria = xp.zeros(6, dtype=xp.int32)
    try:
        defaults = [float(_safe_nanmin(freq)), float(_safe_nanmax(freq))]
    except Exception:
        defaults = [float(np.nan), float(np.nan)]
    limits = []
    limits_were_both_none = True
    for limit, default in zip(search_range_in_hz, defaults):
        if limit is None:
            limits.append(default)
        else:
            limits.append(float(limit))
            limits_were_both_none = False
    search_range_in_hz = tuple(limits)
    if not limits_were_both_none:
        freq, mean_c, std_c = trim_curve(search_range_in_hz, freq, mean_c, std_c, verbose=verbose)
        if freq.size == 0:
            if verbose > 0:
                print("Warning: trimmed frequency array empty; clarity cannot be assessed.")
            return np.zeros(6, dtype=np.int32)
    mc_peak_index = peak_index(mean_c)
    mc_peak_frq = float(_to_numpy(freq)[mc_peak_index])
    mc_peak_amp = float(_to_numpy(mean_c)[mc_peak_index])

    # Criteria i)
    a_low = mean_c[xp.logical_and(freq < mc_peak_frq, freq > mc_peak_frq / 4.0)]
    if a_low.size > 0:
        if xp.sum(a_low < (mc_peak_amp / 2.0)):
            criteria[0] = 1
        if verbose > 1:
            try:
                amin = float(_to_numpy(xp.nanmin(a_low)))
            except Exception:
                amin = float(np.nan)
            print(f" min(A[fnmc/4,fnmc])={amin:.03f} {is_isnot(criteria[0])} < A0[fnmc]/2={mc_peak_amp:.03f}/2={mc_peak_amp / 2:.03f}")
    else:
        if verbose > 1:
            print(" Warning: no values in low-frequency band for Criteria i).")
    if verbose > 0:
        print(f" Criteria i): {pass_fail(criteria[0])}")

    # Criteria ii)
    a_high = mean_c[xp.logical_and(freq > mc_peak_frq, freq < 4.0 * mc_peak_frq)]
    if a_high.size > 0:
        if xp.sum(a_high < (mc_peak_amp / 2.0)):
            criteria[1] = 1
    else:
        if verbose > 1:
            print(" Warning: no values in high-frequency band for Criteria ii).")
    if verbose > 0:
        print(f"  Criteria ii): {pass_fail(criteria[1])}")

    # Criteria iii)
    if mc_peak_amp > 2:
        criteria[2] = 1
    if verbose > 0:
        print(f"  Criteria iii): {pass_fail(criteria[2])}")

    # Criteria iv)
    positive_mask = mean_c > 0
    upper_curve = xp.full_like(mean_c, xp.nan, dtype=mean_c.dtype)
    lower_curve = xp.full_like(mean_c, xp.nan, dtype=mean_c.dtype)
    if xp.any(positive_mask):
        mc_pos = mean_c[positive_mask]
        std_pos = std_c[positive_mask]
        upper_curve[positive_mask] = xp.exp(xp.log(mc_pos) + std_pos)
        lower_curve[positive_mask] = xp.exp(xp.log(mc_pos) - std_pos)
    try:
        upper_peak_index = peak_index(upper_curve)
        lower_peak_index = peak_index(lower_curve)
    except ValueError:
        upper_peak_index = None
        lower_peak_index = None
    if upper_peak_index is None or lower_peak_index is None:
        cond_1 = cond_2 = False
    else:
        try:
            f_plus = float(_to_numpy(freq)[upper_peak_index])
            f_minus = float(_to_numpy(freq)[lower_peak_index])
            cond_1 = (f_plus > mc_peak_frq * 0.95) and (f_plus < mc_peak_frq * 1.05)
            cond_2 = (f_minus > mc_peak_frq * 0.95) and (f_minus < mc_peak_frq * 1.05)
        except Exception:
            cond_1 = cond_2 = False
    if cond_1 and cond_2:
        criteria[3] = 1
    if verbose > 0:
        print(f"  Criteria iv): {pass_fail(criteria[3])}")

    # Table for v) and vi)
    if mc_peak_frq < 0.2:
        epsilon = 0.25
        theta = 3
    elif mc_peak_frq < 0.5:
        epsilon = 0.2
        theta = 2.5
    elif mc_peak_frq < 1:
        epsilon = 0.15
        theta = 2
    elif mc_peak_frq < 2:
        epsilon = 0.1
        theta = 1.78
    else:
        epsilon = 0.05
        theta = 1.58

    # Criteria v)
    try:
        if fn_std < epsilon * mc_peak_frq:
            criteria[4] = 1
    except Exception:
        criteria[4] = 0
    if verbose > 0:
        print(f"  Criteria v): {pass_fail(criteria[4])}")

    # Criteria vi)
    sigma_a = xp.full_like(mean_c, xp.nan, dtype=mean_c.dtype)
    positive_mask = mean_c > 0
    if xp.any(positive_mask):
        sigma_a[positive_mask] = upper_curve[positive_mask] / mean_c[positive_mask]
    try:
        sigma_a_peak = float(_to_numpy(sigma_a[mc_peak_index])) if (
                    mc_peak_index is not None and mc_peak_index >= 0) else float(np.nan)
    except Exception:
        sigma_a_peak = float(np.nan)

    if np.isfinite(sigma_a_peak) and sigma_a_peak < theta:
        criteria[5] = 1
    if verbose > 0:
        print(f"  Criteria vi): {pass_fail(criteria[5])}")
    overall = colored("PASSES", "green") if int(xp.sum(criteria)) > 4 else colored("FAILS", "red")
    if verbose > 0:
        print(f"  The chosen peak {overall} the peak clarity criteria, with {int(xp.sum(criteria))} of 6.")
    return _to_numpy(criteria).astype(np.int32)



####################################
# Seismic Recording 3C
####################################
def _is_cupy_array(x) -> bool:
    return USE_CUPY and (cp is not None) and isinstance(x, cp.ndarray)

def _to_xp(arr):
    if arr is None:
        raise ValueError("_to_xp: input is None")
    if _is_cupy_array(arr) and xp is np:
        return cp.asnumpy(arr)
    if (not _is_cupy_array(arr)) and (USE_CUPY and xp is cp):
        return cp.asarray(np.asarray(arr))
    return xp.asarray(arr)

def _to_host_list(arr) -> List:
    if _is_cupy_array(arr):
        arr_host = cp.asnumpy(arr)
    else:
        arr_host = np.asarray(arr)
    return arr_host.tolist()

def _assign_amplitude_back(
    ts_obj: Time_Series_CUDA,
    new_arr_xp
):
    orig = getattr(ts_obj, "amplitude", None)

    if _is_cupy_array(orig):
        if not _is_cupy_array(new_arr_xp) and USE_CUPY:
            new_arr = cp.asarray(np.asarray(new_arr_xp))
        else:
            new_arr = new_arr_xp
        ts_obj.amplitude = new_arr
    else:
        if _is_cupy_array(new_arr_xp):
            ts_obj.amplitude = cp.asnumpy(new_arr_xp)
        else:
            ts_obj.amplitude = np.asarray(new_arr_xp)

def _safe_from_timeseries(obj):
    try:
        return Time_Series_CUDA.from_timeseries(obj)
    except Exception:
        pass
    try:
        Time_Series_CUDA(obj)
    except Exception:
        pass
    try:
        return copy.deepcopy(obj)
    except Exception as ex:
        raise TuntimeError("Unable to copy Time_Series_CUDA instance.") from ex


def _apply_nan_policy_to_array(
    arr,
    policy
):
    arr_xp = xp.asarray(arr)
    invalid_mask = ~xp.isfinite(arr_xp)
    any_invalid = bool(xp.any(invalid_mask))
    if not any_invalid:
        return arr_xp
    if policy == "raise":
        raise ValueError("Input contains NaN or Inf (nan_policy='raise').")
    elif policy == "warn":
        warnings.warn("Input contains NaN/Inf; replacing invalid values with zero (nan_policy='warn').")
        return xp.nan_to_num(arr_xp, nan=0.0, posinf=0.0, neginf=0.0)
    elif policy == "fill_zero":
        return xp.nan_to_num(arr_xp, nan=0.0, posinf=0.0, neginf=0.0)
    elif policy == "keep":
        return arr_xp
    else:
        raise ValueError(f"Unknown nan_policy '{policy}'")


class SeismicRecording3C:

    def __init__(
        self,
        ns: Time_Series_CUDA,
        ew: Time_Series_CUDA,
        vt: Time_Series_CUDA,
        degrees_from_north: float = 0.0,
        meta: Optional[dict] = None,
        fill_gaps: bool = False,
        nan_policy: str = "raise"
    ):
        self.nan_policy = str(nan_policy)
        if self.nan_policy not in ("raise", "warn", "fill_zero", "keep"):
            raise ValueError("nan_policy must be one of 'raise', 'warn', 'fill_zero', 'keep'")

        try:
            _fill_fxn = Fill_Gaps_Linear
        except NameError:
            _fill_fxn = None

        if not hasattr(ns, "is_similar"):
            raise AttributeError("Provided 'ns' object does not have method is_similar().")

        for name, component in zip(["ns", "ew", "vt"], [ns, ew, vt]):
            if not ns.is_similar(component):
                msg = f"Component {name} is not similar to component ns; all components must be similar."
                raise ValueError(msg)

        tseries = []
        for comp in (ns, ew, vt):
            tseries.append(_safe_from_timeseries(comp))
        self.ns, self.ew, self.vt = tseries

        for name, ts in zip(["ns", "ew", "vt"], (self.ns, self.ew, self.vt)):
            if not hasattr(ts, "amplitude"):
                raise AttributeError(f"Time_Series object for {name} has no attribute 'amplitude'.")
            amp = getattr(ts, "amplitude")
            if amp is None:
                raise ValueError(f"Amplitude data for {name} is None.")

        if fill_gaps:
            if _fill_fxn is None:
                raise RuntimeError("fill_gaps requested but helper 'Fill_Gaps_Linear' is not defined/imported.")
            try:
                self.ns.amplitude = _fill_fxn(self.ns.amplitude, max_gap=20)
                self.ew.amplitude = _fill_fxn(self.ew.amplitude, max_gap=20)
                self.vt.amplitude = _fill_fxn(self.vt.amplitude, max_gap=20)
            except Exception as ex:
                raise RuntimeError(f"Fill_Gaps_Linear failed during SeismicRecording3C init: {ex}") from ex

        try:
            self.ns.amplitude = _apply_nan_policy_to_array(self.ns.amplitude, self.nan_policy)
            self.ew.amplitude = _apply_nan_policy_to_array(self.ew.amplitude, self.nan_policy)
            self.vt.amplitude = _apply_nan_policy_to_array(self.vt.amplitude, self.nan_policy)
        except Exception as ex:
            raise

        try:
            self.degrees_from_north = float(degrees_from_north % 360.0)
        except Exception:
            self.degrees_from_north = float(math.fmod(float(degrees_from_north), 360.0))

        meta = {} if meta is None else meta.copy()
        self.meta = {
            "file name(s)": "seismic recording was not created from file",
            "deployed degrees from north": self.degrees_from_north,
            "current degrees from north": self.degrees_from_north,
            **meta
        }

        try:
            len_ns = int(xp.asarray(self.ns.amplitude).shape[0])
            len_ew = int(xp.asarray(self.ew.amplitude).shape[0])
            len_vt = int(xp.asarray(self.vt.amplitude).shape[0])
        except Exception:
            len_ns = int(np.asarray(self.ns.amplitude).shape[0])
            len_ew = int(np.asarray(self.ew.amplitude).shape[0])
            len_vt = int(np.asarray(self.vt.amplitude).shape[0])
        if not (len_ns == len_ew == len_vt):
            raise ValueError(f"Component amplitude lengths differ: ns={len_ns}, ew={len_ew}, vt={len_vt}")

    def trim(
        self,
        start_time: float,
        end_time: float
    ) -> None:
        self.meta["trim"] = (start_time, end_time)
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).trim(start_time=start_time, end_time=end_time)

    def detrend(
        self,
        type: str = "linear"
    ) -> None:
        self.meta["detrend"] = type
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).detrend(type=type)

    def split(
        self,
        window_length_in_seconds: float
    ) -> List["SeismicRecording3C"]:
        self.meta["split"] = window_length_in_seconds
        split_recordings = []
        for _ns, _ew, _vt in zip(self.ns.split(window_length_in_seconds),
                                 self.ew.split(window_length_in_seconds),
                                 self.vt.split(window_length_in_seconds)):
            split_recordings.append(
                SeismicRecording3C(_ns, _ew, _vt,
                                   degrees_from_north=self.degrees_from_north,
                                   meta=self.meta)
            )
        return split_recordings


    def window(
        self,
        type: str = "tukey",
        width: float = 0.1
    ) -> None:
        self.meta["window_type_and_width"] = (type, width)
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).window(type=type, width=width)

    def butterworth_filter(
        self,
        fcs_in_hz: Tuple[Optional[float],
        Optional[float]], order: int = 5
    ) -> None:
        self.meta["butterworth_filter"] = fcs_in_hz
        for component in ["ns", "ew", "vt"]:
            getattr(self, component).butterworth_filter(fcs_in_hz=fcs_in_hz, order=order)


    def orient_sensor_to(
        self,
        degrees_from_north: float
    ) -> None:

        angle_diff_degrees = degrees_from_north - self.degrees_from_north
        angle_diff_radians = float(np.radians(angle_diff_degrees))

        c = float(xp.cos(angle_diff_radians))
        s = float(xp.sin(angle_diff_radians))

        ns_amp_orig = getattr(self.ns, "amplitude", None)
        ew_amp_orig = getattr(self.ew, "amplitude", None)

        if ns_amp_orig is None or ew_amp_orig is None:
            raise ValueError("orient_sensor_to: missing amplitude data in ns/ew components")

        ns_xp = _to_xp(ns_amp_orig)
        ew_xp = _to_xp(ew_amp_orig)

        if ns_xp.shape != ew_xp.shape:
            raise ValueError(f"orient_sensor_to: ns and ew amplitude shapes differ: {ns_xp.shape} vs {ew_xp.shape}")

        invalid_ns = bool(xp.any(~xp.isfinite(ns_xp)).item() if hasattr(xp.any(~xp.isfinite(ns_xp)), "item") else bool(xp.any(~xp.isfinite(ns_xp))))
        invalid_ew = bool(xp.any(~xp.isfinite(ew_xp)).item() if hasattr(xp.any(~xp.isfinite(ew_xp)), "item") else bool(xp.any(~xp.isfinite(ew_xp))))
        has_invalid = invalid_ns or invalid_ew

        policy = getattr(self, "nan_policy", "raise")
        if has_invalid:
            if nan_policy == "raise":
                raise ValueError("orient_sensor_to: input amplitude contains NaN/Inf")
            elif nan_policy in ("warn", "fill_zero"):
                if nan_policy == "warn":
                    warnings.warn("orient_sensor_to: inputamplitude contains NaN/Inf; replacing with zeros.")
                ns_xp = xp.nan_to_num(ns_xp, nan=0.0, posinf=0.0, neginf=0.0)
                ew_xp = xp.nan_to_num(ew_xp, nan=0.0, posinf=0.0, neginf=0.0)
            elif nan_policy == "keep":
                pass
            else:
                raise ValueError(f"Unknown nan_policy '{nan_policy}'")

        # ew_new = ew*c - ns*s
        # ns_new = ew*s + ns*c
        ew_new_xp = ew_xp * c - ns_xp * s
        ns_new_xp = ew_xp * s + ns_xp * c

        _assign_amplitude_back(self.ew, ew_new_xp)
        _assign_amplitude_back(self.ns, ns_new_xp)

        self.degrees_from_north = float(degrees_from_north % 360.0)
        self.meta["current degrees from north"] = self.degrees_from_north


    def _to_dict(self) -> dict:
        return dict(
            dt_in_seconds=self.ns.dt_in_seconds,
            ns_amplitude=_to_host_list(self.ns.amplitude),
            ew_amplitude=_to_host_list(self.ew.amplitude),
            vt_amplitude=_to_host_list(self.vt.amplitude),
            degrees_from_north=self.degrees_from_north,
            meta=self.meta
        )

    @classmethod
    def _from_dict(
        cls,
        data: dict
    ) -> "SeismicRecording3C":
        ns = Time_Series_CUDA(data["ns_amplitude"], data["dt_in_seconds"])
        ew = Time_Series_CUDA(data["ew_amplitude"], data["dt_in_seconds"])
        vt = Time_Series_CUDA(data["vt_amplitude"], data["dt_in_seconds"])
        degrees_from_north = float(data.get("degrees_from_north", 0.0))
        meta = data.get("meta", {})
        return cls(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)

    def save(self, fname: str) -> None:
        with open(fname, "w") as f:
            json.dump(self._to_dict(), f)

    @classmethod
    def load(cls, fname: str) -> "SeismicRecording3C":
        with open(fname, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def from_seismic_recording_3c(
        cls,
        seismic_recording_3c: "SeismicRecording3C"
    ) -> "SeismicRecording3C":
        new_components = []
        for comp in ["ns", "ew", "vt"]:
            ts = getattr(seismic_recording_3c, comp)
            new_components.append(Time_Series_CUDA.from_timeseries(ts))
        return cls(*new_components, degrees_from_north=seismic_recording_3c.degrees_from_north, meta=seismic_recording_3c.meta)

    def is_similar(
        self,
        other: "SeismicRecording3C"
    ) -> bool:
        if not isinstance(other, SeismicRecording3C):
            return False
        for attr in ["ns", "ew", "vt"]:
            if not getattr(self, attr).is_similar(getattr(other, attr)):
                return False
        return True

    def __eq__(
        self,
        other: object
    ) -> bool:
        if not self.is_similar(other):
            return False
        for attr in ["ns", "ew", "vt", "meta"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        if abs(self.degrees_from_north - other.degrees_from_north) > 0.1:
            return False
        return True

    def __str__(self) -> str:
        return f"SeismicRecording3C at {id(self)}"

    def __repr__(self) -> str:
        return f"SeismicRecording3C(ns={self.ns}, ew={self.ew}, vt={self.vt}, meta={self.meta})"



####################################
# Instrument Response
####################################
#=================
# Helpers
#=================
def _is_cupy_array(x) -> bool:
    return USE_CUPY and (cp is not None) and isinstance(x, cp.ndarray)

def _to_xp_array(arr):
    if USE_CUPY:
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return cp.asarray(arr)
    else:
        return np.asarray(arr)

def _to_host_array_if_needed(arr):
    if _is_cupy_array(arr):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def _preserve_array_type(orig_array, out_xp_array):
    if _is_cupy_array(orig_array):
        return out_xp_array
    if USE_CUPY and isinstance(out_xp_array, cp.ndarray):
        return cp.asnumpy(out_xp_array)
    return np.asarray(out_xp_array)

@contextmanager
def safe_errstate(xp, **kwargs):
    if xp.__name__ == "numpy":
        with np.errstate(**kwargs):
            yield
    else:
        yield

#==================================
# Instrument Transfer Function
#==================================
class InstrumentTransferFunction():

    def __init__(
        self, poles: Iterable[complex],
        zeros: Iterable[complex],
        instrument_sensitivity: float,
        normalization_factor: float
    ):
        self._zeros = xp.asarray([complex(z) for z in zeros], dtype=xp.complex128) if zeros is not None else xp.asarray([], dtype=xp.complex128)
        self._poles = xp.asarray([complex(p) for p in poles], dtype=xp.complex128) if poles is not None else xp.asarray([], dtype=xp.complex128)
        self.instrument_sensitivity = float(instrument_sensitivity)
        self.normalization_factor = float(normalization_factor)


    def _h(
        self,
        frequencies
    ):
        frq_xp = xp.asarray(frequencies, dtype=xp.float64)
        s = 1j * 2.0 * xp.pi * frq_xp

        zeros = xp.asarray(self._zeros, dtype=xp.complex128)
        poles = xp.asarray(self._poles, dtype=xp.complex128)

        # Numerator
        if zeros.size > 0:
            num_factors = s.reshape((-1, 1)) - zeros.reshape((1, -1))
            numerator = xp.prod(num_factors, axis=1, dtype=xp.complex128)
        else:
            numerator = xp.ones_like(s, dtype=xp.complex128)

        # Denominator
        if poles.size > 0:
            den_factors = s.reshape((-1, 1)) - poles.reshape((1, -1))
            denominator = xp.prod(den_factors, axis=1, dtype=xp.complex128)
        else:
            denominator = xp.ones_like(s, dtype=xp.complex128)

        abs_den = xp.abs(denominator)
        eps = DEFAULT_EPS
        safe_mask = abs_den > eps

        h = xp.zeros_like(denominator, dtype=xp.complex128)
        if xp.any(safe_mask):
            h[safe_mask] = numerator[safe_mask] / denominator[safe_mask]

        h = h * self.normalization_factor * self.instrument_sensitivity
        return h


    def response(
        self,
        frequencies
    ):
        h = self._h(frequencies)
        amplitude = xp.abs(h)
        phase_rad = xp.angle(h)
        phase_deg = (phase_rad * (180.0 / xp.pi))
        return amplitude, phase_deg


    def from_resp(self, fname):
        raise NotImplementedError("from_resp not implemented in GPU variant")

    def __str__(self):
        return f"InstrumentTransferFunction at {id(self)}"

    def __repr__(self):
        return (f"InstrumentTransferFunction(poles={list(self._poles)}, zeros={list(self._zeros)}, "
                f"instrument_sensitivity={self.instrument_sensitivity}, normalization_factor={self.normalization_factor})")


#=======================================================
# Domain transforms: derivative / integral using FFT
#=======================================================
def Domain_Transform(
    transform_type: str,
    timeseries: Time_Series_CUDA,
    fft_settings: Optional[Dict] = None
) -> Time_Series_CUDA:

    if fft_settings is None:
        fft_settings = {}

    n = int(fft_settings.get("n", timeseries.n_samples))

    orig_amp = getattr(timeseries, "amplitude")
    amp_xp = xp.asarray(orig_amp, dtype=xp.float64)

    fft = xp.fft.rfft(amp_xp, n=n)
    frq = xp.fft.rfftfreq(n, d=timeseries.dt_in_seconds)

    if transform_type == "derivative":
        transfer_function = 1j * 2.0 * xp.pi * frq.astype(xp.complex128)
    elif transform_type == "integral":
        transfer_function = xp.zeros(frq.shape, dtype=xp.complex128)
        nonzero_mask = frq != 0.0
        if xp.any(nonzero_mask):
            denom = 1j * 2.0 * xp.pi * frq[nonzero_mask]
            transfer_function_mask = 1.0 / denom
            transfer_function[nonzero_mask] = transfer_function_mask
    else:
        raise NotImplementedError(f"Unknown transform_type {transform_type}")

    fft = fft * transfer_function

    try:
        fft = xp.nan_to_num(fft, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        fft = xp.where(xp.isfinite(fft), fft, 0.0+0.0j)

    ifft = xp.fft.irfft(fft, n=n)
    ifft = ifft[:timeseries.n_samples]

    out_amp = _preserve_array_type(orig_amp, ifft)
    return Time_Series_CUDA(out_amp, dt_in_seconds=timeseries.dt_in_seconds)


def Differentiate(
    timeseries: Time_Series_CUDA,
    fft_settings: Optional[Dict] = None
) -> Time_Series_CUDA:
    return Domain_Transform("derivative", timeseries, fft_settings)


def Integrate(
    timeseries: Time_Series_CUDA,
    fft_settings: Optional[Dict] = None
) -> Time_Series_CUDA:
    return Domain_Transform("integral", timeseries, fft_settings)


#=================================
# Remove instrument response
#=================================
def Remove_Instrument_Response(
    timeseries: Time_Series_CUDA,
    instrument_transfer_function: InstrumentTransferFunction,
    fft_settings: Optional[Dict] = None
) -> Time_Series_CUDA:

    if fft_settings is None:
        fft_settings = {}

    n = int(fft_settings.get("n", timeseries.n_samples))
    orig_amp = getattr(timeseries, "amplitude")
    amp_xp = xp.asarray(orig_amp, dtype=xp.float64)

    fft = xp.fft.rfft(amp_xp, n=n)
    frq = xp.fft.rfftfreq(n, d=timeseries.dt_in_seconds)

    h = instrument_transfer_function._h(frq)

    eps = DEFAULT_EPS
    abs_h = xp.abs(h)
    safe_mask = abs_h > eps

    invh = xp.zeros_like(h, dtype=xp.complex128)
    if xp.any(safe_mask):
        invh[safe_mask] = 1.0 / h[safe_mask]

    dc_mask = frq == 0.0
    if xp.any(dc_mask):
        invh[dc_mask] = 0.0 + 0.0j

    fft = fft * invh

    try:
        fft = xp.nan_to_num(fft, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        fft = xp.where(xp.isfinite(fft), fft, 0.0+0.0j)

    ifft = xp.fft.irfft(fft, n=n)
    ifft = ifft[:timeseries.n_samples]

    out_amp = _preserve_array_type(orig_amp, ifft)
    return Time_Series_CUDA(out_amp, dt_in_seconds=timeseries.dt_in_seconds)



########################################
# Interact (Plot Interaction module)
########################################
#=================================
# Helpers (backend-agnostic)
#=================================
def _to_host_scalar(x):
    if x is None:
        raise ValueError("_to_host_scalar received None")
    if USE_CUPY:
        try:
            arr = cp.asarray(x)
            if isinstance(arr, cp.ndarray):
                host = cp.asnumpy(arr)
                try:
                    return float(host.item())
                except Exception:
                    return float(np.asarray(host).tolist())
        except Exception:
            try:
                return float(x)
            except Exception:
                raise TypeError(f" _to_host_scalar: cannot convert {type(x)} to float: {e}")
    else:
        try:
            arr = np.asarray(x)
            if arr.shape == ():
                return float(arr.item())
            if arr.size == 1:
                return float(arr.flatten()[0])
            raise TypeError("_to_host_scalar expects a scalar or 0-dim array-like")
        except Exception:
            try:
                return float(x)
            except Exception as e:
                raise TypeError(f" _to_host_scalar: cannot convert {type(x)} to float: {e}")

def _to_host_array(a):
    if a is None:
        return np.asarray([])
    if USE_CUPY and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)

def _ensure_float(value) -> float:
    try:
        return _to_host_scalar(value)
    except Exception:
        try:
            return float(np.asarray(value).astype(float).item())
        except Exception as e:
            raise TypeError(f" _ensure_float: cannot coerce value to float: {e}")


#=========================
# Interactive session
#=========================
def ginput_session(
    fig, ax,
    initial_adjustment=True,
    initial_adjustment_message=None,
    n_points=1,
    ask_to_confirm_point=True,
    ask_to_continue=True,
    ask_to_continue_message=None
):
    Cursor(ax, useblit=True, color="k", linewidth=1)

    if initial_adjustment:
        if initial_adjustment_message is None:
            initial_adjustment_message = "Adjust view,\nspacebar when ready."
        text = ax.text(0.95, 0.95, initial_adjustment_message, ha="right", va="top", transform=ax.transAxes)
        while True:
            if plt.waitforbuttonpress(timeout=-1):
                text.set_visible(False)
                break

    npt, xs, ys = 0, [], []
    while npt < n_points:
        if ask_to_confirm_point:
            selection_message = "Left click to add,\nright click to remove,\nenter to accept."
            text = ax.text(0.95, 0.95, selection_message,
                           ha="right", va="top", transform=ax.transAxes)
            vals = plt.ginput(n=-1, timeout=0)
            text.set_visible(False)
        else:
            vals = plt.ginput(n=1, timeout=0)

        if len(vals) > 1:
            msg = "More than one point selected, ignoring all but the last point."
            warnings.warn(msg)

        if len(vals) == 0:
            msg = "No points selected, try again."
            warnings.warn(msg)
            continue

        x, y = vals[-1]
        ax.plot(float(x), float(y), "r", marker="+", linestyle="")
        xs.append(float(x))
        ys.append(float(y))
        npt += 1
        fig.canvas.draw_idle()

        if ask_to_continue:
            if ask_to_continue_message is None:
                ask_to_continue_message = "Adjust view,\npress spacebar\nonce to contine,\ntwice to exit."
            text = ax.text(0.95, 0.95, ask_to_continue_message,
                           ha="right", va="top",
                           transform=ax.transAxes)
            while True:
                if plt.waitforbuttonpress(timeout=-1):
                    text.set_visible(False)
                    break
        if plt.waitforbuttonpress(timeout=0.3):
            break

    finish_message = "Interactive session complete,\nclose figure(s) when ready."
    ax.text(0.95, 0.95, finish_message, ha="right", va="top", transform=ax.transAxes)
    return (xs, ys)


#================================
# Numeric helpers using xp
#================================
def _relative_to_absolute(
    relative,
    range_absolute: Sequence[float],
    scale: str = "linear"
):
    if relative is None:
        raise ValueError("_relative_to_absolute: relative must be provided")

    abs_min, abs_max = float(range_absolute[0]), float(range_absolute[1])
    rel = float(relative)
    if math.isnan(rel):
        raise ValueError("_relative_to_absolute: relative is NaN")
    rel = max(0.0, min(1.0, rel))

    if scale == "linear":
        if abs_max == abs_min:
            raise ValueError("Absolute range has zero width (abs_max == abs_min)")
        val = abs_min + rel * (abs_max - abs_min)
        return float(val)
    elif scale == "log":
        if abs_min <= 0 or abs_max <= 0:
            raise ValueError("For log scale, absolute range must be positive.")
        val_xp = xp.log10(abs_min) + rel * (xp.log10(abs_max) - xp.log10(abs_min))
        return _to_host_scalar(xp.power(10.0, val_xp))
    else:
        raise NotImplementedError(f" _relative_to_absolute: unknown scale {scale}")

def _absolute_to_relative(
    absolute,
    range_absolute: Sequence[float],
    scale: str = "linear"
):
    abs_min, abs_max = float(range_absolute[0]), float(range_absolute[1])
    a = _ensure_float(absolute)
    if scale == "linear":
        return (a - abs_min) / (abs_max - abs_min)
    elif scale == "log":
        if a <= 0 or abs_min <= 0:
            raise ValueError("For log scale, values must be positive.")
        num = xp.log10(a / abs_min)
        denom = xp.log10(abs_max / abs_min)
        return _to_host_scalar(num / denom)
    else:
        raise NotImplementedError

def _absolute_to_relative(
    absolute,
    range_absolute: Sequence[float],
    scale: str = "linear"
):
    abs_min, abs_max = float(range_absolute[0]), float(range_absolute[1])
    a = _ensure_float(absolute)
    if scale == "linear":
        if abs_max == abs_min:
            raise ValueError("Absolute range has zero width (abs_max == abs_min)")
        return float((a - abs_min) / (abs_max - abs_min))
    elif scale == "log":
        if a <= 0 or abs_min <= 0:
            raise ValueError("For log scale, values must be positive.")
        num = xp.log10(a / abs_min)
        denom = xp.log10(abs_max / abs_min)
        return _to_host_scalar(num / denom)
    else:
        raise NotImplementedError(f"_absolute_to_relative: unknown scale {scale}")

def _relative_box_coordinates(
    upper_right_corner_relative=(0.95, 0.95),
    box_size_relative=(0.1, 0.05)
):
    x_upper_rel, y_upper_rel = (float(upper_right_corner_relative[0]), float(upper_right_corner_relative[1]))
    x_box_rel, y_box_rel = (float(box_size_relative[0]), float(box_size_relative[1]))
    x_lower_rel, y_lower_rel = x_upper_rel - x_box_rel, y_upper_rel - y_box_rel
    x_lower_rel = max(0.0, min(1.0, x_lower_rel))
    x_upper_rel = max(0.0, min(1.0, x_upper_rel))
    y_lower_rel = max(0.0, min(1.0, y_lower_rel))
    y_upper_rel = max(0.0, min(1.0, y_upper_rel))
    return (x_lower_rel, x_upper_rel, y_lower_rel, y_upper_rel)

def _absolute_box_coordinates(
    x_range_absolute: Sequence[float],
    y_range_absolute: Sequence[float],
    upper_right_corner_relative=(0.95, 0.95),
    box_size_relative=(0.1, 0.05),
    x_scale="linear",
    y_scale="linear"
):
    x_box_lower_abs = _relative_to_absolute(_relative_box_coordinates(upper_right_corner_relative, box_size_relative)[0], x_range_absolute, scale=x_scale)
    x_box_upper_abs = _relative_to_absolute(_relative_box_coordinates(upper_right_corner_relative, box_size_relative)[1], x_range_absolute, scale=x_scale)
    y_box_lower_abs = _relative_to_absolute(_relative_box_coordinates(upper_right_corner_relative, box_size_relative)[2], y_range_absolute, scale=y_scale)
    y_box_upper_abs = _relative_to_absolute(_relative_box_coordinates(upper_right_corner_relative, box_size_relative)[3], y_range_absolute, scale=y_scale)
    return (x_box_lower_abs, x_box_upper_abs, y_box_lower_abs, y_box_upper_abs)


#=======================================================
# Plot utilities (convert xp -> host where needed)
#=======================================================
def plot_continue_button(
    ax,
    upper_right_corner_relative=(0.95, 0.95),
    box_size_relative=(0.1, 0.05),
    fill_kwargs=None
):
    x_scale = ax.get_xscale()
    y_scale = ax.get_yscale()
    x_range_absolute = ax.get_xlim()
    y_range_absolute = ax.get_ylim()
    box_abs = _absolute_box_coordinates(x_range_absolute=x_range_absolute,
                                        y_range_absolute=y_range_absolute,
                                        upper_right_corner_relative=upper_right_corner_relative,
                                        box_size_relative=box_size_relative,
                                        x_scale=x_scale,
                                        y_scale=y_scale)
    x_box_lower_abs, x_box_upper_abs, y_box_lower_abs, y_box_upper_abs = box_abs

    default_kwargs = dict(color="lightgreen")
    if fill_kwargs is None:
        fill_kwargs = {}
    fill_kwargs = {**default_kwargs, **fill_kwargs}

    ax.fill([x_box_lower_abs, x_box_lower_abs, x_box_upper_abs, x_box_upper_abs],
            [y_box_lower_abs, y_box_upper_abs, y_box_upper_abs, y_box_lower_abs],
            **fill_kwargs)

    ax.text((upper_right_corner_relative[0] - box_size_relative[0]/2),
            (upper_right_corner_relative[1] - box_size_relative[1]/2),
            "continue?",
            ha="center", va="center", transform=ax.transAxes)


def is_absolute_point_in_relative_box(
    ax,
    absolute_point: Sequence[float],
    upper_right_corner_relative=(0.95, 0.95),
    box_size_relative=(0.1, 0.05)
):
    x_min_rel, x_max_rel, y_min_rel, y_max_rel = _relative_box_coordinates(upper_right_corner_relative, box_size_relative)
    abs_x, abs_y = float(absolute_point[0]), float(absolute_point[1])
    rel_x = _absolute_to_relative(abs_x, ax.get_xlim(), ax.get_xscale())
    rel_y = _absolute_to_relative(abs_y, ax.get_ylim(), ax.get_yscale())
    return (rel_x > x_min_rel) and (rel_x < x_max_rel) and (rel_y > y_min_rel) and (rel_y < y_max_rel)



####################################
# Data Wrangler
####################################
#==========================
# Backend helpers
#==========================
def to_xp_array(
    arr,
    dtype=xp.float32
):
    if USE_CUPY:
        if isinstance(arr, cp.ndarray):
            return arr.astype(dtype, copy=False)
        return cp.asarray(np.asarray(arr), dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype)

def to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def to_gpu_timeseries(ts: Time_Series_CUDA):
    try:
        amp = getattr(ts, "amplitude", None)
    except Exception:
        return ts
    if amp is None:
        return ts
    if USE_CUPY and isinstance(amp, np.ndarray):
        try:
            ts.amplitude = cp.asarray(amp, dtype=cp.float32)
        except Exception:
            ts.amplitude = np.asarray(amp, dtype=np.float32)
    else:
        if isinstance(amp, np.ndarray):
            ts.amplitude = np.asarray(amp, dtype=np.float32)
        elif USE_CUPY and isinstance(amp, cp.ndarray):
            ts.amplitude = amp.astype(cp.float32, copy=False)
    return ts


#============================
# Small utility wrappers
#============================
def Arrange_Traces(traces):
    found_ew = found_ns = found_vt = False
    for trace in traces:
        ch = getattr(trace, "meta", {}).get("channel", "")
        if not ch and hasattr(trace, "stats"):
            ch = getattr(trace.stats, "channel", "")
        if ch.endswith("E") and not found_ew:
            ew = Time_Series_CUDA.from_trace(trace)
            found_ew = True
        elif ch.endswith("N") and not found_ns:
            ns = Time_Series_CUDA.from_trace(trace)
            found_ns = True
        elif ch.endswith("Z") and not found_vt:
            vt = Time_Series_CUDA.from_trace(trace)
            found_vt = True
        else:
            label = ch.upper()
            if label.endswith("E") and not found_ew:
                ew = Time_Series_CUDA.from_trace(trace); found_ew = True
            elif label.endswith("N") and not found_ns:
                ns = Time_Series_CUDA.from_trace(trace); found_ns = True
            elif label.endswith("Z") and not found_vt:
                vt = Time_Series_CUDA.from_trace(trace); found_vt = True
            else:
                msg = "Missing, duplicate, or incorrectly named components."
                raise ValueError(msg)
    ns = to_gpu_timeseries(ns)
    ew = to_gpu_timeseries(ew)
    vt = to_gpu_timeseries(vt)
    return ns, ew, vt


def Check_npts(
    npts_header,
    npts_found
):
    if npts_header != npts_found:
        msg = (f"Points listed in file header ({npts_header}) does not match "
               f"the number of points found ({npts_found}) please report this "
               "issue to the CuHVSR developers via GitHub issues.")
        raise ValueError(msg)

def Quiet_Obspy_Read(
    *args,
    **kwargs
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = obspy.read(*args, **kwargs)
    return results


#======================================================================
# Readers for various formats (convert amplitude to GPU if available)
#======================================================================
def Read_mseed(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "MSEED"}

    def Read_stream(path_like):
        return Quiet_Obspy_Read(path_like, **obspy_read_kwargs)

    if isinstance(fnames, (str, pathlib.Path, io.BytesIO)):
        traces = Read_stream(fnames)
        used_fnames = str(fnames)
    elif isinstance(fnames, (list, tuple)):
        if len(fnames) == 1 and isinstance(fnames[0], (str, pathlib.Path, io.BytesIO)):
            tmp = Read_stream(fnames[0])
            if len(tmp) == 3:
                traces = tmp
                used_fnames = [str(fnames[0])]
            else:
                trace_list = []
                meta_names = []
                for f in fnames:
                    st = Read_stream(f)
                    for tr in st:
                        trace_list.append(tr)
                    meta_names.append(str(f))
                traces = obspy.Stream(trace_list)
                used_fnames = meta_names
        else:
            trace_list = []
            meta_names = []
            for f in fnames:
                st = Read_stream(f)
                for tr in st:
                    trace_list.append(tr)
                meta_names.append(str(f))
            traces = obspy.Stream(trace_list)
            used_fnames = meta_names
    else:
        raise ValueError(f"`fnames` must be str, Path, BytesIO, or list/tuple thereof; got {type(fnames)}")

    if len(traces) > 3:
        found = {"N": None, "E": None, "Z": None}
        for tr in traces:
            ch = getattr(tr.stats, "channel", "") or ""
            ch_up = ch.upper()
            if "N" in ch_up and found["N"] is None:
                found["N"] = tr
            elif "E" in ch_up and found["E"] is None:
                found["E"] = tr
            elif "Z" in ch_up and found["Z"] is None:
                found["Z"] = tr
            if all(found.values()):
                break
        if all(found.values()):
            traces = obspy.Stream([found["N"], found["E"], found["Z"]])

    if len(traces) != 3:
        raise ValueError(
            f"Provided {len(traces)} traces; expected exactly 3 (one per component). "
            "If you have a single multi-trace file please provide its path (string), "
            "or ensure your list-of-files yields 3 traces in total. "
            "If your file contains >3 traces, ensure it includes distinct channel names with N, E, and Z."
        )

    ts_by_component = {"N": None, "E": None, "Z": None}

    for tr in traces:
        ch = getattr(tr.stats, "channel", None)
        if ch is None:
            raise ValueError("Trace missing stats.channel; cannot determine component.")
        ch_up = ch.upper()

        if "N" in ch_up and ts_by_component["N"] is None:
            comp = "N"
        elif "E" in ch_up and ts_by_component["E"] is None:
            comp = "E"
        elif "Z" in ch_up and ts_by_component["Z"] is None:
            comp = "Z"
        else:
            last = ch_up[-1]
            if last in ("N", "E", "Z") and ts_by_component.get(last) is None:
                comp = last
            else:
                raise ValueError(f"Unable to map channel name '{ch}' to component N/E/Z")

        # Get data and dt
        data_np = np.asarray(tr.data, dtype=np.float32)

        sampling_rate = getattr(tr.stats, "sampling_rate", None)
        dt = None
        if sampling_rate not in (None, 0):
            dt = 1.0 / float(sampling_rate)
        else:
            delta = getattr(tr.stats, "delta", None)
            if delta not in (None, 0):
                dt = float(delta)

        if dt is None:
            raise ValueError("Trace missing sampling information (sampling_rate or delta); cannot determine dt_in_seconds.")

        starttime = getattr(tr.stats, "starttime", None)

        try:
            amp_xp = to_xp_array(data_np)
        except Exception:
            amp_xp = data_np

        TS = Time_Series_CUDA
        ts_obj = None

        try:
            ts_obj = TS(amp_xp, dt)
        except Exception:
            try:
                ts_obj = TS(data_np, dt)
                if USE_CUPY:
                    ts_obj = to_gpu_timeseries(ts_obj)
            except Exception as final_e:
                raise RuntimeError(f"Unable to construct Time_Series_CUDA for component {comp}: {final_e}")
        if starttime is not None:
            try:
                setattr(ts_obj, "start_time", starttime)
            except Exception:
                pass
        ts_by_component[comp] = ts_obj

    if any(ts_by_component[c] is None for c in ("N", "E", "Z")):
        missing = [c for c in ("N", "E", "Z") if ts_by_component[c] is None]
        present = [getattr(t, "stats", {}).channel if getattr(t, "stats", None) else None for t in traces]
        raise ValueError(f"Missing components: {missing}. Traces present: {present}")
    ns = ts_by_component["N"]
    ew = ts_by_component["E"]
    vt = ts_by_component["Z"]
    if degrees_from_north is None:
        degrees_from_north = 0.0
    meta = {"file name(s)": used_fnames}
    return SeismicRecording3C(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)


def Read_Minishark(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if isinstance(fnames, (list, tuple)):
        raise ValueError(f"Only 1 minishark file allowed; {len(fnames)} provided.")
    elif isinstance(fnames, io.StringIO):
        fnames.seek(0, 0)
        text = fnames.read()
        fname = fnames
    else:
        fname = fnames
        with open(fname, "r") as f:
            text = f.read()
    match = mshark_npts_exec.search(text)
    if match is None:
        raise ValueError("Failed to parse npts header in minishark input (regex did not match).")
    npts_header = int(match.groups()[0])
    dt = 1.0 / float(mshark_fs_exec.search(text).groups()[0])
    conversion = int(mshark_conversion_exec.search(text).groups()[0])
    gain = int(mshark_gain_exec.search(text).groups()[0])
    data = np.empty((npts_header, 3), dtype=np.float32)
    idx = 0
    for group in mshark_row_exec.finditer(text):
        vt_s, ns_s, ew_s = group.groups()
        data[idx, 0] = float(vt_s)
        data[idx, 1] = float(ns_s)
        data[idx, 2] = float(ew_s)
        idx += 1
    Check_npts(npts_header, idx)
    data /= gain
    data /= conversion
    vt_arr, ns_arr, ew_arr = data.T
    vt = Time_Series_CUDA(vt_arr, dt_in_seconds=dt)
    ns = Time_Series_CUDA(ns_arr, dt_in_seconds=dt)
    ew = Time_Series_CUDA(ew_arr, dt_in_seconds=dt)
    if USE_CUPY:
        ns = to_gpu_timeseries(ns)
        ew = to_gpu_timeseries(ew)
        vt = to_gpu_timeseries(vt)
    if degrees_from_north is None:
        degrees_from_north = 0.0
    meta = {"file name(s)": str(fname)}
    return SeismicRecording3C(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)


def Read_SAC(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "SAC"}

    if not isinstance(fnames, (list, tuple)):
        raise ValueError("Must provide 3 sac files (one per trace); only one provided.")

    trace_list = []
    for fname in fnames:
        last_exc = None
        for byteorder in ["little", "big"]:
            if isinstance(fname, io.BytesIO):
                fname.seek(0, 0)
            obspy_read_kwargs["byteorder"] = byteorder
            try:
                stream = Quiet_Obspy_Read(fname, **obspy_read_kwargs)
            except Exception as e:
                last_exc = e
                continue
            else:
                break
        else:
            raise last_exc

        trace = stream[0]
        trace_list.append(trace)
    traces = obspy.Stream(trace_list)

    if len(traces) != 3:
        raise ValueError(f"Provided {len(traces)} traces, but must only provide 3.")

    ns, ew, vt = Arrange_Traces(traces)

    if degrees_from_north is None:
        degrees_from_north = 0.0

    meta = {"file name(s)": [str(fname) for fname in fnames]}
    return SeismicRecording3C(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)


# GCF (Guralp Compressed Format)
def Read_GCF(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if obspy_read_kwargs is None:
        obspy_read_kwargs = {"format": "GCF"}
    if isinstance(fnames, (list, tuple)):
        raise ValueError(f"Only 1 gcf file allowed; {len(fnames)} provided.")
    else:
        fname = fnames
    traces = Quiet_Obspy_Read(fname, **obspy_read_kwargs)
    if len(traces) != 3:
        raise ValueError(f"Provided {len(traces)} traces, but must only provide 3.")
    ns, ew, vt = Arrange_Traces(traces)
    if degrees_from_north is None:
        degrees_from_north = 0.0
    meta = {"file name(s)": str(fname)}
    return SeismicRecording3C(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)


def Read_PEER(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if not isinstance(fnames, (list, tuple)):
        raise ValueError("Must provide 3 peer files (one per trace) as list or tuple.")

    component_list = []
    component_keys = []
    dts = []
    for fname in fnames:
        if isinstance(fname, io.StringIO):
            fname.seek(0, 0)
            text = fname.read()
        else:
            with open(fname, "r") as f:
                text = f.read()

        component_keys.append(peer_direction_exec.search(text).groups()[0])
        npts_header = int(peer_npts_exec.search(text).groups()[0])
        dt = float(peer_dt_exec.search(text).groups()[0])
        dts.append(dt)
        amplitude = np.empty((npts_header,), dtype=np.float32)
        idx = 0
        for group in peer_sample_exec.finditer(text):
            sample, = group.groups()
            amplitude[idx] = float(sample)
            idx += 1
        Check_npts(npts_header, idx)
        component_list.append(Time_Series_CUDA(amplitude, dt_in_seconds=dt))

    for idx, dt in enumerate(dts):
        if dt != dts[0]:
            raise ValueError("All time steps must be equal across PEER components.")

    # Organize components - vertical
    orientation_is_numeric = False
    try:
        vt_id = component_keys.index("UP")
        orientation_is_numeric = True
    except ValueError:
        try:
            vt_id = component_keys.index("VER")
            orientation_is_numeric = True
        except ValueError:
            for vt_id, _key in enumerate(component_keys):
                if _key[-1].lower() == "z":
                    break
            else:
                raise ValueError("Components in header are not recognized.")

    vt = component_list[vt_id]
    del component_list[vt_id], component_keys[vt_id]

    # Organize horizontals
    if orientation_is_numeric:
        component_keys_abs = np.array(component_keys, dtype=int)
        component_keys_rel = component_keys_abs.copy()
        component_keys_rel[component_keys_abs > 180] -= 360
        ns_id = np.argmin(np.abs(component_keys_rel))
        ns = component_list[ns_id]
        ew_id = np.argmax(np.abs(component_keys_rel))
        ew = component_list[ew_id]
        del component_list, component_keys
    else:
        for _id, _key in enumerate(component_keys):
            if _key[-1] == "N":
                ns_id = _id
                ns = component_list[ns_id]
            elif _key[-1] == "E":
                ew = component_list[_id]
            else:
                raise ValueError("Components in header are not recognized.")

    if degrees_from_north is None:
        degrees_from_north = float(component_keys_abs[ns_id])
        degrees_from_north = float(degrees_from_north - 360 * (degrees_from_north // 360))

    npts = [component.n_samples for component in [ns, ew, vt]]
    min_n = min(npts)
    ns.amplitude = ns.amplitude[:min_n]
    ew.amplitude = ew.amplitude[:min_n]
    vt.amplitude = vt.amplitude[:min_n]

    if USE_CUPY:
        ns = to_gpu_timeseries(ns)
        ew = to_gpu_timeseries(ew)
        vt = to_gpu_timeseries(vt)

    meta = {"file name(s)": [str(fname) for fname in fnames]}
    return SeismicRecording3C(ns, ew, vt, degrees_from_north=degrees_from_north, meta=meta)


def Read_SAF(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if isinstance(fnames, (list, tuple)):
        msg = f"Only 1 saf file allowed; {len(fnames)} provided. "
        raise ValueError(msg)

    if isinstance(fnames, io.StringIO):
        fname = fnames
        fname.seek(0, 0)
        text = fname.read()
    else:
        fname = fnames
        with open(fname, "r", encoding="utf-8", errors="surrogateescape") as f:
            text = f.read()

    _ = saf_version_exec.search(text).groups()[0]

    npts_header = int(saf_npts_exec.search(text).groups()[0])
    dt = 1.0 / float(saf_fs_exec.search(text).groups()[0])
    v_ch = int(saf_v_ch_exec.search(text).groups()[0])
    n_ch = int(saf_n_ch_exec.search(text).groups()[0])
    e_ch = int(saf_e_ch_exec.search(text).groups()[0])

    if degrees_from_north is None:
        try:
            north_rot = float(saf_north_rot_exec.search(text).groups()[0])
        except Exception:
            msg = f"The provided saf file {fname} does not include the NORTH_ROT keyword, assuming equal to zero."
            warnings.warn(msg, UserWarning)
            degrees_from_north = 0.0
        else:
            if n_ch == 1:
                degrees_from_north = north_rot
            elif e_ch == 1:
                degrees_from_north = north_rot + 90.0
            else:
                msg = f"The provided saf file {fname} is not properly formatted. CH1 must be vertical; CH2 & CH3 the horizontals."
                raise ValueError(msg)

    matches = list(saf_row_exec.finditer(text))
    if len(matches) != npts_header:
        Check_npts(npts_header, len(matches))

    all_groups = [m.groups() for m in matches]

    vt_np = np.fromiter((float(g[v_ch]) for g in all_groups), dtype=np.float32, count=npts_header)
    ns_np = np.fromiter((float(g[n_ch]) for g in all_groups), dtype=np.float32, count=npts_header)
    ew_np = np.fromiter((float(g[e_ch]) for g in all_groups), dtype=np.float32, count=npts_header)

    Check_npts(npts_header, vt_np.size)
    Check_npts(npts_header, ns_np.size)
    Check_npts(npts_header, ew_np.size)

    try:
        if USE_CUPY:
            vt_xp = to_xp_array(vt_np)
            ns_xp = to_xp_array(ns_np)
            ew_xp = to_xp_array(ew_np)
        else:
            vt_xp = vt_np
            ns_xp = ns_np
            ew_xp = ew_np
    except Exception as e:
        logger.warning("Failed to convert SAF arrays to GPU arrays (CuPy). Falling back to CPU arrays. Error: %s", e)
        vt_xp = vt_np
        ns_xp = ns_np
        ew_xp = ew_np

    try:
        vt_ts = Time_Series_CUDA(vt_xp, dt_in_seconds=dt)
        ns_ts = Time_Series_CUDA(ns_xp, dt_in_seconds=dt)
        ew_ts = Time_Series_CUDA(ew_xp, dt_in_seconds=dt)
    except Exception as construct_exc:
        logger.debug("Direct Time_Series_CUDA construction with xp arrays failed: %s. Falling back to NumPy construction.", construct_exc)
        vt_ts = Time_Series_CUDA(to_host_array(vt_xp), dt_in_seconds=dt)
        ns_ts = Time_Series_CUDA(to_host_array(ns_xp), dt_in_seconds=dt)
        ew_ts = Time_Series_CUDA(to_host_array(ew_xp), dt_in_seconds=dt)
        if USE_CUPY:
            vt_ts = to_gpu_timeseries(vt_ts)
            ns_ts = to_gpu_timeseries(ns_ts)
            ew_ts = to_gpu_timeseries(ew_ts)

    meta = {"file name(s)": str(fname)}
    if degrees_from_north is None:
        degrees_from_north = 0.0

    return SeismicRecording3C(ns_ts, ew_ts, vt_ts, degrees_from_north=degrees_from_north, meta=meta)


READ_FUNCTION_DICT = {
    "mseed": Read_mseed,
    "saf": Read_SAF,
    "minishark": Read_Minishark,
    "sac": Read_SAC,
    "gcf": Read_GCF,
    "peer": Read_PEER
}


def Read_Single(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    logger.info(f"Attempting to read {fnames!r}")
    exceptions = []
    forms_to_try = [fnames]

    try:
        if isinstance(fnames, (list, tuple)) and len(fnames) == 1:
            inner = fnames[0]
            if isinstance(inner, (str, pathlib.Path, io.BytesIO)):
                forms_to_try.append(inner)
    except Exception:
        logger.debug("Exception while trying to detect unwrapped form", exc_info=True)

    #------------------------------------------------
    # Ensure a globally-visible fill_gaps exists
    #------------------------------------------------
    APPLY_FILL_GAPS_LOCAL = globals().get("APPLY_FILL_GAPS", False)
    FILL_MAX_GAP = globals().get("FILL_MAX_GAP", 10)

    filler_func = None

    if filler_func is not None and not hasattr(builtins, "fill_gaps"):
        builtins.fill_gaps = filler_func

    if not hasattr(builtins, "fill_gaps"):
        def Fallback_Fill_Gaps(arr, max_gap_local=FILL_MAX_GAP):
            try:
                cp_local = globals().get("cp", None)
                is_cupy = (cp_local is not None) and isinstance(arr, getattr(cp_local, "ndarray", type(None)))
                if is_cupy:
                    a = cp_local.asnumpy(arr).astype(float)
                else:
                    a = np.asarray(arr, dtype=float)
            except Exception:
                a = np.asarray(list(arr), dtype=float)

            if a.size == 0:
                return a
            nans = np.isnan(a)
            if not np.any(nans):
                return a
            idx = np.arange(a.size)
            valid = ~nans
            if valid.sum() < 2:
                return np.nan_to_num(a, nan=0.0)
            a[nans] = np.interp(idx[nans], idx[valid], a[valid])
            return a

        builtins.fill_gaps = Fallback_Fill_Gaps

    #--------------------------
    # Try readers
    #--------------------------
    for ftype, read_function in READ_FUNCTION_DICT.items():
        for form in forms_to_try:
            if ftype == "peer":
                if not (isinstance(form, (list, tuple)) and len(form) == 3):
                    logger.debug("Skipping 'peer' reader for form type %s (not list-of-3)", type(form).__name__)
                    continue
            try:
                srecording_3c = read_function(form,
                                              obspy_read_kwargs=obspy_read_kwargs,
                                              degrees_from_north=degrees_from_north)
                if srecording_3c is None:
                    raise ValueError(f"Reader {ftype} returned None")
                try:
                    if hasattr(srecording_3c, "__len__") and len(srecording_3c) == 0:
                        raise ValueError(f"Reader {ftype} returned empty result")
                except TypeError:
                    pass
            except Exception as e:
                logger.debug("Tried reading as %s with form=%s, got exception: %s",
                             ftype, type(form).__name__, e, exc_info=True)
                exceptions.append((ftype, type(form).__name__, str(e)))
                continue
            else:
                logger.info("File type identified as %s.", ftype)

                if APPLY_FILL_GAPS_LOCAL:
                    try:
                        filler = builtins.fill_gaps
                        for comp_name in ("ns", "ew", "vt"):
                            comp = getattr(srecording_3c, comp_name, None)
                            if comp is None:
                                continue
                            for attr in ("amplitude", "values", "data", "array"):
                                if hasattr(comp, attr):
                                    orig = getattr(comp, attr)
                                    try:
                                        filled = filler(orig, FILL_MAX_GAP)
                                    except TypeError:
                                        filled = filler(orig)
                                    try:
                                        cp_local = globals().get("cp", None)
                                        if (cp_local is not None) and isinstance(orig, getattr(cp_local, "ndarray", type(None))):
                                            filled = cp_local.asarray(filled)
                                    except Exception:
                                        pass
                                    try:
                                        setattr(comp, attr, filled)
                                    except Exception:
                                        pass
                                    break
                    except Exception as ex_fill:
                        warnings.warn(f"apply_fill_gaps failed: {ex_fill}")

                return srecording_3c

    msg_lines = [
        "File format not recognized.",
        f"Supported readers: {list(READ_FUNCTION_DICT.keys())}",
        "Tried the following reader attempts (format, form_type, exception):"
    ]
    for ftype, form_type, exc_str in exceptions:
        msg_lines.append(f" - {ftype} | {form_type} | {exc_str}")

    peer_msgs = [e for e in exceptions if e[0] == "peer"]
    if peer_msgs:
        msg_lines.append("Note: 'peer' reader requires a list/tuple of 3 files (one per trace).")
        msg_lines.append("If you have a single multi-trace MiniSEED file, provide its path (string),")
        msg_lines.append("or let the caller pass the single path without wrapping inside another list.")

    full_msg = "\n".join(msg_lines)
    raise ValueError(full_msg)


def Read(
    fnames,
    obspy_read_kwargs=None,
    degrees_from_north=None
):
    if not isinstance(fnames, (list, tuple)):
        warnings.warn("fnames should be iterable of str or iterable of iterable of str; wrapping single entry.")
        fnames = [fnames]

    if isinstance(obspy_read_kwargs, (dict, type(None))):
        read_kwargs_iter = itertools.repeat(obspy_read_kwargs)
    else:
        read_kwargs_iter = obspy_read_kwargs

    if isinstance(degrees_from_north, (float, int, type(None))):
        degrees_from_north_iter = itertools.repeat(degrees_from_north)
    else:
        degrees_from_north_iter = degrees_from_north

    seismic_recordings = []
    for fname, read_kwargs, df_north in zip(fnames, read_kwargs_iter, degrees_from_north_iter):
        logger.debug("Calling Read_Single for entry of type %s", type(fname).__name__)
        sr = Read_Single(fname, obspy_read_kwargs=read_kwargs, degrees_from_north=df_north)
        seismic_recordings.append(sr)
    return seismic_recordings



####################################
# PSD (Power Spectral Density)
####################################
def _to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


class Psd:

    @staticmethod
    def _check_input(
        value: Iterable,
        name: str
    ):
        try:
            arr = xp.asarray(value, dtype=xp.float64)
        except Exception as exc:
            msg = f"{name} must be castable to array of doubles, not {type(value)}."
            raise TypeError(msg) from exc
        try:
            has_nan = bool(xp.any(xp.isnan(arr)))
        except Exception:
            arr_host = _to_host_array(arr)
            if np.isnan(arr_host).any():
                raise ValueError(f"{name} may not contain nan.")
            return xp.asarray(arr_host, dtype=xp.float64)
        if has_nan:
            raise ValueError(f"{name} may not contain nan.")
        try:
            has_negative = bool(xp.any(arr < 0))
        except Exception:
            arr_host = _to_host_array(arr)
            if (arr_host < 0).any():
                raise ValueError(f"{name} must be >= 0.")
            return xp.asarray(arr_host, dtype=xp.float64)
        if has_negative:
            raise ValueError(f"{name} must be >= 0.")
        return arr


    def __init__(
        self,
        frequency,
        amplitude,
        meta: Optional[dict] = None
    ):
        self.frequency = self._check_input(frequency, "frequency")
        self.amplitude = self._check_input(amplitude, "amplitude")
        if self.frequency.size != self.amplitude.size:
            msg = f"Length of amplitude {self.amplitude.size} and length of frequency {self.frequency.size} must agree."
            raise ValueError(msg)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()

    def to_numpy(self):
        return _to_host_array(self.frequency), _to_host_array(self.amplitude)

    def is_similar(
        self,
        other: "Psd",
        atol: float = 1e-9,
        rtol: float = 0.0
    ) -> bool:
        if not isinstance(other, Psd):
            return False
        if self.frequency.size != other.frequency.size:
            return False
        f1 = _to_host_array(self.frequency)
        f2 = _to_host_array(other.frequency)
        if not np.allclose(f1, f2, atol=atol, rtol=rtol):
            return False
        return True


    def __repr__(self):
        freq_summary = f"len={int(self.frequency.size)}"
        amp_summary = f"len={int(self.amplitude.size)}"
        backend = "cupy" if USE_CUPY else "numpy"
        return f"Psd(frequency={freq_summary}, amplitude={amp_summary}, backend={backend})"

    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, Psd):
            return False
        if not self.is_similar(other):
            return False
        a1 = _to_host_array(self.amplitude)
        a2 = _to_host_array(other.amplitude)
        return np.allclose(a1, a2)



####################################
# HVSR Azimuthal
####################################
def to_host_if_cupy(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def Fill_Curve(
    xp_arr,
    max_gap: int = 10
):
    try:
        filled = Fill_Gaps_Linear(xp_arr, max_gap=max_gap)
        if USE_CUPY:
            if isinstance(filled, cp.ndarray):
                return filled
            else:
                return cp.asarray(filled)
        else:
            return np.asarray(filled)
    except Exception as e:
        logger.debug("Fill_Gaps_Linear failed: %s", e)
        return to_host_if_cupy(xp_arr) if not USE_CUPY else (xp_arr if isinstance(xp_arr, cp.ndarray) else cp.asarray(xp_arr))


def Call_Find_Peak_Bounded(
    freq,
    amp,
    search_range_in_hz=None,
    find_peaks_kwargs=None
):
    try:
        return HvsrCurve._find_peak_bounded(freq, amp,
                                            search_range_in_hz=search_range_in_hz,
                                            find_peaks_kwargs=find_peaks_kwargs)
    except Exception:
        try:
            freq_h = to_host_if_cupy(freq)
            amp_h = to_host_if_cupy(amp)
            return HvsrCurve._find_peak_bounded(freq_h, amp_h,
                                                search_range_in_hz=search_range_in_hz,
                                                find_peaks_kwargs=find_peaks_kwargs)
        except Exception as e:
            logger.debug("peak finding failed after fallback: %s", e)
            return (None, None)


class HvsrAzimuthal():

    @staticmethod
    def Check_Input(
        hvsr,
        azimuth
    ):
        if not isinstance(hvsr, HvsrTraditional):
            msg = "each hvsr must be an instance of HvsrTraditional; "
            msg += f"not {type(hvsr)}."
            raise TypeError(msg)

        azimuth = float(azimuth)
        if (azimuth < 0) or (azimuth > 180):
            raise ValueError(f"azimuth is {azimuth}; azimuth must be between 0 and 180.")

        return (hvsr, azimuth)

    def __init__(
        self,
        hvsrs: Sequence[HvsrTraditional],
        azimuths: Sequence[float],
        meta: Optional[dict] = None
    ):
        if len(hvsrs) == 0:
            raise ValueError("hvsrs must be a non-empty iterable of HvsrTraditional")

        self.hvsrs = []
        self.azimuths = []
        ex_hvsr = hvsrs[0]
        for _idx, (hvsr, azimuth) in enumerate(zip(hvsrs, azimuths)):
            hvsr, azimuth = self.Check_Input(hvsr, azimuth)
            if not ex_hvsr.is_similar(hvsr):
                raise ValueError(f"All HvsrTraditional must be similar; hvsrs[0] is not similar to hvsrs[{_idx}]")
            self.hvsrs.append(hvsr)
            self.azimuths.append(float(azimuth))

        self.meta = dict(meta) if isinstance(meta, dict) else dict()
        self.Update_Peaks_Bounded()

    @property
    def Search_Range_in_Hz(self):
        return self.hvsrs[0].Search_Range_in_Hz

    @property
    def Find_Peaks_Kwargs(self):
        return self.hvsrs[0].Find_Peaks_Kwargs

    def Update_Peaks_Bounded(
        self,
        search_range_in_hz=(None, None),
        find_peaks_kwargs=None
    ):
        self.meta["search_range_in_hz"] = tuple(search_range_in_hz)
        self.meta["find_peaks_kwargs"] = None if find_peaks_kwargs is None else dict(find_peaks_kwargs)
        for hvsr in self.hvsrs:
            hvsr.Update_Peaks_Bounded(search_range_in_hz=search_range_in_hz,
                                      find_peaks_kwargs=find_peaks_kwargs)

    @property
    def Peak_Frequencies(self):
        return [hvsr.Peak_Frequencies for hvsr in self.hvsrs]

    @property
    def Peak_Amplitudes(self):
        return [hvsr.Peak_Amplitudes for hvsr in self.hvsrs]

    @property
    def n_azimuths(self):
        return len(self.azimuths)

    def Compute_Statistical_Weights(self):
        weights = []
        n_azimuths = len(self.azimuths)
        for hvsr in self.hvsrs:
            n_valid_peaks = int(xp.sum(hvsr.valid_peak_boolean_mask))
            if n_valid_peaks > 0:
                weights.extend([1.0 / (n_azimuths * n_valid_peaks)] * n_valid_peaks)
        if len(weights) == 0:
            return xp.asarray([], dtype=xp.float64)
        return xp.asarray(weights, dtype=xp.float64)

    def Mean_fn_Frequency(
        self,
        distribution="lognormal"
    ):
        weights = self.Compute_Statistical_Weights()
        values = xp.asarray(Flatten_List(self.Peak_Frequencies))
        return Generalized_Weighted_Mean(distribution=distribution,
                                         weights=weights,
                                         values=values)

    def Mean_fn_Amplitude(
        self,
        distribution="lognormal"
    ):
        weights = self.Compute_Statistical_Weights()
        values = xp.asarray(Flatten_List(self.Peak_Amplitudes))
        return Generalized_Weighted_Mean(distribution=distribution,
                                         weights=weights,
                                         values=values)

    def cov_fn(
        self,
        distribution="lognormal"
    ):
        freqs = xp.asarray(Flatten_List(self.Peak_Frequencies), dtype=xp.float64)
        amps = xp.asarray(Flatten_List(self.Peak_Amplitudes), dtype=xp.float64)
        weights = self.Compute_Statistical_Weights().astype(xp.float64)

        if freqs.size == 0 or amps.size == 0 or weights.size == 0:
            return xp.full((2, 2), xp.nan, dtype=xp.float64)

        if distribution == "lognormal":
            freqs = xp.log(freqs)
            amps = xp.log(amps)
        elif distribution != "normal":
            raise NotImplementedError(f"distribution type {distribution} not recognized.")

        w_sum = xp.sum(weights)
        if w_sum == 0:
            return xp.full((2, 2), xp.nan, dtype=xp.float64)
        w = weights / w_sum

        mu_f = xp.sum(w * freqs)
        mu_a = xp.sum(w * amps)
        cov_ff_num = xp.sum(w * (freqs - mu_f) * (freqs - mu_f))
        cov_fa_num = xp.sum(w * (freqs - mu_f) * (amps - mu_a))
        cov_aa_num = xp.sum(w * (amps - mu_a) * (amps - mu_a))

        sum_w2 = xp.sum(w * w)
        if sum_w2 == 0:
            return xp.full((2, 2), xp.nan, dtype=xp.float64)
        n_eff = 1.0 / sum_w2
        if n_eff > 1.0:
            cov_ff = cov_ff_num * (1.0 / (1.0 - sum_w2))
            cov_fa = cov_fa_num * (1.0 / (1.0 - sum_w2))
            cov_aa = cov_aa_num * (1.0 / (1.0 - sum_w2))
        else:
            cov_ff = xp.nan
            cov_fa = xp.nan
            cov_aa = xp.nan
        cov = xp.array([[cov_ff, cov_fa], [cov_fa, cov_aa]], dtype=xp.float64)
        return cov

    def std_fn_frequency(
        self,
        distribution="lognormal"
    ):
        weights = self.Compute_Statistical_Weights()
        values = xp.asarray(Flatten_List(self.Peak_Frequencies))
        return Weighted_Standard_Deviation(distribution=distribution,
                                           weights=weights,
                                           values=values,
                                           denominator="cheng")

    def std_fn_amplitude(
        self,
        distribution="lognormal"
    ):
        weights = self.Compute_Statistical_Weights()
        values = xp.asarray(Flatten_List(self.Peak_Amplitudes))
        return Weighted_Standard_Deviation(distribution=distribution,
                                           weights=weights,
                                           values=values,
                                           denominator="cheng")

    # @property
    # def amplitude(self):
    #     return [hvsr.amplitude for hvsr in self.hvsrs]

    @property
    def amplitude(self):
        try:
            return np.vstack([hvsr.amplitude for hvsr in self.hvsrs])
        except Exception:
            return [hvsr.amplitude for hvsr in self.hvsrs]

    @property
    def frequency(self):
        return self.hvsrs[0].frequency

    def mean_curve_by_azimuth(
        self,
        distribution="lognormal"
    ):
        curves = []
        for hvsr in self.hvsrs:
            c = hvsr.mean_curve(distribution=distribution)
            if c is None:
                continue
            c_filled = Fill_Curve(c, max_gap=10)
            curves.append(c_filled)
        if len(curves) == 0:
            return xp.empty((0,0), dtype=np.float32) if not USE_CUPY else cp.empty((0,0), dtype=cp.float32)
        stacked = xp.vstack([xp.asarray(c) for c in curves])
        return stacked

    def mean_curve_peak_by_azimuth(
        self,
        distribution="lognormal"
    ):
        frqs = xp.empty(self.n_azimuths, dtype=xp.float64)
        amps = xp.empty(self.n_azimuths, dtype=xp.float64)
        for idx, hvsr in enumerate(self.hvsrs):
            f_peak, a_peak = hvsr.mean_curve_peak(distribution=distribution)
            frqs[idx] = xp.asarray(f_peak, dtype=xp.float64)
            amps[idx] = xp.asarray(a_peak, dtype=xp.float64)
        return frqs, amps

    def mean_curve(
        self,
        distribution="lognormal"
    ):
        stacked = self.mean_curve_by_azimuth(distribution=distribution)
        if stacked is None:
            return xp.full_like(self.frequency, xp.nan)

        try:
            n_az = int(stacked.shape[0])
        except Exception:
            return xp.full_like(self.frequency, xp.nan)

        if n_az == 0:
            return xp.full_like(self.frequency, xp.nan)

        try:
            mean_curve = xp.nanmean(stacked, axis=0)
        except Exception:
            mask = ~xp.isnan(stacked)
            stacked_zero = xp.where(mask, stacked, 0.0)
            sum_valid = xp.sum(stacked_zero, axis=0)
            count_valid = xp.sum(mask, axis=0)
            mean_curve = xp.where(count_valid == 0, xp.nan, sum_valid / count_valid)

        if distribution == "lognormal":
            with_nan = xp.where(stacked > 0.0, stacked, xp.nan)
            try:
                logmean = xp.nanmean(xp.log(with_nan), axis=0)
                mean_curve = xp.exp(logmean)
            except Exception:
                try:
                    stacked_host = np.asarray(xp.asnumpy(stacked)) if hasattr(xp, "asnumpy") else np.asarray(stacked)
                    stacked_host = np.where(stacked_host > 0.0, stacked_host, np.nan)
                    logmean_host = np.nanmean(np.log(stacked_host), axis=0)
                    mean_curve_host = np.exp(logmean_host)
                    mean_curve = xp.asarray(mean_curve_host)
                except Exception:
                    pass
        try:
            mean_curve = Fill_Curve(mean_curve, max_gap=10)
        except Exception:
            pass
        try:
            if hasattr(xp, "asarray") and not isinstance(mean_curve, getattr(xp, "ndarray", type(None))):
                mean_curve = xp.asarray(mean_curve)
        except Exception:
            mean_curve = np.asarray(mean_curve)
        return mean_curve

    """
    def std_curve(
        self,
        distribution="lognormal"
    ):
        stacked = self.mean_curve_by_azimuth(distribution=distribution)
        if stacked.shape[0] <= 1:
            raise ValueError("The standard deviation of the mean curve is not defined for a single azimuth.")
        try:
            std_curve = xp.nanstd(stacked, axis=0, ddof=1)
        except Exception:
            mask = ~xp.isnan(stacked)
            stacked_zero = xp.where(mask, stacked, 0.0)
            sum_valid = xp.sum(stacked_zero, axis=0)
            count_valid = xp.sum(mask, axis=0)
            mean_vals = xp.where(count_valid == 0, xp.nan, sum_valid / count_valid)
            dev2 = xp.where(mask, (stacked - mean_vals) ** 2, 0.0)
            sum_dev2 = xp.sum(dev2, axis=0)
            std_curve = xp.where(count_valid <= 1, xp.nan, xp.sqrt(sum_dev2 / (count_valid - 1)))
        return std_curve
    """

    def std_curve(
        self,
        distribution="lognormal"
    ):
        stacked = self.mean_curve_by_azimuth(distribution=distribution)
        if stacked.shape[0] <= 1:
            raise ValueError(
                "The standard deviation of the mean curve is not defined for a single azimuth."
            )
        if USE_CUPY:
            stacked_host = cp.asnumpy(stacked)
        else:
            stacked_host = np.asarray(stacked, dtype=float)

        stacked_host = np.asarray(stacked_host, dtype=float)

        if distribution.lower() in ("lognormal", "log-normal"):
            log_stack = np.where(stacked_host > 0.0, np.log(stacked_host), np.nan)
            std_log = np.nanstd(log_stack, axis=0, ddof=1)
            med = np.nanmedian(log_stack, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(log_stack - med), axis=0)
            bad = (~np.isfinite(std_log)) | (std_log > 2.0)
            std_log = np.where(bad, mad, std_log)
            return cp.asarray(std_log) if USE_CUPY else std_log

        std_curve = np.nanstd(stacked_host, axis=0, ddof=1)
        med = np.nanmedian(stacked_host, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(stacked_host - med), axis=0)
        bad = (~np.isfinite(std_curve)) | (std_curve > 2.0 * np.nanmedian(std_curve))
        std_curve = np.where(bad, mad, std_curve)

        return cp.asarray(std_curve) if USE_CUPY else std_curve


    def nth_std_curve(
        self,
        n,
        distribution="lognormal"
    ):
        mean_c = self.mean_curve(distribution=distribution)
        std_c = self.std_curve(distribution=distribution)
        return Standard_Score_Shift(n=n,
                                    distribution=distribution,
                                    mean=mean_c,
                                    std=std_c)

    def nth_std_fn_frequency(
        self,
        n,
        distribution="lognormal"
    ):
        return Standard_Score_Shift(n=n,
                                    distribution=distribution,
                                    mean=self.Mean_fn_Frequency(distribution=distribution),
                                    std=self.std_fn_frequency(distribution=distribution))

    def nth_std_fn_amplitude(
        self,
        n,
        distribution="lognormal"
    ):
        return Standard_Score_Shift(n=n,
                                    distribution=distribution,
                                    mean=self.Mean_fn_Amplitude(distribution=distribution),
                                    std=self.std_fn_amplitude(distribution=distribution))

    def mean_curve_peak(
        self,
        distribution="lognormal"
    ):
        try:
            amplitude = self.mean_curve(distribution)
        except Exception as ex:
            warnings.warn(f"Failed to compute mean curve: {ex}. Returning (nan, nan).")
            return np.nan, np.nan

        def to_host(arr):
            if globals().get("to_host_if_cupy", None) is not None:
                try:
                    return to_host_if_cupy(arr)
                except Exception:
                    pass
            try:
                if hasattr(arr, "dtype") and str(type(arr)).find("cupy") >= 0:
                    return __import__("cupy").asnumpy(arr)
            except Exception:
                pass
            return np.asarray(arr)

        try:
            amplitude = Fill_Curve(amplitude, max_gap=10)
        except Exception:
            pass

        amp_host = to_host(amplitude)
        amp_host = np.nan_to_num(amp_host, nan=0.0, posinf=0.0, neginf=0.0)
        amp_host = np.clip(amp_host, 0.0, 1e2)

        if amp_host.size == 0 or np.all(np.isnan(amp_host)) or np.all(~np.isfinite(amp_host)) or np.all(
                amp_host <= 0.0):
            warnings.warn("Mean curve does not contain positive finite amplitude values for peak finding; returning (nan, nan).")
            return np.nan, np.nan

        freq = getattr(self, "frequency", None)
        if freq is None:
            warnings.warn("No frequency vector available; returning (nan, nan).")
            return np.nan, np.nan

        try:
            if hasattr(xp, "asarray") and not isinstance(freq, getattr(xp, "ndarray", type(None))):
                freq_xp = xp.asarray(freq)
            else:
                freq_xp = freq
        except Exception:
            freq_xp = np.asarray(freq)

        floor = 1e-3
        try:
            freq_clipped = xp.clip(freq_xp, floor, None)
        except Exception:
            freq_host = to_host(freq_xp)
            freq_host = np.clip(freq_host, floor, None)
            try:
                freq_clipped = xp.asarray(freq_host)
            except Exception:
                freq_clipped = freq_host

        try:
            f_peak_raw, a_peak_raw = Call_Find_Peak_Bounded(freq_clipped, amplitude,
                                                            search_range_in_hz=getattr(self, "Search_Range_in_Hz", None),
                                                            find_peaks_kwargs=getattr(self, "Find_Peaks_Kwargs", None))
        except Exception as ex:
            warnings.warn(f"Peak-finding routine failed: {ex}. Returning (nan, nan).")
            return np.nan, np.nan

        def _to_float_host(x):
            try:
                return float(to_host(x).item()) if hasattr(to_host(x), "item") else float(to_host(x)[0])
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return np.nan

        a_val = _to_float_host(a_peak_raw)
        f_val = None

        try:
            if isinstance(f_peak_raw, (int, np.integer)):
                freq_host = to_host(freq_clipped)
                if 0 <= int(f_peak_raw) < freq_host.size:
                    f_val = float(freq_host[int(f_peak_raw)])
                else:
                    f_val = np.nan
            else:
                f_val = _to_float_host(f_peak_raw)
                if np.isnan(f_val) or f_val <= 0.0:
                    try:
                        idx_try = np.asarray(to_host(f_peak_raw)).ravel()
                        if idx_try.size > 0:
                            idx0 = int(idx_try[0])
                            freq_host = to_host(freq_clipped)
                            if 0 <= idx0 < freq_host.size:
                                f_val = float(freq_host[idx0])
                            else:
                                f_val = np.nan
                    except Exception:
                        f_val = np.nan
        except Exception:
            f_val = np.nan

        if not np.isfinite(f_val) or not np.isfinite(a_val) or f_val <= 0.0:
            warnings.warn("Mean curve peak not found or invalid; returning (nan, nan).")
            return np.nan, np.nan

        return float(f_val), float(a_val)


    def is_similar(self, other):
        if not isinstance(other, HvsrAzimuthal):
            return False
        if len(self.hvsrs) != len(other.hvsrs):
            return False
        if not self.hvsrs[0].is_similar(other.hvsrs[0]):
            return False
        for a, b in zip(self.azimuths, other.azimuths):
            if abs(a - b) > 0.1:
                return False
        return True

    def __eq__(self, other):
        if not self.is_similar(other):
            return False
        for self_hvsr, other_hvsr in zip(self.hvsrs, other.hvsrs):
            if self_hvsr != other_hvsr:
                return False
        return True

    def __str__(self):
        return f"HvsrAzimuthal at {id(self)}"

    def __repr__(self):
        return f"HvsrAzimuthal(n_azimuths={self.n_azimuths}, meta={self.meta})"



####################################
# HVSR Diffuse Field
####################################
def _to_py_float(xp_obj):
    try:
        return float(xp_obj.item())
    except Exception:
        try:
            if USE_CUPY:
                return float(xp.asnumpy(xp_obj))
        except Exception:
            pass
        try:
            return float(np.asarray(xp_obj))
        except Exception:
            raise TypeError(f"Could not convert object {type(xp_obj)} to Python float.")


class HvsrDiffuseField(HvsrCurve):

    def mean_curve(
        self,
        distribution: Optional[object] = None,
        axis: int = 0,
        keepdims: bool = False
    ) -> xp.ndarray:
        amp = self.amplitude
        if amp.ndim == 1:
            return xp.asarray(amp)
        nanmean = getattr(xp, "nanmean", None)
        if nanmean is not None:
            try:
                return nanmean(amp, axis=axis, keepdims=keepdims)
            except Exception:
                pass
        valid_mask = ~xp.isnan(amp)
        sum_valid = xp.sum(xp.where(valid_mask, amp, 0.0).astype(xp.float64), axis=axis, keepdims=keepdims)
        count_valid = xp.sum(valid_mask, axis=axis, keepdims=keepdims).astype(xp.float64)
        mean = xp.where(count_valid == 0, xp.nan, sum_valid / count_valid)
        return mean.astype(amp.dtype, copy=False) if not keepdims else mean


    def mean_curve_peak(
        self,
        distribution: Optional[object] = None,
        search_range_in_hz: Tuple[Optional[float],
        Optional[float]] = (None, None),
        find_peaks_kwargs: Optional[dict] = None,
        axis: int = 0
    ) -> Tuple[float, float]:
        mean_amp = self.mean_curve(distribution=distribution, axis=axis, keepdims=False)
        mean_amp = xp.ravel(mean_amp)
        try:
            any_finite = bool(xp.any(xp.isfinite(mean_amp)))
        except Exception:
            any_finite = bool(np.any(np.isfinite(np.asarray(mean_amp))))
        if not any_finite:
            raise ValueError("Mean curve contains no finite values; cannot determine peak.")
        try:
            f_peak, a_peak = HvsrCurve._find_peak_bounded(self.frequency, mean_amp,
                                                          search_range_in_hz=search_range_in_hz,
                                                          find_peaks_kwargs=find_peaks_kwargs)
        except Exception as exc:
            logger.debug("Host peak-finder raised exception: %s", exc)
            f_peak = a_peak = None
        if f_peak is not None and a_peak is not None:
            try:
                return (float(f_peak), float(a_peak))
            except Exception:
                return (_to_py_float(f_peak), _to_py_float(a_peak))
        safe_amp = xp.where(xp.isnan(mean_amp), -xp.inf, mean_amp)
        try:
            argmax_scalar = xp.argmax(safe_amp)
            try:
                idx = int(argmax_scalar.item())
            except Exception:
                idx = int(np.asarray(argmax_scalar).tolist())
        except Exception as exc:
            logger.debug("GPU argmax failed: %s", exc)
            raise ValueError("Unable to compute peak index on mean curve.") from exc
        try:
            freq_xp = self.frequency[idx]
            amp_xp = safe_amp[idx]
            freq_val = _to_py_float(freq_xp)
            amp_val = _to_py_float(amp_xp)
        except Exception as exc:
            logger.debug("Failed converting peak values to Python floats: %s", exc)
            raise ValueError("Peak found but could not convert values to Python floats.") from exc
        if not math.isfinite(amp_val) or amp_val == -np.inf:
            raise ValueError("Mean curve has no finite amplitude peak (all values invalid).")
        return (float(freq_val), float(amp_val))



####################################
# HVSR Geopsy
####################################
class HvsrGeopsy:

    def __init__(
        self,
        frequency,
        mean_curve,
        std_curve
    ):
        freq_x = xp.asarray(frequency, dtype=xp.float64)
        mean_x = xp.asarray(mean_curve, dtype=xp.float64)
        std_x = xp.asarray(std_curve, dtype=xp.float64)
        if freq_x.ndim != 1 or mean_x.ndim != 1 or std_x.ndim != 1:
            raise ValueError("frequency, mean_curve and std_curve must be 1-D arrays.")
        if freq_x.size != mean_x.size or freq_x.size != std_x.size:
            raise ValueError("frequency, mean_curve and std_curve must have same length.")
        self.frequency = freq_x
        self._mean_curve = mean_x
        self._std_curve = std_x

    def mean_curve(self):
        return self._mean_curve

    def std_curve(self):
        return self._std_curve

    def nth_std_curve(self, n: float):
        return xp.exp(xp.log(self._mean_curve) + (float(n) * self._std_curve))

    def mean_curve_peak(
        self,
        distribution: Optional[str] = None,
        search_range_in_hz: Tuple[Optional[float], Optional[float]] = (None, None),
        find_peaks_kwargs: Optional[dict] = None
    ) -> Tuple[float, float]:
        amplitude = self.mean_curve()
        f_peak, a_peak = HvsrCurve._find_peak_bounded(self.frequency,
                                                      amplitude,
                                                      search_range_in_hz=search_range_in_hz,
                                                      find_peaks_kwargs=find_peaks_kwargs)
        if f_peak is None or a_peak is None:
            raise ValueError("Mean curve does not have a peak in the specified range.")
        try:
            return float(f_peak), float(a_peak)
        except Exception:
            f_val = float(xp.asnumpy(f_peak)) if USE_CUPY else float(f_peak)
            a_val = float(xp.asnumpy(a_peak)) if USE_CUPY else float(a_peak)
            return f_val, a_val

    def to_numpy(self):
        if USE_CUPY:
            return cp.asnumpy(self.frequency), cp.asnumpy(self._mean_curve), cp.asnumpy(self._std_curve)
        return (np.asarray(self.frequency), np.asarray(self._mean_curve), np.asarray(self._std_curve))

    @classmethod
    def from_file(
        cls,
        fname: str
    ) -> "HvsrGeopsy":
        with open(fname, "r") as f:
            text = f.read()
        frequency = []
        mean_curve = []
        minus_one_std_curve = []
        for group in geopsy_line_exec.finditer(text):
            _f, _a, _m = group.groups()
            frequency.append(float(_f))
            mean_curve.append(float(_a))
            minus_one_std_curve.append(float(_m))
        frequency_np = np.array(frequency, dtype=float)
        mean_np = np.array(mean_curve, dtype=float)
        minus_one_std_np = np.array(minus_one_std_curve, dtype=float)
        std_np = np.log(mean_np) - np.log(minus_one_std_np)
        return cls(frequency_np, mean_np, std_np)

    def __repr__(self):
        backend = "cupy" if USE_CUPY else "numpy"
        return f"HvsrGeopsy(len={int(self.frequency.size)}, backend={backend})"



####################################
# HVSR Spatial
####################################
#==========================
# Helper conversions
#==========================
def _to_xp(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if not (isinstance(a, xp.ndarray)) else (a.astype(dtype) if dtype else a)

def _to_numpy(a):
    if USE_CUPY:
        return cp.asnumpy(a)
    return np.asarray(a)


#===============================
# Core statistics (GPU/CPU)
#===============================
def Statistics(
    values,
    weights
):
    V = _to_xp(values, dtype=xp.float64)
    w = _to_xp(weights, dtype=xp.float64)

    if V.ndim != 2:
        raise ValueError("values must be 2D (R, J).")
    if w.ndim != 1 or w.size != V.shape[0]:
        raise ValueError("weights must be 1D with the same size as the number of rows of values.")
    if not xp.all(xp.isfinite(V)):
        raise ValueError("values contain NaN or Inf.")
    if not xp.all(xp.isfinite(w)):
        raise ValueError("weights contain NaN or Inf.")
    if xp.any(w < 0):
        raise ValueError("weights must be non-negatif.")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Sum of the weights must be > 0.")

    w = w / w_sum
    m_per_real = xp.sum(V * w[:, None], axis=0)
    mean_scalar = xp.mean(m_per_real)
    dd = 1 if m_per_real.size > 1 else 0
    std_scalar = xp.std(m_per_real, ddof=dd)
    return mean_scalar, std_scalar


#===========================
# Monte-Carlo (GPU/CPU)
#===========================
def MonteCarlo(
    generator_means,
    generator_stddevs,
    generator_weights,
    distribution_generators: str = "lognormal",
    distribution_spatial: str = "lognormal",
    n_realizations: int = 1000,
    rng: Optional[object] = None
):

    if distribution_generators not in ("normal", "lognormal"):
        raise NotImplementedError(f"dist_generators = {distribution_generators} unrecognized.")
    if distribution_spatial not in ("normal", "lognormal"):
        raise NotImplementedError(f"dist_spatial = {distribution_spatial} unrecognized.")

    mu = _to_xp(generator_means, dtype=xp.float64).ravel()
    sg = _to_xp(generator_stddevs, dtype=xp.float64).ravel()
    w  = _to_xp(generator_weights, dtype=xp.float64).ravel()

    if mu.size != sg.size or mu.size != w.size:
        raise ValueError("Length of generator_means/stddevs/weights must be the same.")
    if not xp.all(xp.isfinite(mu)) or not xp.all(xp.isfinite(sg)):
        raise ValueError("generator_means/generator_stddevs must be finite.")
    if xp.any(sg < 0):
        raise ValueError("generator_stddevs must be non-negative.")
    if not xp.all(xp.isfinite(w)):
        raise ValueError("generator_weights containa NaN/Inf.")
    if xp.any(w < 0):
        raise ValueError("generator_weights contains negative values which is not allowed.")
    if w.sum() <= 0:
        raise ValueError("Sum of generator_weights must be > 0.")

    R = int(mu.size)
    J = int(n_realizations)

    #----------------------------------------------------------
    # Sample standard normal on the correct backend
    #----------------------------------------------------------
    def Sample_Standard_Normal(shape, rng_obj):
        # CuPy backend
        if USE_CUPY:
            if isinstance(rng_obj, (int, np.integer)):
                cp.random.seed(int(rng_obj))
                out = cp.random.normal(loc=0.0, scale=1.0, size=shape)
                return out
            try:
                if cp is not None and isinstance(rng_obj, cp.random.RandomState):
                    return rng_obj.normal(loc=0.0, scale=1.0, size=shape)
            except Exception:
                pass
            try:
                if cp is not None and hasattr(rng_obj, "normal"):
                    out = rng_obj.normal(0.0, 1.0, size=shape)
                    return cp.asarray(out)
            except Exception:
                pass
            return cp.random.normal(loc=0.0, scale=1.0, size=shape)

        # NumPy backend
        else:
            if isinstance(rng_obj, (int, np.integer)):
                np.random.seed(int(rng_obj))
                return np.random.normal(loc=0.0, scale=1.0, size=shape)
            import numpy as _np_local
            try:
                if hasattr(rng_obj, "normal"):
                    return rng_obj.normal(0.0, 1.0, size=shape)
            except Exception:
                pass
            return _np_local.random.normal(loc=0.0, scale=1.0, size=shape)


    base = Sample_Standard_Normal((R, J), rng)
    base = _to_xp(base, dtype=xp.float64)

    samples_gen = mu[:, None] + base * sg[:, None]

    if distribution_spatial == "lognormal":
        CLIP_MIN, CLIP_MAX = -700.0, 700.0
        clipped = xp.clip(samples_gen, CLIP_MIN, CLIP_MAX)
        if xp.any(clipped != samples_gen):
            logger.warning("Some samples_gen values are clipped before exp() to prevent overflow.")
        samples_spatial = xp.exp(clipped)
    else:
        samples_spatial = samples_gen

    fn_mean, fn_stddev = _statistics(samples_spatial, w)

    return fn_mean, fn_stddev, samples_spatial


#=============================
# Spatial container (CPU)
#=============================
class HvsrSpatial:

    def __init__(self, coordinates):
        coordinates = np.array(coordinates, dtype=np.double)
        npts, dim = coordinates.shape
        if dim != 2:
            raise ValueError(f"coordinates harus (N,2), bukan {coordinates.shape}.")
        if npts < 3:
            raise ValueError("Minimal tiga koordinat diperlukan.")
        self.coordinates = coordinates

    def spatial_weights(self, boundary, declustering_method="voronoi"):  # pragma: no cover
        if declustering_method == "voronoi":
            weights, indices = self._voronoi_weights(boundary)
        else:
            raise NotImplementedError
        return (weights, indices)

    @staticmethod
    def _boundary_to_mask(boundary):
        boundary = np.array(boundary)
        if boundary.shape[1] != 2:
            raise ValueError(f"boundary harus (N,2), bukan {boundary.shape}.")
        bounding_pts = MultiPoint([Point(i) for i in boundary])
        return bounding_pts.convex_hull

    def _voronoi_weights(self, boundary):
        mask = self._boundary_to_mask(boundary)
        total_area = mask.area
        regions, indices = self._bounded_voronoi(mask)
        areas = np.empty(len(regions))
        for i, region in enumerate(regions):
            closed_points = np.vstack((region, region[0]))
            areas[i] = Polygon(closed_points).area
        return (areas/total_area, indices)

    def _cull_points(self, mask):
        passing_points, passing_indices = [], []
        for index, (x, y) in enumerate(self.coordinates):
            p = Point(x, y)
            if mask.contains(p):
                passing_points.append([x, y])
                passing_indices.append(index)
            else:
                logger.info(f"Discarding point ({x}, {y})")
        return (np.array(passing_points), passing_indices)

    def bounded_voronoi(self, boundary):
        mask = self._boundary_to_mask(boundary)
        return self._bounded_voronoi(mask)

    def _bounded_voronoi(self, mask, radius=1E6):
        points, indices = self._cull_points(mask)
        vor = Voronoi(points)
        regions, vertices = self._voronoi_finite_polygons_2d(vor, radius=radius)
        new_vertices = []
        for region in regions:
            unique_points = vertices[region]
            closed_points = np.vstack((unique_points, unique_points[0]))
            polygon_before = Polygon(closed_points)
            polygon_after = polygon_before.intersection(mask)
            xs, ys = polygon_after.boundary.xy
            new_unique_points = np.array(list(zip(xs[:-1], ys[:-1])))
            new_vertices.append(new_unique_points)
        return (new_vertices, indices)

    @staticmethod
    def _voronoi_finite_polygons_2d(vor, radius=None):
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Map ridges
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue
                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        return (new_regions, np.asarray(new_vertices))



####################################
# HVSR Traditional
####################################
def _to_host_array_safe(x):
    if x is None:
        return None
    if USE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _xp_to_py_scalar(x):
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return float(x)
    try:
        return float(x.item())
    except Exception:
        pass
    if USE_CUPY:
        try:
            return float(cp.asnumpy(x).item())
        except Exception:
            pass
    try:
        return float(np.asarray(x).item())
    except Exception as e:
        raise TypeError(f" Cannot convert xp-scalar to python float: {e}")


def _nanmean_weighted_xp(
        distribution: str,
        data,
        mean_kwargs: Optional[Dict] = None
):
    if mean_kwargs is None:
        mean_kwargs = dict(axis=0)

    arr = xp.asarray(data, dtype=xp.float64)

    # 1-D case
    if arr.ndim == 1:
        mask = ~xp.isnan(arr)
        valid_count = int(xp.sum(mask).item())
        if valid_count == 0:
            return xp.nan
        vals = arr[mask]
        if distribution == "lognormal":
            pos_mask = vals > 0
            pos_count = int(xp.sum(pos_mask).item())
            if pos_count == 0:
                return xp.nan
            vals_pos = vals[pos_mask]
            return xp.mean(xp.log(vals_pos))
        else:
            return xp.mean(vals)

    # ND (compute along axis)
    axis = mean_kwargs.get("axis", 0)
    mask = ~xp.isnan(arr)

    if distribution == "lognormal":
        pos_mask = mask & (arr > 0)
        count_valid = xp.sum(pos_mask, axis=axis)
        sum_log = xp.sum(xp.where(pos_mask, xp.log(arr), 0.0), axis=axis)
        mean_log = xp.where(count_valid == 0, xp.nan, sum_log / count_valid)
        return mean_log
    else:
        arr_zeroed = xp.where(mask, arr, 0.0)
        sum_valid = xp.sum(arr_zeroed, axis=axis)
        count_valid = xp.sum(mask, axis=axis)
        mean = xp.where(count_valid == 0, xp.nan, sum_valid / count_valid)
        return mean


def _nanstd_weighted_xp(
        distribution: str,
        data,
        std_kwargs: Optional[Dict] = None
):
    if std_kwargs is None:
        std_kwargs = dict(axis=0)

    arr = xp.asarray(data, dtype=xp.float64)
    axis = std_kwargs.get("axis", 0)

    if arr.ndim == 1:
        mask = ~xp.isnan(arr)
        if distribution == "lognormal":
            pos_mask = mask & (arr > 0)
            valid = arr[pos_mask]
            n = int(valid.size)
            if n <= 1:
                return xp.nan
            return xp.std(xp.log(valid), ddof=1)
        else:
            valid = arr[mask]
            n = int(valid.size)
            if n <= 1:
                return xp.nan
            return xp.std(valid, ddof=1)

    # 2D or more
    mask = ~xp.isnan(arr)
    if distribution == "lognormal":
        pos_mask = mask & (arr > 0)
        count_valid = xp.sum(pos_mask, axis=axis)
        arr_log = xp.where(pos_mask, xp.log(arr), 0.0)
        mean_log = xp.where(count_valid == 0, xp.nan, xp.sum(arr_log, axis=axis) / count_valid)
        if arr_log.ndim > 1:
            keepdims_mean = xp.expand_dims(mean_log, axis=axis)
        else:
            keepdims_mean = mean_log
        dev2 = xp.where(pos_mask, (arr_log - keepdims_mean) ** 2, 0.0)
        sum_dev2 = xp.sum(dev2, axis=axis)
        var = xp.where(count_valid <= 1, xp.nan, sum_dev2 / (count_valid - 1))
        return xp.sqrt(var)
    else:
        count_valid = xp.sum(mask, axis=axis)
        arr_proc = xp.where(mask, arr, 0.0)
        sum_proc = xp.sum(arr_proc, axis=axis)
        mean_proc = xp.where(count_valid == 0, xp.nan, sum_proc / count_valid)
        keepdims_mean = xp.expand_dims(mean_proc, axis=axis) if arr_proc.ndim > 1 else mean_proc
        dev2 = xp.where(mask, (arr_proc - keepdims_mean) ** 2, 0.0)
        sum_dev2 = xp.sum(dev2, axis=axis)
        var = xp.where(count_valid <= 1, xp.nan, sum_dev2 / (count_valid - 1))
        return xp.sqrt(var)


def _nth_std_factory_xp(
        n,
        distribution,
        mean_val,
        std_val
):
    if distribution == "lognormal":
        return xp.exp(xp.asarray(mean_val) + float(n) * xp.asarray(std_val))
    else:
        return xp.asarray(mean_val) + float(n) * xp.asarray(std_val)


def _compute_mean_and_band_xp(
        amplitude_matrix,
        frequency,
        distribution="lognormal",
        xp_lib=None,
        std_threshold=2.0
):
    if xp_lib is None:
        xp_lib = xp

    A = xp_lib.asarray(amplitude_matrix, dtype=xp_lib.float64)
    finite_mask = xp_lib.isfinite(A)
    count_valid = xp_lib.sum(finite_mask, axis=0)
    if USE_CUPY:
        count_valid_host = cp.asnumpy(count_valid)
        A_host = cp.asnumpy(A)
    else:
        count_valid_host = np.asarray(count_valid)
        A_host = np.asarray(A)

    enough_mask = count_valid_host >= 2

    if distribution == "lognormal":
        pos_mask = (A > 0) & finite_mask
        try:
            logvals = xp_lib.where(pos_mask, xp_lib.log(A), xp_lib.nan)
            mean_log = xp_lib.nanmean(logvals, axis=0)
            std_log = xp_lib.nanstd(logvals, axis=0, ddof=1)
        except Exception:
            logvals_host = np.where(A_host > 0, np.log(A_host), np.nan)
            mean_log = np.nanmean(logvals_host, axis=0)
            std_log = np.nanstd(logvals_host, axis=0, ddof=1)
            mean_log = xp_lib.asarray(mean_log)
            std_log = xp_lib.asarray(std_log)

        mean_linear = xp_lib.exp(mean_log)
        lower = xp_lib.exp(mean_log - std_log)
        upper = xp_lib.exp(mean_log + std_log)

        std_log_host = cp.asnumpy(std_log) if USE_CUPY else np.asarray(std_log)
        bad_std = (~np.isfinite(std_log_host)) | (std_log_host > std_threshold)

        if np.any(bad_std):
            p16 = np.nanpercentile(A_host, 16, axis=0)
            p84 = np.nanpercentile(A_host, 84, axis=0)
            med = np.nanmedian(A_host, axis=0)
            p16_xp = xp_lib.asarray(p16)
            p84_xp = xp_lib.asarray(p84)
            med_xp = xp_lib.asarray(med)
            for i, b in enumerate(bad_std):
                if b:
                    lower[i] = p16_xp[i]
                    upper[i] = p84_xp[i]
                    mean_linear[i] = med_xp[i]
        return mean_linear, lower, upper, enough_mask

    else:
        mean_curve = xp_lib.nanmean(A, axis=0)
        std_curve = xp_lib.nanstd(A, axis=0, ddof=1)
        lower = mean_curve - std_curve
        upper = mean_curve + std_curve
        return mean_curve, lower, upper, enough_mask


class HvsrTraditional():

    def __init__(
        self,
        frequency,
        amplitude,
        meta=None
    ):
        freq_x = HvsrCurve._check_input(frequency, "frequency")
        amp_x = HvsrCurve._check_input(amplitude, "amplitude")
        amp_x = xp.atleast_2d(amp_x)

        if freq_x.size != amp_x.shape[1]:
            msg = f"Shape of amplitude={amp_x.shape} and frequency={freq_x.shape} must be compatible."
            raise ValueError(msg)

        self.frequency = freq_x
        self.amplitude = amp_x
        self.n_curves = int(self.amplitude.shape[0])

        self.valid_window_boolean_mask = xp.ones((self.n_curves,), dtype=bool)
        self.valid_peak_boolean_mask = xp.ones((self.n_curves,), dtype=bool)
        self.meta = dict(meta) if isinstance(meta, dict) else dict()

        self._main_peak_frq = xp.empty(self.n_curves, dtype=xp.float64)
        self._main_peak_amp = xp.empty(self.n_curves, dtype=xp.float64)
        self._search_range_in_hz = (None, None)
        self._find_peaks_kwargs = None
        self.Update_Peaks_Bounded()

    @classmethod
    def from_hvsr_curves(
        cls,
        hvsr_curves,
        meta=None
    ):
        example = hvsr_curves[0]
        amplitude = xp.empty((len(hvsr_curves), len(example.frequency)), dtype=xp.float64)
        for idx, hvsr_curve in enumerate(hvsr_curves):
            if hvsr_curve.is_similar(example):
                amplitude[idx, :] = xp.asarray(hvsr_curve.amplitude)
            else:
                msg = f"All HvsrCurve objects must be similar, index {idx} is not similar to index 0."
                raise ValueError(msg)
        return cls(example.frequency, amplitude, meta=meta)

    @property
    def Peak_Frequencies(self):
        return self._main_peak_frq[self.valid_peak_boolean_mask]

    @property
    def Peak_Amplitudes(self):
        return self._main_peak_amp[self.valid_peak_boolean_mask]


    """
    def Update_Peaks_Bounded(
        self,
        search_range_in_hz=(None, None),
        find_peaks_kwargs=None
    ):

        if (search_range_in_hz == self._search_range_in_hz) and (find_peaks_kwargs == self._find_peaks_kwargs):
            return

        self._search_range_in_hz = tuple(search_range_in_hz)
        self._find_peaks_kwargs = {} if find_peaks_kwargs is None else dict(find_peaks_kwargs)
        self.meta["search_range_in_hz"] = self._search_range_in_hz
        self.meta["find_peaks_kwargs"] = None if find_peaks_kwargs is None else dict(find_peaks_kwargs)

        all_curves_flat = True
        for idx in range(self.n_curves):
            amplitude_row = self.amplitude[idx, :]
            f_peak, a_peak = HvsrCurve._find_peak_bounded(self.frequency,
                                                          amplitude_row,
                                                          search_range_in_hz=self._search_range_in_hz,
                                                          find_peaks_kwargs=self._find_peaks_kwargs)
            # if f_peak is None:
            #    logger.info(f"No peak found in window {idx}.")
            #    self._main_peak_frq[idx] = xp.nan
            #    self._main_peak_amp[idx] = xp.nan
            #    self.valid_window_boolean_mask[idx] = False
            #    self.valid_peak_boolean_mask[idx] = False
            # else:
            #    all_curves_flat = False
            #    # ensure xp storage (HvsrCurve may return python floats)
            #    self._main_peak_frq[idx] = xp.asarray(f_peak, dtype=xp.float64)
            #    self._main_peak_amp[idx] = xp.asarray(a_peak, dtype=xp.float64)
            #    self.valid_window_boolean_mask[idx] = True
            #    self.valid_peak_boolean_mask[idx] = True

            if f_peak is None:
                valid_mask = xp.isfinite(amplitude_row)
                valid_count = int(xp.sum(valid_mask).item())
                if valid_count == 0:
                    self._main_peak_frq[idx] = xp.nan
                    self._main_peak_amp[idx] = xp.nan
                    self.valid_window_boolean_mask[idx] = False
                    self.valid_peak_boolean_mask[idx] = False
                    logger.warning(f"No valid amplitude values for window {idx}; marking peak as NaN.")
                    continue
                safe_amp = xp.where(valid_mask, amplitude_row, -xp.inf)
                max_idx = int(xp.argmax(safe_amp).item())

                try:
                    f_peak_candidate = _xp_to_py_scalar(self.frequency[max_idx])
                except Exception:
                    f_peak_candidate = float(self.frequency[max_idx])
                try:
                    a_peak_candidate = _xp_to_py_scalar(safe_amp[max_idx])
                except Exception:
                    a_peak_candidate = float(safe_amp[max_idx])

                if not np.isfinite(a_peak_candidate):
                    self._main_peak_frq[idx] = xp.nan
                    self._main_peak_amp[idx] = xp.nan
                    self.valid_window_boolean_mask[idx] = False
                    self.valid_peak_boolean_mask[idx] = False
                    logger.warning(f"No finite maximum for window {idx}; marking peak as NaN.")
                    continue

                f_peak = float(f_peak_candidate)
                a_peak = float(a_peak_candidate)
                logger.warning(
                    f"No bounded peak found for window {idx}, fallback to global max at {f_peak:.3f} Hz (amp={a_peak}).")

            if f_peak is not None:
                all_curves_flat = False
                self._main_peak_frq[idx] = xp.asarray(f_peak, dtype=xp.float64)
                self._main_peak_amp[idx] = xp.asarray(a_peak, dtype=xp.float64)
                self.valid_window_boolean_mask[idx] = True
                self.valid_peak_boolean_mask[idx] = True
            else:
                self._main_peak_frq[idx] = xp.nan
                self._main_peak_amp[idx] = xp.nan
                self.valid_window_boolean_mask[idx] = False
                self.valid_peak_boolean_mask[idx] = False

        if all_curves_flat:
            self.valid_window_boolean_mask[:] = True
            logger.info("None of the curves contained a peak.")

        n_invalid_windows = int(xp.sum(~self.valid_window_boolean_mask).item()) \
            if USE_CUPY \
            else int(np.sum(~np.asarray(self.valid_window_boolean_mask)))
        if n_invalid_windows > 0:
            logger.warning("There are %d invalid windows (NaN/empty). These will be ignored in statistics.",
                           n_invalid_windows)
    """


    def Update_Peaks_Bounded(
        self,
        search_range_in_hz=(None, None),
        find_peaks_kwargs=None
    ):
        if (search_range_in_hz == self._search_range_in_hz) and (find_peaks_kwargs == self._find_peaks_kwargs):
            return

        self._search_range_in_hz = tuple(search_range_in_hz)
        self._find_peaks_kwargs = {} if find_peaks_kwargs is None else dict(find_peaks_kwargs)
        self.meta["search_range_in_hz"] = self._search_range_in_hz
        self.meta["find_peaks_kwargs"] = None if find_peaks_kwargs is None else dict(find_peaks_kwargs)

        n_flat = 0
        n_invalid = 0

        for idx in range(self.n_curves):
            amplitude_row = self.amplitude[idx, :]

            f_peak, a_peak = HvsrCurve._find_peak_bounded(
                self.frequency,
                amplitude_row,
                search_range_in_hz=self._search_range_in_hz,
                find_peaks_kwargs=self._find_peaks_kwargs
            )

            if f_peak is None or a_peak is None or (not np.isfinite(f_peak)) or (not np.isfinite(a_peak)):
                self._main_peak_frq[idx] = xp.nan
                self._main_peak_amp[idx] = xp.nan
                self.valid_window_boolean_mask[idx] = False
                self.valid_peak_boolean_mask[idx] = False
                n_invalid += 1
                logger.warning(
                    f"No bounded peak found for window {idx}; marking window invalid."
                )
                continue

            self._main_peak_frq[idx] = xp.asarray(float(f_peak), dtype=xp.float64)
            self._main_peak_amp[idx] = xp.asarray(float(a_peak), dtype=xp.float64)
            self.valid_window_boolean_mask[idx] = True
            self.valid_peak_boolean_mask[idx] = True

        if n_invalid == self.n_curves:
            n_flat = 1
            logger.info("None of the curves contained a bounded peak in the requested band.")

        n_invalid_windows = int(xp.sum(~self.valid_window_boolean_mask).item()) if USE_CUPY else int(
            np.sum(~np.asarray(self.valid_window_boolean_mask)))
        if n_invalid_windows > 0:
            logger.warning(
                "There are %d invalid windows (NaN/empty/no bounded peak). These will be ignored in statistics.",
                n_invalid_windows
            )


    def Mean_fn_Frequency(
        self,
        distribution="lognormal"
    ):
        dist = distribution
        vals = self.Peak_Frequencies
        if vals.size == 0:
            return xp.nan
        if USE_CUPY:
            mean_log_or_raw = _nanmean_weighted_xp(dist, vals)
            if dist == "lognormal":
                return xp.exp(mean_log_or_raw)
            else:
                return mean_log_or_raw
        else:
            return _nanmean_weighted(dist, np.asarray(vals))

    def Mean_fn_Amplitude(
        self,
        distribution="lognormal"
    ):
        vals = self.Peak_Amplitudes
        if vals.size == 0:
            return xp.nan
        if USE_CUPY:
            mean_log_or_raw = _nanmean_weighted_xp(distribution, vals)
            if distribution == "lognormal":
                return xp.exp(mean_log_or_raw)
            else:
                return mean_log_or_raw
        else:
            return _nanmean_weighted(distribution, np.asarray(vals))

    def std_fn_frequency(
        self,
        distribution="lognormal"
    ):
        vals = self.Peak_Frequencies
        if vals.size == 0:
            return xp.nan
        if USE_CUPY:
            return _nanstd_weighted_xp(distribution, vals)
        else:
            return _nanstd_weighted(distribution, np.asarray(vals))

    def std_fn_amplitude(
        self,
        distribution="lognormal"
    ):
        vals = self.Peak_Amplitudes
        if vals.size == 0:
            return xp.nan
        if USE_CUPY:
            return _nanstd_weighted_xp(distribution, vals)
        else:
            return _nanstd_weighted(distribution, np.asarray(vals))


    def cov_fn(
        self,
        distribution="lognormal"
    ):
        dist = distribution
        frq = self.Peak_Frequencies
        amp = self.Peak_Amplitudes

        if dist == "lognormal":
            frq_proc = xp.log(frq)
            amp_proc = xp.log(amp)
        else:
            frq_proc = frq
            amp_proc = amp

        mask = ~xp.isnan(frq_proc) & ~xp.isnan(amp_proc)
        if xp.sum(mask) <= 1:
            return xp.full((2, 2), xp.nan)

        fr = frq_proc[mask]
        ap = amp_proc[mask]
        fr_mean = xp.mean(fr)
        ap_mean = xp.mean(ap)
        cov_ff = xp.sum((fr - fr_mean) * (fr - fr_mean)) / (fr.size - 1)
        cov_fa = xp.sum((fr - fr_mean) * (ap - ap_mean)) / (fr.size - 1)
        cov_aa = xp.sum((ap - ap_mean) * (ap - ap_mean)) / (fr.size - 1)
        cov = xp.array([[cov_ff, cov_fa], [cov_fa, cov_aa]], dtype=xp.float64)
        return cov

    def mean_curve(
        self,
        distribution="lognormal"
    ):
        mask = self.valid_window_boolean_mask
        sel = self.amplitude[mask, :]
        if sel.shape[0] == 0:
            return xp.full((self.frequency.size,), xp.nan)
        if sel.shape[0] == 1:
            return sel.flatten()
        mean_curve, _, _, _ = _compute_mean_and_band_xp(sel, self.frequency, distribution=distribution, xp_lib=xp)
        return mean_curve

    def std_curve(
        self,
        distribution="lognormal"
    ):
        mask = self.valid_window_boolean_mask
        sel = self.amplitude[mask, :]
        if sel.shape[0] <= 1:
            warnings.warn("Standard deviation undefined for single or zero windows; returning zeros.")
            mean_c = self.mean_curve(distribution)
            return xp.zeros_like(mean_c)
        mean_c, lower, upper, valid_mask = _compute_mean_and_band_xp(sel, self.frequency, distribution=distribution, xp_lib=xp)

        try:
            if distribution == "lognormal":
                logvals = xp.where(sel > 0, xp.log(sel), xp.nan)
                std_log = xp.nanstd(logvals, axis=0, ddof=1)
                return std_log
            else:
                return xp.nanstd(sel, axis=0, ddof=1)
        except Exception:
            sel_host = cp.asnumpy(sel) if USE_CUPY else np.asarray(sel)
            if distribution == "lognormal":
                log_host = np.where(sel_host > 0, np.log(sel_host), np.nan)
                return xp.asarray(np.nanstd(log_host, axis=0, ddof=1))
            else:
                return xp.asarray(np.nanstd(sel_host, axis=0, ddof=1))

    def mean_curve_peak(
        self,
        distribution: Union[str, Sequence[float], np.ndarray] = "lognormal",
        mode: str = "warn"
    ) -> Tuple[Optional[float], Optional[float]]:

        #------------------------------------------
        # 1) Get frequency array on host (NumPy)
        #------------------------------------------
        freq_host = _to_host_array_safe(self.frequency)
        if freq_host is None:
            msg = "Frequency array is None; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        freq_host = np.asarray(freq_host).ravel()
        if freq_host.size == 0:
            msg = "Frequency array is empty; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        #----------------------------------------
        # 2) Obtain amplitude array on host:
        #----------------------------------------
        if isinstance(distribution, str):
            amp_host = _to_host_array_safe(self.mean_curve(distribution))
        else:
            amp_host = np.asarray(distribution) if not isinstance(distribution, np.ndarray) else distribution

        if amp_host is None:
            msg = "Amplitude array is None; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        amp_host = np.asarray(amp_host).ravel()
        if amp_host.size == 0:
            msg = "Amplitude array is empty; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        if amp_host.size != freq_host.size:
            msg = f"Frequency ({freq_host.size}) and amplitude ({amp_host.size}) lengths differ."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            nmin = min(freq_host.size, amp_host.size)
            freq_host = freq_host[:nmin]
            amp_host = amp_host[:nmin]
            if nmin == 0:
                return None, None

        med = np.nanmedian(amp_host)
        maxv = np.nanmax(amp_host)
        if np.isfinite(med) and med > 0 and maxv / med > 50:  # threshold: 50x median
            warnings.warn(f"mean_curve_peak: detected extremely large max/median ratio ({maxv / max(1e-12, med):.1f})."
                          " This may indicate outlier windows; inspect preprocessing.")

        #----------------------
        # 3) All-NaN check
        #----------------------
        if np.all(np.isnan(amp_host)):
            msg = "Amplitude curve is all NaN."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        #-------------------------------
        # 4) Determine search bounds
        #-------------------------------
        raw_range = getattr(self, "_search_range_in_hz", (None, None))
        try:
            low_raw, high_raw = raw_range
        except Exception:
            low_raw, high_raw = (None, None)

        positive_freqs = freq_host[freq_host > 0]
        if positive_freqs.size > 0:
            global_min_pos = float(np.min(positive_freqs))
        else:
            global_min_pos = float(np.min(freq_host))

        global_max = float(np.max(freq_host))

        low = float(low_raw) if (low_raw is not None) else global_min_pos
        high = float(high_raw) if (high_raw is not None) else global_max

        if low >= high:
            msg = f"Invalid search_range_in_hz ({raw_range}); low >= high."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg + " Falling back to global search range.")
            low, high = global_min_pos, global_max

        # ----------------------------------------
        # 5) Mask data inside bounds and finite
        # ----------------------------------------
        mask = (freq_host >= low) & (freq_host <= high) & np.isfinite(freq_host) & np.isfinite(amp_host)

        if not np.any(mask):
            msg = f"Mean curve has no finite data in search range {raw_range}."
            if mode == "strict":
                raise ValueError(msg)
            if mode == "warn":
                warnings.warn(msg + " Returning (None, None).")
                return None, None

        #--------------------------------------------------------------------
        # 6) If masked data exists, attempt to find bounded positive peak
        #--------------------------------------------------------------------
        if np.any(mask):
            masked_amp = np.where(mask, amp_host, np.nan)
            try:
                local_idx = int(np.nanargmax(masked_amp))
                local_max = float(masked_amp[local_idx])
            except ValueError:
                local_idx = None
                local_max = float("-inf")

            if local_idx is not None and np.isfinite(local_max):
                if local_max > 0.0:
                    f_peak = float(freq_host[local_idx])
                    a_peak = float(local_max)
                    return f_peak, a_peak
                else:
                    msg = f"Local maximum in bounded region is non-positive ({local_max})."
                    if mode == "strict":
                        raise ValueError(msg)
                    if mode == "warn":
                        warnings.warn(msg + " Returning (None, None).")
                        return None, None

        #-----------------------------------
        # 7) Fallback: global maximum
        #-----------------------------------
        try:
            global_idx = int(np.nanargmax(amp_host))
            global_max_val = float(amp_host[global_idx])
        except ValueError:
            msg = "Mean curve is NaN everywhere; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        if not np.isfinite(global_max_val):
            msg = "Global maximum is not finite; cannot determine peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        if global_max_val <= 0.0:
            msg = "Global maximum of mean curve is non-positive; cannot determine a positive peak."
            if mode == "strict":
                raise ValueError(msg)
            warnings.warn(msg)
            return None, None

        f_peak = float(freq_host[global_idx])
        a_peak = float(global_max_val)
        warnings.warn(f"No bounded positive peak found. Using global maximum: {f_peak:.6g} Hz, amplitude={a_peak:.6g}.")
        return f_peak, a_peak

    def nth_std_fn_frequency(
        self,
        n,
        distribution="lognormal"
    ):
        mean_val = self.Mean_fn_Frequency(distribution)
        std_val = self.std_fn_frequency(distribution)
        if USE_CUPY:
            return _nth_std_factory_xp(n, distribution, mean_val, std_val)
        else:
            return _nth_std_factory(n, distribution, mean_val, std_val)

    def nth_std_fn_amplitude(
        self,
        n,
        distribution="lognormal"
    ):
        mean_val = self.Mean_fn_Amplitude(distribution)
        std_val = self.std_fn_amplitude(distribution)
        if USE_CUPY:
            return _nth_std_factory_xp(n, distribution, mean_val, std_val)
        else:
            return _nth_std_factory(n, distribution, mean_val, std_val)

    def nth_std_curve(
        self,
        n,
        distribution="lognormal"
    ):
        mean_c = self.mean_curve(distribution)
        std_c = self.std_curve(distribution)
        if USE_CUPY:
            return _nth_std_factory_xp(n, distribution, mean_c, std_c)
        else:
            return _nth_std_factory(n, distribution, mean_c, std_c)

    def is_similar(
        self,
        other
    ):
        if not isinstance(other, HvsrTraditional):
            return False
        a = getattr(self, "frequency", None)
        b = getattr(other, "frequency", None)
        try:
            a_size = int(a.size)
            b_size = int(b.size)
        except Exception:
            a_np = np.asarray(a)
            b_np = np.asarray(b)
            return bool(np.allclose(a_np, b_np))
        if a_size != b_size:
            return False
        if cp is not None and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            return bool(cp.allclose(a, b))
        if cp is not None:
            if isinstance(a, cp.ndarray):
                a = cp.asnumpy(a)
            if isinstance(b, cp.ndarray):
                b = cp.asnumpy(b)
        return bool(np.allclose(np.asarray(a), np.asarray(b)))


    def __eq__(
        self,
        other
    ):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if not self.is_similar(other):
            return False

        if getattr(self, "n_curves", None) != getattr(other, "n_curves", None):
            return False

        def _to_host(x):
            if x is None:
                return None
            if USE_CUPY:
                if isinstance(x, cp.ndarray):
                    return cp.asnumpy(x)
                if isinstance(x, cp.generic):
                    return float(cp.asnumpy(x))
            try:
                return np.asarray(x)
            except Exception:
                return np.asarray(list(x))

        amp_self = _to_host(getattr(self, "amplitude", None))
        amp_other = _to_host(getattr(other, "amplitude", None))

        if amp_self is None or amp_other is None:
            return False
        if getattr(amp_self, "shape", None) != getattr(amp_other, "shape", None):
            return False
        if not np.allclose(amp_self, amp_other, rtol=1e-5, atol=1e-8, equal_nan=True):
            return False

        mask_self = _to_host(getattr(self, "valid_window_boolean_mask", None))
        mask_other = _to_host(getattr(other, "valid_window_boolean_mask", None))

        if mask_self is None or mask_other is None:
            return False
        if getattr(mask_self, "shape", None) != getattr(mask_other, "shape", None):
            return False
        if not np.array_equal(mask_self, mask_other):
            return False

        peakmask_self = _to_host(getattr(self, "valid_peak_boolean_mask", None))
        peakmask_other = _to_host(getattr(other, "valid_peak_boolean_mask", None))

        if peakmask_self is None or peakmask_other is None:
            return False
        if getattr(peakmask_self, "shape", None) != getattr(peakmask_other, "shape", None):
            return False
        if not np.array_equal(peakmask_self, peakmask_other):
            return False
        return True

    def to_numpy(self):
        if USE_CUPY:
            return cp.asnumpy(self.frequency), cp.asnumpy(self.amplitude)
        return (np.asarray(self.frequency), np.asarray(self.amplitude))

    def __str__(self):
        return f"HvsrTraditional at {id(self)}"

    def __repr__(self):
        return f"HvsrTraditional(frequency={self.frequency}, amplitude={self.amplitude}, meta={self.meta})"



####################################
# Object IO (Input/Output)
####################################
#----------------------
# Helper conversions
#----------------------
def _to_xp_array(arr, dtype=None):
    if USE_CUPY:
        if isinstance(arr, cp.ndarray):
            return arr.astype(dtype) if dtype is not None else arr
        return cp.asarray(np.asarray(arr), dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype)

def _to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    try:
        return np.asarray(arr)
    except Exception:
        return np.asarray(list(arr))

def _to_python_bool_list(mask):
    host = _to_host_array(mask)
    host_bool = host.astype(bool)
    return host_bool.tolist()


#------------------------
# Write HVSR object
#------------------------
# Helper: ensure host numpy arrays
def _host(x, dtype=None):
    arr = _to_host_array(x)
    if dtype is not None:
        return np.asarray(arr, dtype=dtype)
    return np.asarray(arr)


# Helper: check finiteness and optionally clean
def _check_and_clean(arr, name):
    a = np.asarray(arr)
    if not np.all(np.isfinite(a)):
        if allow_clean_nan:
            warnings.warn(f"{name} contains non-finite values; replacing with {nan_fill_value}.", UserWarning)
            a = np.nan_to_num(a, nan=nan_fill_value, posinf=nan_fill_value, neginf=nan_fill_value)
        else:
            coords = np.argwhere(~np.isfinite(a))
            r, c = (coords[0][0], coords[0][1]) if coords.size else (0, 0)
            raise ValueError(f"{name} contains non-finite values (first at index {coords[0]}). "
                             f"Set allow_clean_nan=True to auto-replace.")
    return a


def write_hvsr_object_to_file(
    hvsr,
    fname,
    distribution_mc="lognormal",
    distribution_fn="lognormal",
    allow_clean_nan: bool = False,
    nan_fill_value: float = 0.0,
):
    meta = deepcopy(hvsr.meta if hasattr(hvsr, "meta") else {})

    if isinstance(hvsr, HvsrTraditional):
        meta["valid_peak_boolean_mask"] = _to_python_bool_list(hvsr.valid_peak_boolean_mask)
        meta["valid_window_boolean_mask"] = _to_python_bool_list(hvsr.valid_window_boolean_mask)
        n_curves = int(hvsr.n_curves)
        data_headers_line = ["frequency (Hz)"]
        data_headers_line.extend([f"hvsr curve {x}" for x in range(1, n_curves + 1)])
        data_headers_line.extend([f"mean curve ({distribution_mc})", f"mean curve std ({distribution_mc})"])
        header_line = ",".join(data_headers_line)

        freq_host = _host(hvsr.frequency, dtype=float).ravel()
        amp_host = _host(hvsr.amplitude, dtype=float)

        if amp_host.ndim != 2:
            raise RuntimeError(f"hvsr.amplitude must be 2D (n_curves, n_freq); got shape {amp_host.shape}")
        if amp_host.shape[0] != n_curves:
            if amp_host.shape[1] == n_curves:
                amp_host = amp_host.T
            else:
                raise RuntimeError(f"Mismatch: hvsr.n_curves={n_curves} but amplitude.shape={amp_host.shape}")

        nfreq = freq_host.size
        if amp_host.shape[1] != nfreq:
            raise RuntimeError(f"Frequency length ({nfreq}) does not match amplitude second dim ({amp_host.shape[1]})")

        mean_host = _host(hvsr.mean_curve(distribution=distribution_mc), dtype=float).ravel()
        std_host = _host(hvsr.std_curve(distribution=distribution_mc), dtype=float).ravel()

        if mean_host.size != nfreq:
            warnings.warn("mean_curve length mismatch -> recomputing mean (nanmean) from amplitude.", UserWarning)
            mean_host = np.nanmean(amp_host, axis=0)
        if std_host.size != nfreq:
            warnings.warn("std_curve length mismatch -> recomputing std (nanstd) from amplitude.", UserWarning)
            std_host = np.nanstd(amp_host, axis=0)

        freq_host = _check_and_clean(freq_host, "frequency")
        amp_host = _check_and_clean(amp_host, "amplitude")
        mean_host = _check_and_clean(mean_host, "mean_curve")
        std_host = _check_and_clean(std_host, "std_curve")

        ncols = 1 + n_curves + 2
        array = np.empty((nfreq, ncols), dtype=float)
        array[:, 0] = freq_host
        array[:, 1:-2] = amp_host.T
        array[:, -2] = mean_host
        array[:, -1] = std_host

    elif isinstance(hvsr, HvsrAzimuthal):
        valid_window_boolean_masks = []
        valid_peak_boolean_masks = []
        for _h in hvsr.hvsrs:
            valid_window_boolean_masks.append(_to_python_bool_list(_h.valid_window_boolean_mask))
            valid_peak_boolean_masks.append(_to_python_bool_list(_h.valid_peak_boolean_mask))
        meta["valid_window_boolean_masks"] = valid_window_boolean_masks
        meta["valid_peak_boolean_masks"] = valid_peak_boolean_masks

        data_headers_line = ["frequency (Hz)"]
        total_curves = 0
        for az, _h in zip(hvsr.azimuths, hvsr.hvsrs):
            for curve_idx in range(1, _h.n_curves + 1):
                data_headers_line.append(f"azimuth {az} deg | hvsr curve {curve_idx}")
            total_curves += _h.n_curves
        data_headers_line.extend([f"mean curve ({distribution_mc})", f"mean curve std ({distribution_mc})"])
        header_line = ",".join(data_headers_line)

        freq_host = _host(hvsr.frequency, dtype=float).ravel()
        nfreq = freq_host.size
        ncols = 1 + total_curves + 2
        array = np.empty((nfreq, ncols), dtype=float)
        array[:, 0] = freq_host

        col_idx = 1
        for _h in hvsr.hvsrs:
            amp_h = _host(_h.amplitude, dtype=float)
            if amp_h.ndim != 2:
                raise RuntimeError(f"Per-azimuth amplitude must be 2D; got {_h} shape {amp_h.shape}")
            if amp_h.shape[1] != nfreq:
                raise RuntimeError(
                    f"Azimuth group amplitude freq dim ({amp_h.shape[1]}) != global freq length ({nfreq})")
            array[:, col_idx: col_idx + amp_h.shape[0]] = amp_h.T
            col_idx += amp_h.shape[0]

        mean_host = _host(hvsr.mean_curve(distribution=distribution_mc), dtype=float).ravel()
        std_host = _host(hvsr.std_curve(distribution=distribution_mc), dtype=float).ravel()

        if mean_host.size != nfreq:
            warnings.warn("Azimuthal mean length mismatch -> recomputing from amplitudes.", UserWarning)
            mean_host = np.nanmean(array[:, 1:1 + total_curves], axis=1)
        if std_host.size != nfreq:
            warnings.warn("Azimuthal std length mismatch -> recomputing from amplitudes.", UserWarning)
            std_host = np.nanstd(array[:, 1:1 + total_curves], axis=1)

        freq_host = _check_and_clean(freq_host, "frequency")
        array[:, 1:1 + total_curves] = _check_and_clean(array[:, 1:1 + total_curves], "amplitude (azimuthal)")
        mean_host = _check_and_clean(mean_host, "mean_curve")
        std_host = _check_and_clean(std_host, "std_curve")

        array[:, -2] = mean_host
        array[:, -1] = std_host

    elif isinstance(hvsr, HvsrDiffuseField):
        data_headers_line = ["frequency (Hz)", "hvsr curve 1"]
        header_line = ",".join(data_headers_line)
        freq_host = _host(hvsr.frequency, dtype=float).ravel()
        amp_host = _host(hvsr.amplitude, dtype=float).ravel()
        if freq_host.size != amp_host.size:
            raise RuntimeError("Diffuse field frequency and amplitude lengths do not match.")
        freq_host = _check_and_clean(freq_host, "frequency")
        amp_host = _check_and_clean(amp_host, "amplitude")
        array = np.empty((freq_host.size, 2), dtype=float)
        array[:, 0] = freq_host
        array[:, 1] = amp_host

    else:
        raise NotImplementedError("Unsupported HVSR object type for writing.")

    try:
        serial_meta = _to_serializable(meta)
    except Exception:
        serial_meta = meta

    header = "".join([json.dumps(serial_meta, indent=2), "\n", header_line])
    np.savetxt(fname, array, delimiter=",", header=header, encoding="utf-8")


#---------------------
# Read HVSR object
#---------------------
def read_hvsr_object_from_file(
    fname,
    allow_clean_nan=False
):
    with open(fname, "r") as f:
        raw_lines = f.readlines()

    header_lines = []
    data_start_line = 0
    for i, ln in enumerate(raw_lines):
        if ln.lstrip().startswith("#"):
            header_lines.append(ln.lstrip("#").strip())
        else:
            data_start_line = i
            break

    if not header_lines:
        raise ValueError(f"File {fname} contains no header lines starting with '#'. Cannot parse metadata.")

    try:
        meta_json = "\n".join(header_lines[:-1]) if len(header_lines) > 1 else "{}"
        meta = json.loads(meta_json) if meta_json.strip() else {}
    except Exception as e:
        raise ValueError(f"Failed to parse JSON metadata in header of {fname}: {e}")

    header_line = header_lines[-1]
    col_labels = [s.strip() for s in header_line.split(",")]

    try:
        array = np.genfromtxt(fname, comments="#", delimiter=",", dtype=float)
    except Exception as e:
        raise RuntimeError(f"Failed to read numeric data from {fname}: {e}")

    if array.size == 0:
        raise ValueError(f"No numeric data found in {fname} after header.")

    if array.ndim == 1:
        array = array.reshape((1, -1))

    nrows, ncols = array.shape

    nonfinite_mask = ~np.isfinite(array)
    if np.any(nonfinite_mask):
        if allow_clean_nan:
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            warnings.warn(f"Non-finite values found in {fname} and replaced with 0.0 because allow_clean_nan=True.",
                          UserWarning)
        else:
            coords = np.argwhere(nonfinite_mask)
            r0, c0 = coords[0]
            raise ValueError(
                f"Non-finite numeric value detected in {fname} at row {r0 + 1}, column {c0 + 1}."
                " Clean the file or call read_hvsr_object_from_file(..., allow_clean_nan=True) to auto-fill."
            )

    def _xp(x, dtype=None):
        try:
            return _to_xp_array(x, dtype=dtype)
        except Exception:
            return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    processing_method = meta.get("processing_method", None)
    if processing_method is None:
        raise ValueError("metadata key 'processing_method' missing in file header meta.")

    #----------------
    # Traditional
    #----------------
    if processing_method == "traditional":
        if ncols < 2:
            raise ValueError("Traditional format expects at least frequency + 1 curve column.")

        use_mean_std = False
        if ncols >= 3:
            if meta.get("has_mean_std", None) is True:
                use_mean_std = True
            else:
                if len(col_labels) >= 3:
                    last_label = col_labels[-1].lower()
                    second_last = col_labels[-2].lower()
                    if ("std" in last_label) or ("sigma" in last_label) or ("std" in second_last) or (
                            "mean" in second_last):
                        use_mean_std = True
                if ncols >= 4 and not use_mean_std:
                    pass

        if use_mean_std:
            freq = array[:, 0]
            curves = array[:, 1:-2].T
            mean_col = array[:, -2]
            std_col = array[:, -1]
        else:
            freq = array[:, 0]
            curves = array[:, 1:].T
            mean_col = None
            std_col = None

        if curves.ndim != 2:
            raise RuntimeError("Parsed curves array has unexpected shape.")

        freq_xp = _xp(freq)
        curves_xp = _xp(curves)
        try:
            hvsr_obj = HvsrTraditional(freq_xp, curves_xp)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate HvsrTraditional: {e}")

        hvsr_obj.meta = meta.copy()
        try:
            hvsr_obj.update_peaks_bounded(
                search_range_in_hz=tuple(meta.get("search_range_in_hz", (None, None))),
                find_peaks_kwargs=meta.get("find_peaks_kwargs", None),
            )
        except Exception:
            warnings.warn("update_peaks_bounded failed after reading HvsrTraditional: continuing.", UserWarning)

        vwbm = meta.get("valid_window_boolean_mask", None)
        vpbm = meta.get("valid_peak_boolean_mask", None)
        if vwbm is not None:
            vwbm_arr = np.asarray(vwbm, dtype=bool)
            if vwbm_arr.size != curves.shape[0]:
                warnings.warn("valid_window_boolean_mask length does not match number of curves; ignoring mask.",
                              UserWarning)
            else:
                hvsr_obj.valid_window_boolean_mask = _xp(vwbm_arr, dtype=bool)
        if vpbm is not None:
            vpbm_arr = np.asarray(vpbm, dtype=bool)
            if vpbm_arr.size != curves.shape[0]:
                warnings.warn("valid_peak_boolean_mask length does not match number of curves; ignoring mask.",
                              UserWarning)
            else:
                hvsr_obj.valid_peak_boolean_mask = _xp(vpbm_arr, dtype=bool)

        return hvsr_obj

    #---------------
    # Azimuthal
    #---------------
    elif processing_method == "azimuthal":
        if len(col_labels) < 2:
            raise ValueError("Azimuthal format expects header column labels describing azimuth groups.")

        end_cut = 1

        stop_idx = len(col_labels)
        if meta.get("has_mean_std", False) and len(col_labels) >= 3:
            stop_idx = len(col_labels) - 2

        curve_labels = col_labels[1:stop_idx]
        if len(curve_labels) == 0:
            raise ValueError("No curve labels found for azimuthal processing.")

        az_list = []
        for lab in curve_labels:
            m = azimuth_exec.search(lab)
            if m is None:
                raise ValueError(f"Could not parse azimuth from column label '{lab}'. Check header format.")
            az_list.append(m.groups()[0])

        groups = []
        curr_az = az_list[0]
        curr_cols = [1]
        for i, az in enumerate(az_list[1:], start=1):
            if az == curr_az:
                curr_cols.append(i + 1)
            else:
                groups.append((float(curr_az), list(curr_cols)))
                curr_az = az
                curr_cols = [i + 1]
        groups.append((float(curr_az), list(curr_cols)))

        hvsrs = []
        for az_val, cols in groups:
            start = cols[0]
            stop = cols[-1] + 1
            freq = array[:, 0]
            curves = array[:, start:stop].T
            if curves.size == 0:
                warnings.warn(f"No data for azimuth group {az_val}; skipping.", UserWarning)
                continue
            h = HvsrTraditional(_xp(freq), _xp(curves), meta={})
            hvsrs.append(h)

        if not hvsrs:
            raise RuntimeError("No azimuth groups could be constructed from file.")

        azimuths = [g[0] for g in groups]
        hvsr_obj = HvsrAzimuthal(hvsrs=hvsrs, azimuths=azimuths, meta=meta.copy())
        try:
            hvsr_obj.update_peaks_bounded(
                search_range_in_hz=tuple(meta.get("search_range_in_hz", (None, None))),
                find_peaks_kwargs=meta.get("find_peaks_kwargs", None),
            )
        except Exception:
            warnings.warn("update_peaks_bounded failed after reading HvsrAzimuthal: continuing.", UserWarning)

        vwbms = meta.get("valid_window_boolean_masks", None)
        vpbms = meta.get("valid_peak_boolean_masks", None)
        if vwbms is not None and vpbms is not None:
            if len(vwbms) != len(hvsr_obj.hvsrs) or len(vpbms) != len(hvsr_obj.hvsrs):
                warnings.warn(
                    "Length of mask lists in metadata does not match number of azimuth groups; ignoring masks.",
                    UserWarning)
            else:
                for _hvsr, _vwbm, _vpbm in zip(hvsr_obj.hvsrs, vwbms, vpbms):
                    _hvsr.valid_window_boolean_mask = _xp(np.asarray(_vwbm, dtype=bool), dtype=bool)
                    _hvsr.valid_peak_boolean_mask = _xp(np.asarray(_vpbm, dtype=bool), dtype=bool)

        return hvsr_obj

    # -----------------
    # Deffuse Field
    # -----------------
    elif processing_method == "diffuse_field":
        if ncols < 2:
            raise ValueError("Diffuse field format expects frequency + amplitude columns.")
        freq = array[:, 0]
        amp = array[:, 1]
        if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(amp)):
            if allow_clean_nan:
                freq = np.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
                amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError(
                    "Non-finite values detected in diffuse_field data; clean file or set allow_clean_nan=True.")
        hvsr_obj = HvsrDiffuseField(_xp(freq), _xp(amp), meta=meta.copy())
        try:
            hvsr_obj.update_peaks_bounded(search_range_in_hz=meta.get("search_range_in_hz", None),
                                          find_peaks_kwargs=meta.get("find_peaks_kwargs", None))
        except Exception:
            warnings.warn("update_peaks_bounded failed after reading HvsrDiffuseField: continuing.", UserWarning)
        return hvsr_obj
    else:
        raise NotImplementedError("Unknown processing_method in metadata when reading HVSR file.")


#---------------
# Settings IO
#---------------
def write_settings_object_to_file(settings_object, fname):
    settings_object.save(fname)

def read_settings_object_from_file(fname):
    with open(fname, "r") as f:
        attr_dict = json.load(f)
    if "preprocessing_method" in attr_dict.keys():
        if attr_dict["preprocessing_method"] == "psd":
            settings_object = PsdPreProcessingSettings()
        elif attr_dict["preprocessing_method"] == "hvsr":
            settings_object = HvsrPreProcessingSettin_



####################################
# CLI (Command Line Interface)
####################################
def Pool_Worker_Init(use_cupy: bool, gpu_device: int) -> None:
    logging.basicConfig(level=logging.INFO)
    if use_cupy and _HAS_CUPY:
        try:
            cp.cuda.Device(int(gpu_device)).use()
            try:
                import hvsrpy as _h
                _h.USE_CUPY = True
                _h.xp = cp
            except Exception:
                pass
            _ = cp.zeros(1)
            logger.info(f"[worker pid={os.getpid()}] Using CuPy device {gpu_device}.")
        except Exception as e:
            logger.warning(f"[worker pid={os.getpid()}] Failed to initialize CuPy device {gpu_device}: {e}")
    else:
        try:
            import hvsrpy as _h
            _h.USE_CUPY = False
        except Exception:
            pass

def Process_HVSR(
    fname: str,
    preprocessing_settings,
    processing_settings,
    cli_settings: Dict,
) -> None:
    # Backend
    use_cupy = cli_settings.get("use_cupy", False)
    gpu_device = cli_settings.get("gpu_device", 0)

    if use_cupy and _HAS_CUPY:
        try:
            cp.cuda.Device(int(gpu_device)).use()
            try:
                hvsrpy.USE_CUPY = True
                hvsrpy.xp = cp
            except Exception:
                pass
        except Exception as e:
            print(f"WARNING: Could not set CuPy device {gpu_device} in pid {os.getpid()}: {e}")
            use_cupy = False

    start = time.perf_counter()
    try:
        srecords = hvsrpy.read([[fname]])
        srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
        hvsr = hvsrpy.process(srecords, processing_settings)
        if not cli_settings.get("no_figure", False):
            plt.style.use(hvsrpy.HVSRPY_MPL_STYLE)
            fig, ax = plt.subplots(figsize=(3.75, 2.5), dpi=150)
            ax.set_ylim((0, cli_settings.get("ymax", 10.0)))
            hvsrpy.plot_single_panel_hvsr_curves(hvsr, ax=ax)
            out_png = f"{pathlib.Path(fname).stem}.png"
            fig.savefig(out_png)
            plt.close(fig)

        # Write hvsrcsv
        if not cli_settings.get("no_file", False):
            out_csv = f"{pathlib.Path(fname).stem}.csv"
            hvsrpy.write_hvsr_object_to_file(
                hvsr,
                out_csv,
                distribution_mc=cli_settings.get("distribution_mc", "lognormal"),
                distribution_fn=cli_settings.get("distribution_fn", "lognormal"),
            )

        end = time.perf_counter()
        print(f"{fname} completed in {end - start:.3f} seconds. (pid={os.getpid()})")

    finally:
        if _HAS_CUPY:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


@click.command()
@click.argument("file_names", nargs=-1, type=click.Path())
@click.option("--preprocessing_settings_file", default=None, type=click.Path(), help="Path to preprocessing settings file.")
@click.option("--processing_settings_file", default=None, type=click.Path(), help="Path to processing settings file.")
@click.option("--distribution_fn", default="lognormal", type=click.Choice(["lognormal", "normal"]), help="Distribution assumed to describe the site frequency.")
@click.option("--distribution_mc", default="lognormal", type=click.Choice(["lognormal", "normal"]), help="Distribution assumed to describe the median curve.")
@click.option("--no_figure", is_flag=True, help="Flag to prevent figure creation.")
@click.option("--no_file", is_flag=True, help="Flag to prevent HVSR from being saved.")
@click.option("--ymax", default=10.0, type=float, help="Upper y limit of the HVSR figure.")
@click.option("--nproc", default=None, type=int, help="Number of subprocesses to launch, default is number of CPUs minus 1.")
@click.option("--use-cupy/--no-use-cupy", default=None, help="Force use of CuPy in workers (default: auto detect).")
@click.option("--gpu-device", default=0, type=int, help="GPU device id to use for CuPy workers (default=0).")
@click.pass_context
def cli(ctx, **kwargs):
    # Read settings objects
    preprocessing_settings = read_settings_object_from_file(kwargs.pop("preprocessing_settings_file"))
    processing_settings = read_settings_object_from_file(kwargs.pop("processing_settings_file"))

    # Interpret args
    file_names = kwargs.pop("file_names")
    if len(file_names) == 0:
        print("No input files provided. Exiting.")
        return

    # Decide whether to use cupy
    use_cupy_arg = kwargs.pop("use_cupy")
    if use_cupy_arg is None:
        use_cupy = _HAS_CUPY
    else:
        use_cupy = bool(use_cupy_arg)
        if use_cupy and not _HAS_CUPY:
            print("WARNING: --use-cupy passed but CuPy is not installed. Falling back to CPU (NumPy).")
            use_cupy = False

    gpu_device = int(kwargs.pop("gpu_device"))

    if kwargs.get("no_figure") and kwargs.get("no_file"):
        return

    nproc = os.cpu_count() - 1 if kwargs.pop("nproc") is None else int(kwargs.pop("nproc"))
    ntasks = len(file_names)
    nworkers = max(1, min(ntasks, nproc))

    cli_settings = {
        "distribution_fn": kwargs.get("distribution_fn", "lognormal"),
        "distribution_mc": kwargs.get("distribution_mc", "lognormal"),
        "no_figure": kwargs.get("no_figure", False),
        "no_file": kwargs.get("no_file", False),
        "ymax": float(kwargs.get("ymax", 10.0)),
        "use_cupy": use_cupy,
        "gpu_device": gpu_device,
    }

    if use_cupy and _HAS_CUPY:
        try:
            dev = cp.cuda.Device(gpu_device)
            name = dev.name
            print(f"Using CuPy on GPU device {gpu_device}: {name}")
        except Exception:
            print(f"Using CuPy requested but getting device info failed for device {gpu_device}.")

    ctx = get_context("spawn")

    if nworkers == 1:
        for fname in file_names:
            Process_HVSR(fname, preprocessing_settings, processing_settings, cli_settings)
    else:
        with ctx.Pool(processes=nworkers, initializer=Pool_Worker_Init, initargs=(use_cupy, gpu_device)) as pool:
            pool.starmap(
                Process_HVSR,
                zip(
                    file_names,
                    itertools.repeat(preprocessing_settings),
                    itertools.repeat(processing_settings),
                    itertools.repeat(cli_settings),
                ),
                chunksize=max(1, ntasks // nworkers),
            )



###################################
# Preprocessing
###################################
def ensure_recording_backend(recording: SeismicRecording3C) -> None:
    if not USE_CUPY:
        return
    assert cp is not None

    for comp_name in ("ns", "ew", "vt"):
        ts = getattr(recording, comp_name)
        amp = getattr(ts, "amplitude", None)
        if amp is None:
            continue
        if isinstance(amp, cp.ndarray):
            continue
        try:
            host_arr = np.asarray(amp)
            ts.amplitude = cp.asarray(host_arr)
        except Exception:
            warnings.warn("Could not convert TimeSeries amplitude to CuPy array; proceeding with NumPy.", RuntimeWarning)

def sanitize_signal(
    sig,
    xp_lib=None
):
    if xp_lib is None:
        xp_lib = xp

    try:
        arr = xp_lib.asarray(sig, dtype=xp_lib.float64)
    except Exception:
        host = np.asarray(sig, dtype=np.float64)
        host = np.nan_to_num(host, nan=0.0, posinf=0.0, neginf=0.0)
        if USE_CUPY and xp_lib is cp:
            return cp.asarray(host)
        return np.asarray(host)

    try:
        all_finite = bool(xp_lib.all(xp_lib.isfinite(arr)).item()) if USE_CUPY else bool(xp_lib.all(xp_lib.isfinite(arr)))
    except Exception:
        host = np.asarray(arr)
        all_finite = np.all(np.isfinite(host))

    if not all_finite:
        warnings.warn("sanitize_signal: input contains NaN/Inf - replacing invalid values with 0.0", RuntimeWarning)
        try:
            arr = xp_lib.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            host = np.asarray(arr)
            host = np.nan_to_num(host, nan=0.0, posinf=0.0, neginf=0.0)
            if USE_CUPY and xp_lib is cp:
                arr = cp.asarray(host)
            else:
                arr = np.asarray(host)
    return arr


#=============================
# Preprocessing functions
#=============================
def hvsr_preprocess(
    records,
    settings
):
    preprocessed_records: List[SeismicRecording3C] = []

    if isinstance(records, SeismicRecording3C):
        records = [records]
    else:
        records = list(records)

    if len(records) == 0:
        raise ValueError("hvsr_preprocess: empty records list provided.")

    ex_dt = records[0].vt.dt_in_seconds

    for idx, srecord3c in enumerate(records):
        ensure_recording_backend(srecord3c)

        if abs(srecord3c.vt.dt_in_seconds - ex_dt) > 1E-5 and not settings.ignore_dissimilar_time_step_warning:  # pragma: no cover
            msg = (
                f"The dt_in_seconds of all records are not equal, "
                f"dt_in_seconds of record {idx} is {srecord3c.vt.dt_in_seconds} "
                f"which does not match dt_in_seconds of record 0 of {ex_dt}."
            )
            warnings.warn(msg)

        if settings.orient_to_degrees_from_north is not None:
            srecord3c.orient_sensor_to(settings.orient_to_degrees_from_north)

        for comp in ("ns", "ew", "vt"):
            ts = getattr(srecord3c, comp)
            try:
                cleaned = sanitize_signal(getattr(ts, "amplitude", None), xp_lib=xp)
                ts.amplitude = cleaned
            except Exception as e:
                warnings.warn(f"hvsr_preprocess: could not sanitize {comp} of record {idx}: {e}", RuntimeWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        if settings.window_length_in_seconds is not None:
            windows = srecord3c.split(settings.window_length_in_seconds)
        else:
            windows = [srecord3c]

        if (settings.detrend is not None) and (settings.detrend != "none"):
            for window in windows:
                window.detrend(type=settings.detrend)

        for w in windows:
            ensure_recording_backend(w)
            for comp in ("ns", "ew", "vt"):
                ts = getattr(w, comp)
                try:
                    cleaned = sanitize_signal(getattr(ts, "amplitude", None), xp_lib=xp)
                    ts.amplitude = cleaned
                except Exception:
                    pass

        preprocessed_records.extend(windows)

    return preprocessed_records


def psd_preprocess(
    records,
    settings
):
    prepare_fft_settings(records, settings)
    preprocessed_records: List[SeismicRecording3C] = []

    if isinstance(records, SeismicRecording3C):
        records = [records]
    else:
        records = list(records)

    if len(records) == 0:
        raise ValueError("psd_preprocess: empty records list provided.")

    ex_dt = records[0].vt.dt_in_seconds

    for idx, srecord3c in enumerate(records):
        ensure_recording_backend(srecord3c)

        if abs(srecord3c.vt.dt_in_seconds - ex_dt) > 1E-5 and not settings.ignore_dissimilar_time_step_warning:  # pragma: no cover
            msg = (
                f"The dt_in_seconds of all records are not equal, "
                f"dt_in_seconds of record {idx} is {srecord3c.vt.dt_in_seconds} "
                f"which does not match dt_in_seconds of record 0 of {ex_dt}."
            )
            warnings.warn(msg)

        if settings.orient_to_degrees_from_north is not None:
            srecord3c.orient_sensor_to(settings.orient_to_degrees_from_north)

        for comp in ("ns", "ew", "vt"):
            ts = getattr(srecord3c, comp)
            try:
                ts.amplitude = sanitize_signal(getattr(ts, "amplitude", None), xp_lib=xp)
            except Exception as e:
                warnings.warn(f"psd_preprocess: could not sanitize {comp} of record {idx}: {e}", RuntimeWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        if settings.instrument_transfer_function is not None or settings.differentiate:
            srecord3c.detrend(type="constant")
            if getattr(settings, "window_type_and_width", None) is None:
                raise ValueError("psd_preprocess: settings.window_type_and_width must be provided when instrument/differentiate is requested.")
            srecord3c.window(*settings.window_type_and_width)

        if settings.instrument_transfer_function is not None:
            for component in ["ns", "ew", "vt"]:
                orig_ts = getattr(srecord3c, component)
                new_tseries = Remove_Instrument_Response(orig_ts, settings.instrument_transfer_function, settings.fft_settings)
                setattr(srecord3c, component, new_tseries)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                srecord3c.butterworth_filter(settings.filter_corner_frequencies_in_hz)

        if settings.differentiate:
            for component in ["ns", "ew", "vt"]:
                orig_ts = getattr(srecord3c, component)
                new_tseries = Differentiate(orig_ts, settings.fft_settings)
                setattr(srecord3c, component, new_tseries)

        ensure_recording_backend(srecord3c)

        if settings.window_length_in_seconds is not None:
            windows = srecord3c.split(settings.window_length_in_seconds)
        else:
            windows = [srecord3c]

        for w in windows:
            ensure_recording_backend(w)
            for comp in ("ns", "ew", "vt"):
                ts = getattr(w, comp)
                try:
                    ts.amplitude = sanitize_signal(getattr(ts, "amplitude", None), xp_lib=xp)
                except Exception:
                    pass

        if (settings.detrend is not None) and (settings.detrend != "none"):
            for window in windows:
                window.detrend(type=settings.detrend)

        preprocessed_records.extend(windows)

    return preprocessed_records


PREPROCESSING_METHODS = {
    "hvsr": hvsr_preprocess,
    "psd": psd_preprocess,
}

def preprocess(records, settings):
    if settings.preprocessing_method not in PREPROCESSING_METHODS:
        raise KeyError(f"Unknown preprocessing_method: {settings.preprocessing_method}")
    return PREPROCESSING_METHODS[settings.preprocessing_method](records, settings)



####################################
####################################
# Processing
####################################
####################################
#===============
# Helpers
#===============
def nextpow2(
    n,
    minimum_power_of_two=2 ** 15
):
    power_of_two = minimum_power_of_two
    while True:
        if power_of_two > n:
            return power_of_two
        power_of_two <<= 1

def _to_xp_array(
    arr,
    dtype=None
):
    if USE_CUPY:
        if isinstance(arr, cp.ndarray):
            return arr.astype(dtype) if dtype is not None else arr
        return cp.asarray(np.asarray(arr), dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype)

def _to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def _maybe_cpu_call(
    operator_fn,
    *args,
    **kwargs
):
    try:
        return operator_fn(*args, **kwargs)
    except Exception as e:
        logger.debug("Operator raised on xp backend, falling back to host numpy: %s", e)
        args_host = [_to_host_array(a) if isinstance(a, (xp.ndarray, np.ndarray)) else a for a in args]
        kwargs_host = {k: (_to_host_array(v) if isinstance(v, (xp.ndarray, np.ndarray)) else v) for k, v in kwargs.items()}
        return operator_fn(*args_host, **kwargs_host)

def _safe_divide(
    num,
    den,
    eps=EPS,
    xp_lib=None,
    replace_nonfinite_with=None
):
    if xp_lib is None:
        xp_lib = xp
    num_x = xp_lib.asarray(num, dtype=xp_lib.float64)
    den_x = xp_lib.asarray(den, dtype=xp_lib.float64)
    denom_safe = xp_lib.where(xp_lib.abs(den_x) < eps, eps, den_x)
    res = num_x / denom_safe
    if replace_nonfinite_with is None:
        res = xp_lib.where(xp_lib.isfinite(res), res, xp_lib.nan)
    else:
        val = float(replace_nonfinite_with)
        try:
            res = xp_lib.where(xp_lib.isfinite(res), res, xp_lib.asarray(val, dtype=res.dtype))
        except Exception:
            host = np.asarray(res)
            host[~np.isfinite(host)] = val
            res = xp_lib.asarray(host)
    return res


def sanitize_signal(
    sig,
    xp_lib=None
):
    if xp_lib is None:
        xp_lib = xp

    try:
        arr = xp_lib.asarray(sig, dtype=xp_lib.float64)
    except Exception:
        host = np.asarray(sig, dtype=np.float64)
        host = np.nan_to_num(host, nan=0.0, posinf=0.0, neginf=0.0)
        if USE_CUPY and xp_lib is cp:
            return cp.asarray(host)
        return np.asarray(host)

    try:
        all_finite = bool(xp_lib.all(xp_lib.isfinite(arr)).item()) if USE_CUPY else bool(xp_lib.all(xp_lib.isfinite(arr)))
    except Exception:
        host = np.asarray(arr)
        all_finite = np.all(np.isfinite(host))

    if not all_finite:
        warnings.warn("sanitize_signal: input contains NaN/Inf - replacing invalid values with 0.0", RuntimeWarning)
        try:
            arr = xp_lib.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            host = np.asarray(arr)
            host = np.nan_to_num(host, nan=0.0, posinf=0.0, neginf=0.0)
            if USE_CUPY and xp_lib is cp:
                arr = cp.asarray(host)
            else:
                arr = np.asarray(host)
    return arr


#------------------------
# FFT & preparation
#------------------------
def prepare_fft_settings(
    records: Iterable,
    settings
) -> None:
    max_n_samples = 0
    for record in records:
        if record.vt.n_samples > max_n_samples:
            max_n_samples = record.vt.n_samples
    good_n = nextpow2(max_n_samples)
    if settings.fft_settings is None:
        settings.fft_settings = dict(n=good_n)
    else:
        user_n = settings.fft_settings.get("n", max_n_samples)
        if user_n is None:
            settings.fft_settings["n"] = max_n_samples
        else:
            settings.fft_settings["n"] = good_n if good_n > user_n else user_n


def prepare_records_with_inconsistent_dt(
    records: Iterable,
    settings
) -> Tuple[Iterable, Dict]:

    dt_with_count = dict()
    for record in records:
        _dt = record.ns.dt_in_seconds
        try:
            dt_with_count[_dt] += 1
        except KeyError:
            dt_with_count[_dt] = 1

    if settings.handle_dissimilar_time_steps_by == "frequency_domain_resampling":
        return records, dt_with_count

    elif settings.handle_dissimilar_time_steps_by == "keeping_smallest_time_step":
        smallest_dt = min(dt_with_count.keys())
        abbr_records = []
        count = 0
        for record in records:
            if record.ns.dt_in_seconds == smallest_dt:
                abbr_records.append(record)
                count += 1
                if count == dt_with_count[smallest_dt]:
                    break

        if (len(records) - len(abbr_records)) > 0:
            msg = "Keeping the smallest time step resulting in the removal of "
            msg += f"{len(records) - len(abbr_records)} of {len(records)} records. "
            msg += f"{len(abbr_records)} records remain."
            warnings.warn(msg)

        return abbr_records, {smallest_dt: count}

    elif settings.handle_dissimilar_time_steps_by == "keeping_majority_time_step":
        majority_count = 0
        majority_dt = None
        for potential_dt, potential_count in dt_with_count.items():
            if potential_count > majority_count:
                majority_dt = potential_dt
                majority_count = potential_count

        abbr_records = []
        count = 0
        for record in records:
            if record.ns.dt_in_seconds == majority_dt:
                abbr_records.append(record)
                count += 1
                if count == majority_count:
                    break

        if (len(records) - len(abbr_records)) > 0:
            msg = "Keeping the majority time step resulting in the removal of "
            msg += f"{len(records) - len(abbr_records)} of {len(records)} records. "
            msg += f"{len(abbr_records)} records remain."
            warnings.warn(msg)

        return abbr_records, {majority_dt: majority_count}

    return records, dt_with_count


def check_nyquist_frequency(
    dt,
    fcs
):
    fnyq = 1.0 / (2.0 * float(dt))
    if float(np.max(_to_host_array(fcs))) > fnyq:
        msg = f"The maximum resampling frequency of {float(np.max(_to_host_array(fcs))):.2f} Hz "
        msg += f"exceeds the records Nyquist frequency of {fnyq:.2f} Hz"
        raise ValueError(msg)


def arithmetic_mean(fft_ns, fft_ew, settings=None):
    return (fft_ns + fft_ew) / 2.0


if xp.__name__ == "cupy":
    @contextlib.contextmanager
    def errstate(*args, **kwargs):
        yield
    xp.errstate = errstate


#==================================
# Traditional HVSR processing
#==================================
def traditional_hvsr_processing(
    records,
    settings
):
    prepare_fft_settings(records, settings)

    if getattr(settings, "window_type_and_width", None) is None:
        raise ValueError("settings.window_type_and_width must be provided and not None.")
    if not hasattr(settings, "fft_settings"):
        settings.fft_settings = {}

    records, dt_with_count = prepare_records_with_inconsistent_dt(records, settings)

    fcs = _to_xp_array(settings.smoothing["center_frequencies_in_hz"], dtype=xp.float64)
    n_fcs = int(_to_host_array(fcs).size)

    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    hvsr_spectra = xp.full((len(records), n_fcs), xp.nan, dtype=xp.float64)

    processed_order = []
    hvsr_idx = 0

    for dt, count in dt_with_count.items():
        try:
            fft_frq = xp.fft.rfftfreq(settings.fft_settings.get("n", None), dt)
            fft_frq_host = _to_host_array(fft_frq)
        except Exception:
            fft_frq_host = np_rfftfreq(settings.fft_settings.get("n", None), dt)
            fft_frq = _to_xp_array(fft_frq_host)

        nfft_freq = fft_frq_host.size

        raw_spectra = xp.empty((count * 2, nfft_freq), dtype=xp.float64)
        hor_idx = 0
        ver_idx = count

        local_rows_produced = 0
        for org_idx, record in enumerate(records):
            if record.ns.dt_in_seconds != dt:
                continue

            processed_order.append(org_idx)
            local_rows_produced += 1

            record.window(*settings.window_type_and_width)

            try:
                ns_amp = sanitize_signal(_to_xp_array(record.ns.amplitude, dtype=xp.float64), xp_lib=xp)
                ew_amp = sanitize_signal(_to_xp_array(record.ew.amplitude, dtype=xp.float64), xp_lib=xp)
                vt_amp = sanitize_signal(_to_xp_array(record.vt.amplitude, dtype=xp.float64), xp_lib=xp)
            except Exception as e:
                logger.warning("Sanitize failed for record %d: %s — falling back to host sanitize.", org_idx, e)
                ns_amp = _to_xp_array(np.nan_to_num(np.asarray(record.ns.amplitude, dtype=np.float64)), dtype=xp.float64)
                ew_amp = _to_xp_array(np.nan_to_num(np.asarray(record.ew.amplitude, dtype=np.float64)), dtype=xp.float64)
                vt_amp = _to_xp_array(np.nan_to_num(np.asarray(record.vt.amplitude, dtype=np.float64)), dtype=xp.float64)

            try:
                fft_ns = xp.abs(xp.fft.rfft(ns_amp, **settings.fft_settings))
                fft_ew = xp.abs(xp.fft.rfft(ew_amp, **settings.fft_settings))
                fft_vt = xp.abs(xp.fft.rfft(vt_amp, **settings.fft_settings))
            except Exception as e:
                logger.debug("Device FFT failed for record %d: %s — falling back to host FFT.", org_idx, e)
                ns_h = _to_host_array(ns_amp)
                ew_h = _to_host_array(ew_amp)
                vt_h = _to_host_array(vt_amp)
                fft_ns = _to_xp_array(np.abs(np.fft.rfft(ns_h, **settings.fft_settings)), dtype=xp.float64)
                fft_ew = _to_xp_array(np.abs(np.fft.rfft(ew_h, **settings.fft_settings)), dtype=xp.float64)
                fft_vt = _to_xp_array(np.abs(np.fft.rfft(vt_h, **settings.fft_settings)), dtype=xp.float64)

            method_name = getattr(settings, "method_to_combine_horizontals", "arithmetic_mean")
            method_callable = None
            try:
                from .processing import COMBINE_HORIZONTAL_REGISTER as _C_REG
                method_callable = _C_REG.get(method_name, None)
            except Exception:
                method_callable = None
            if method_callable is None:
                method_callable = globals().get(method_name, None) or arithmetic_mean

            try:
                h_full = method_callable(fft_ns, fft_ew, settings)
            except Exception as e:
                logger.warning("Horizontal combine method '%s' failed for record %d: %s — falling back to arithmetic_mean.", method_name, org_idx, e)
                h_full = arithmetic_mean(fft_ns, fft_ew, settings)

            h_full = _to_xp_array(h_full, dtype=xp.float64)
            fft_vt = _to_xp_array(fft_vt, dtype=xp.float64)

            raw_spectra[hor_idx, :nfft_freq] = h_full
            raw_spectra[ver_idx, :nfft_freq] = fft_vt
            hor_idx += 1
            ver_idx += 1

        operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
        try:
            smoothing_fn = SMOOTHING_OPERATORS[operator]
            smooth_spectra = _maybe_cpu_call(smoothing_fn, fft_frq, raw_spectra, fcs, bandwidth)
            smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)
        except Exception as e:
            logger.debug("Smoothing operator failed on device: %s — falling back to host smoothing.", e)
            fft_frq_h = _to_host_array(fft_frq)
            raw_h = _to_host_array(raw_spectra)
            fcs_h = _to_host_array(fcs)
            smooth_h = SMOOTHING_OPERATORS[operator](fft_frq_h, raw_h, fcs_h, bandwidth)
            if hasattr(smooth_h, "get"):
                smooth_h = smooth_h.get()
            smooth_spectra = _to_xp_array(np.asarray(smooth_h, dtype=np.float64), dtype=xp.float64)

        try:
            smooth_cols = int(smooth_spectra.shape[1])
        except Exception:
            smooth_cols = int(_to_host_array(smooth_spectra).shape[1])

        if smooth_cols == n_fcs:
            final_smooth = smooth_spectra
        elif smooth_cols == nfft_freq:
            smooth_h = _to_host_array(smooth_spectra)
            fft_h = fft_frq_host
            fcs_h = _to_host_array(fcs)
            new = np.full((smooth_h.shape[0], n_fcs), np.nan, dtype=np.float64)
            for r in range(smooth_h.shape[0]):
                row = smooth_h[r, :]
                valid = np.isfinite(row)
                if not valid.any():
                    continue
                interp_row = np.interp(fcs_h, fft_h, row, left=np.nan, right=np.nan)
                if np.isnan(interp_row).any():
                    finite_idx = np.where(np.isfinite(row))[0]
                    if finite_idx.size:
                        left_val = row[finite_idx[0]]
                        right_val = row[finite_idx[-1]]
                        interp_row[np.isnan(interp_row) & (fcs_h < fft_h[finite_idx[0]])] = left_val
                        interp_row[np.isnan(interp_row) & (fcs_h > fft_h[finite_idx[-1]])] = right_val
                new[r, :] = interp_row
            final_smooth = _to_xp_array(new, dtype=xp.float64)
            logger.debug("Interpolated smooth_spectra from FFT axis (%d) -> fcs (%d).", nfft_freq, n_fcs)
        else:
            smooth_h = _to_host_array(smooth_spectra)
            fcs_h = _to_host_array(fcs)
            if smooth_h.shape[1] == n_fcs:
                final_smooth = _to_xp_array(smooth_h, dtype=xp.float64)
            else:
                returned_len = smooth_h.shape[1]
                ret_axis = np.linspace(np.nanmin(fft_frq_host), np.nanmax(fft_frq_host), returned_len)
                new = np.full((smooth_h.shape[0], n_fcs), np.nan, dtype=np.float64)
                for r in range(smooth_h.shape[0]):
                    interp_row = np.interp(fcs_h, ret_axis, smooth_h[r, :], left=np.nan, right=np.nan)
                    new[r, :] = interp_row
                final_smooth = _to_xp_array(new, dtype=xp.float64)
                logger.warning("Smoothing returned unexpected freq axis length %d — applied synthetic interpolation to fcs.", returned_len)

        h_part = final_smooth[:local_rows_produced, :]
        v_part = final_smooth[local_rows_produced:local_rows_produced * 2, :]

        if h_part.shape[0] != v_part.shape[0]:
            min_rows = min(h_part.shape[0], v_part.shape[0])
            h_part = h_part[:min_rows, :]
            v_part = v_part[:min_rows, :]

        eps = EPS
        v_safe = xp.where(xp.isfinite(v_part) & (v_part > eps), v_part, eps)

        ratio = _safe_divide(h_part, v_part, eps=1e-8, xp_lib=xp, replace_nonfinite_with=np.nan)

        rows_filled = int(ratio.shape[0])
        for local_idx in range(rows_filled):
            global_row = hvsr_idx + local_idx
            if global_row >= len(processed_order):
                break
            orig_idx = processed_order[global_row]
            if 0 <= orig_idx < hvsr_spectra.shape[0]:
                hvsr_spectra[orig_idx] = ratio[local_idx]
            else:
                logger.debug("Skipping assignment for orig_idx=%s out of bounds.", orig_idx)

        hvsr_idx += rows_filled

    try:
        h_host = _to_host_array(hvsr_spectra)
    except Exception:
        h_host = np.asarray(hvsr_spectra, dtype=np.float64)

    fcs_host = np.asarray(_to_host_array(fcs), dtype=np.float64)
    nrec = h_host.shape[0]

    for r in range(nrec):
        row = h_host[r, :]
        if not np.any(np.isfinite(row)):
            logger.warning("HVSR row %d entirely non-finite; filling with zeros.", r)
            row[:] = 0.0
        elif np.any(~np.isfinite(row)):
            idx = np.arange(row.size)
            valid = np.isfinite(row)
            try:
                interp_vals = np.interp(fcs_host, fcs_host[valid], row[valid], left=np.nan, right=np.nan)
            except Exception:
                interp_vals = np.interp(idx, idx[valid], row[valid])
            if np.isnan(interp_vals).any():
                finite_idx = np.where(valid)[0]
                first, last = finite_idx[0], finite_idx[-1]
                interp_vals[:first] = row[first]
                interp_vals[last + 1 :] = row[last]
            h_host[r, :] = interp_vals

    hvsr_spectra_clean = _to_xp_array(h_host, dtype=xp.float64) if USE_CUPY else h_host

    fcs_out = _to_host_array(fcs) if not USE_CUPY else fcs
    hvsr_out = _to_host_array(hvsr_spectra_clean) if not USE_CUPY else hvsr_spectra_clean
    return HvsrTraditional(fcs_out, hvsr_out, meta={**records[0].meta, **settings.attr_dict})



#================================
# Single azimuth processing
#================================
def single_azimuth(
    ns,
    ew,
    degrees_from_north
):
    ns = xp.asarray(ns, dtype=xp.float32)
    ew = xp.asarray(ew, dtype=xp.float32)
    radians_from_north = xp.radians(degrees_from_north)
    return ns * xp.cos(radians_from_north) + ew * xp.sin(radians_from_north)


def traditional_single_azimuth_hvsr_processing(
    records,
    settings
):
    prepare_fft_settings(records, settings)

    if getattr(settings, "window_type_and_width", None) is None:
        raise ValueError("settings.window_type_and_width must be provided and not None.")
    if not hasattr(settings, "fft_settings"):
        settings.fft_settings = {}

    records, dt_with_count = prepare_records_with_inconsistent_dt(records, settings)

    fcs = _to_xp_array(settings.smoothing["center_frequencies_in_hz"], dtype=xp.float64)
    hvsr_spectra = xp.empty((len(records), fcs.size), dtype=xp.float64)
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    hvsr_idx = 0
    processed_order = []

    for dt, count in dt_with_count.items():
        try:
            fft_frq = xp.fft.rfftfreq(settings.fft_settings.get("n", None), dt)
        except Exception:
            fft_frq = _to_xp_array(np_rfftfreq(settings.fft_settings.get("n", None), dt))

        raw_spectra = xp.empty((count * 2, fft_frq.size), dtype=xp.float64)
        hor_idx = 0
        ver_idx = count

        for org_idx, record in enumerate(records):
            if record.ns.dt_in_seconds != dt:
                continue

            processed_order.append(org_idx)

            ns_amp = sanitize_signal(_to_xp_array(record.ns.amplitude, dtype=xp.float64), xp_lib=xp)
            ew_amp = sanitize_signal(_to_xp_array(record.ew.amplitude, dtype=xp.float64), xp_lib=xp)

            try:
                h_time = single_azimuth(ns_amp, ew_amp, settings.azimuth_in_degrees)
            except Exception as e:
                logger.warning("single_azimuth failed for record %d: %s — attempting host fallback.", org_idx, e)
                ns_h = np.asarray(ns_amp) if not USE_CUPY else np.asarray(_to_host_array(ns_amp))
                ew_h = np.asarray(ew_amp) if not USE_CUPY else np.asarray(_to_host_array(ew_amp))
                h_time_host = single_azimuth(ns_h, ew_h, settings.azimuth_in_degrees)
                h_time = _to_xp_array(h_time_host, dtype=xp.float64)

            h_time = _to_xp_array(h_time, dtype=xp.float64)
            h_time = sanitize_signal(h_time, xp_lib=xp)

            try:
                h_ts = Time_Series_CUDA.from_timeseries(Time_Series_CUDA(h_time, record.ns.dt_in_seconds)) \
                       if hasattr(Time_Series_CUDA, "from_timeseries") is False else Time_Series_CUDA.from_timeseries(Time_Series_CUDA(h_time, record.ns.dt_in_seconds))
            except Exception:
                h_ts = Time_Series_CUDA(h_time, record.ns.dt_in_seconds)
            h_ts.window(*settings.window_type_and_width)
            try:
                v_ts = Time_Series_CUDA.from_timeseries(record.vt)
            except Exception:
                v_ts = Time_Series_CUDA(record.vt.amplitude, record.vt.dt_in_seconds)
            v_ts.window(*settings.window_type_and_width)

            h_amp_win = sanitize_signal(_to_xp_array(h_ts.amplitude, dtype=xp.float64), xp_lib=xp)
            v_amp_win = sanitize_signal(_to_xp_array(v_ts.amplitude, dtype=xp.float64), xp_lib=xp)

            try:
                h_spec = xp.abs(xp.fft.rfft(h_amp_win, **settings.fft_settings))
            except Exception as e:
                logger.debug("Device FFT failed for rotated horizontal (rec %d): %s — falling back to host FFT.", org_idx, e)
                h_spec = _to_xp_array(np.abs(np.fft.rfft(np.asarray(h_amp_win), **settings.fft_settings)), dtype=xp.float64)
            try:
                v_spec = xp.abs(xp.fft.rfft(v_amp_win, **settings.fft_settings))
            except Exception as e:
                logger.debug("Device FFT failed for vertical (rec %d): %s — falling back to host FFT.", org_idx, e)
                v_spec = _to_xp_array(np.abs(np.fft.rfft(np.asarray(v_amp_win), **settings.fft_settings)), dtype=xp.float64)

            raw_spectra[hor_idx] = _to_xp_array(h_spec, dtype=xp.float64)
            raw_spectra[ver_idx] = _to_xp_array(v_spec, dtype=xp.float64)
            hor_idx += 1
            ver_idx += 1

        operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
        try:
            smoothing_fn = SMOOTHING_OPERATORS[operator]
            smooth_spectra = _maybe_cpu_call(smoothing_fn, fft_frq, raw_spectra, fcs, bandwidth)
            smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)
        except Exception as e:
            logger.debug("Smoothing operator failed on xp arrays (single azimuth): %s", e)
            fft_frq_h = _to_host_array(fft_frq)
            raw_spectra_h = _to_host_array(raw_spectra)
            fcs_h = _to_host_array(fcs)
            smooth_h = SMOOTHING_OPERATORS[operator](fft_frq_h, raw_spectra_h, fcs_h, bandwidth)
            smooth_spectra = np.asarray(smooth_h, dtype=np.float64)
            if smooth_spectra.shape != raw_spectra_h.shape:
                raise RuntimeError(f"Smoothing operator returned shape {smooth_spectra.shape}, expected {raw_spectra_h.shape}")
            smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)

        h_part = smooth_spectra[:count]
        v_part = smooth_spectra[count:count + count]
        ratio = _safe_divide(h_part, v_part, eps=1e-8, xp_lib=xp, replace_nonfinite_with=np.nan)
        hvsr_spectra[hvsr_idx:hvsr_idx + count] = ratio
        hvsr_idx += count

    final_hvsr = xp.empty_like(hvsr_spectra)
    for proc_idx, orig_idx in enumerate(processed_order):
        final_hvsr[orig_idx] = hvsr_spectra[proc_idx]
    hvsr_spectra = final_hvsr

    if xp.isnan(hvsr_spectra).any():
        h_host = _to_host_array(hvsr_spectra)
        nan_counts = np.sum(~np.isfinite(h_host), axis=1)
        bad_rows = np.where(nan_counts > 0)[0]
        for r in bad_rows:
            logger.warning(
                "single_azimuth hvsr result row %d contains %d non-finite values (min/median/max = %s/%s/%s)",
                int(r), int(nan_counts[r]),
                float(np.nanmin(h_host[r])), float(np.nanmedian(h_host[r])), float(np.nanmax(h_host[r]))
            )

    fcs_out = _to_host_array(fcs) if not USE_CUPY else fcs
    hvsr_out = _to_host_array(hvsr_spectra) if not USE_CUPY else hvsr_spectra
    return HvsrTraditional(fcs_out, hvsr_out, meta={**records[0].meta, **settings.attr_dict})



#=================================
# RotDpp (ROTDpp) processing
#=================================
# Rotated Directional Power Polarization
def traditional_rotdpp_hvsr_processing(
    records,
    settings
):
    prepare_fft_settings(records, settings)

    if getattr(settings, "window_type_and_width", None) is None:
        raise ValueError("settings.window_type_and_width must be provided and not None.")
    if not hasattr(settings, "fft_settings"):
        settings.fft_settings = {}

    records, dt_with_count = prepare_records_with_inconsistent_dt(records, settings)

    fcs = _to_xp_array(settings.smoothing["center_frequencies_in_hz"], dtype=xp.float64)
    hvsr_spectra = xp.empty((len(records), fcs.size), dtype=xp.float64)
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    hvsr_idx = 0
    processed_order = []

    for dt, _count in dt_with_count.items():
        try:
            fft_frq = xp.fft.rfftfreq(settings.fft_settings.get("n", None), dt)
        except Exception:
            fft_frq = _to_xp_array(np_rfftfreq(settings.fft_settings.get("n", None), dt))

        for org_idx, record in enumerate(records):
            if record.ns.dt_in_seconds != dt:
                continue

            processed_order.append(org_idx)

            try:
                v_ts = Time_Series_CUDA.from_timeseries(record.vt)
            except Exception:
                v_ts = Time_Series_CUDA(record.vt.amplitude, record.vt.dt_in_seconds)
            v_ts.window(*settings.window_type_and_width)

            vt_amp = sanitize_signal(_to_xp_array(v_ts.amplitude, dtype=xp.float64), xp_lib=xp)
            try:
                fft_v = xp.abs(xp.fft.rfft(vt_amp, **settings.fft_settings))
            except Exception as e:
                logger.debug("Device FFT failed for vertical (rec %d): %s — falling back to host FFT.", org_idx, e)
                fft_v = _to_xp_array(np.abs(np.fft.rfft(np.asarray(vt_amp), **settings.fft_settings)), dtype=xp.float64)

            n_az = len(settings.azimuths_in_degrees)
            raw_spectra_per_record = xp.empty((n_az + 1, fft_frq.size), dtype=xp.float64)

            for idx, azimuth in enumerate(settings.azimuths_in_degrees):
                ns_amp = sanitize_signal(_to_xp_array(record.ns.amplitude, dtype=xp.float64), xp_lib=xp)
                ew_amp = sanitize_signal(_to_xp_array(record.ew.amplitude, dtype=xp.float64), xp_lib=xp)

                try:
                    h_time = single_azimuth(ns_amp, ew_amp, azimuth)
                except Exception as e:
                    logger.warning("single_azimuth failed for record %d azimuth %s: %s — using host fallback.", org_idx, azimuth, e)
                    ns_h = np.asarray(ns_amp) if not USE_CUPY else np.asarray(_to_host_array(ns_amp))
                    ew_h = np.asarray(ew_amp) if not USE_CUPY else np.asarray(_to_host_array(ew_amp))
                    h_time_host = single_azimuth(ns_h, ew_h, azimuth)
                    h_time = _to_xp_array(h_time_host, dtype=xp.float64)

                h_time = _to_xp_array(h_time, dtype=xp.float64)
                h_time = sanitize_signal(h_time, xp_lib=xp)

                try:
                    h_ts = Time_Series_CUDA.from_timeseries(Time_Series_CUDA(h_time, record.ns.dt_in_seconds))
                except Exception:
                    h_ts = Time_Series_CUDA(h_time, record.ns.dt_in_seconds)
                h_ts.window(*settings.window_type_and_width)

                h_amp_win = sanitize_signal(_to_xp_array(h_ts.amplitude, dtype=xp.float64), xp_lib=xp)
                try:
                    fft_h = xp.abs(xp.fft.rfft(h_amp_win, **settings.fft_settings))
                except Exception as e:
                    logger.debug("Device FFT failed for rotated horizontal (rec %d az %s): %s — falling back host FFT.", org_idx, azimuth, e)
                    fft_h = _to_xp_array(np.abs(np.fft.rfft(np.asarray(h_amp_win), **settings.fft_settings)), dtype=xp.float64)

                raw_spectra_per_record[idx] = fft_h

            raw_spectra_per_record[-1] = fft_v

            operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
            try:
                smoothing_fn = SMOOTHING_OPERATORS[operator]
                smooth_spectra = _maybe_cpu_call(smoothing_fn, fft_frq, raw_spectra_per_record, fcs, bandwidth)
                smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)
            except Exception as e:
                logger.debug("Smoothing operator failed on xp arrays (rotdpp rec %d): %s", org_idx, e)
                fft_frq_h = _to_host_array(fft_frq)
                raw_spectra_h = _to_host_array(raw_spectra_per_record)
                fcs_h = _to_host_array(fcs)
                smooth_h = SMOOTHING_OPERATORS[operator](fft_frq_h, raw_spectra_h, fcs_h, bandwidth)
                smooth_spectra = np.asarray(smooth_h, dtype=np.float64)
                if smooth_spectra.shape != raw_spectra_h.shape:
                    raise RuntimeError(f"Smoothing operator returned shape {smooth_spectra.shape}, expected {raw_spectra_h.shape}")
                smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)

            try:
                smooth_h = xp.percentile(smooth_spectra[:-1], settings.ppth_percentile_for_rotdpp_computation, axis=0)
            except Exception:
                smooth_h = _to_xp_array(
                    np.percentile(_to_host_array(smooth_spectra[:-1]), settings.ppth_percentile_for_rotdpp_computation, axis=0),
                    dtype=xp.float64
                )

            smooth_v = smooth_spectra[-1]

            ratio = _safe_divide(h_part, v_part, eps=1e-8, xp_lib=xp, replace_nonfinite_with=np.nan)

            hvsr_spectra[hvsr_idx] = ratio
            hvsr_idx += 1

    final_hvsr = xp.empty_like(hvsr_spectra)
    for proc_idx, orig_idx in enumerate(processed_order):
        final_hvsr[orig_idx] = hvsr_spectra[proc_idx]
    hvsr_spectra = final_hvsr

    # Diagnostic logging summary
    if xp.isnan(hvsr_spectra).any():
        h_host = _to_host_array(hvsr_spectra)
        nan_counts = np.sum(~np.isfinite(h_host), axis=1)
        bad_rows = np.where(nan_counts > 0)[0]
        for r in bad_rows:
            logger.warning(
                "rotdpp hvsr result row %d contains %d non-finite values (min/median/max = %s/%s/%s)",
                int(r), int(nan_counts[r]),
                float(np.nanmin(h_host[r])), float(np.nanmedian(h_host[r])), float(np.nanmax(h_host[r]))
            )

    fcs_out = _to_host_array(fcs) if not USE_CUPY else fcs
    hvsr_out = _to_host_array(hvsr_spectra) if not USE_CUPY else hvsr_spectra
    return HvsrTraditional(fcs_out, hvsr_out, meta={**records[0].meta, **settings.attr_dict})


#=========================
# rpds (PSD) helpers
#=========================
# Rotated Power Density Spectra
def _rpds_single_component(
    timeseries_list,
    settings
):
    if not timeseries_list:
        raise ValueError("_rpds_single_component: timeseries_list is empty.")

    if getattr(settings, "window_type_and_width", None) is None:
        raise ValueError("_rpds_single_component: settings.window_type_and_width must be provided and not None.")

    first_ts = timeseries_list[0]
    n = int(settings.fft_settings.get("n", getattr(first_ts, "n_samples", first_ts.amplitude.size)))
    nfreq = int(n // 2) + 1

    psd = xp.zeros(nfreq, dtype=xp.float64)

    for idx, tseries in enumerate(timeseries_list):
        try:
            ts_copy = Time_Series_CUDA.from_timeseries(tseries)
        except Exception:
            ts_copy = Time_Series_CUDA(_to_xp_array(tseries.amplitude, dtype=xp.float64), getattr(tseries, "dt_in_seconds", None))

        ts_copy.window(*settings.window_type_and_width)

        amp = sanitize_signal(_to_xp_array(ts_copy.amplitude, dtype=xp.float64), xp_lib=xp)

        try:
            fft = xp.fft.rfft(amp, **settings.fft_settings)
        except Exception as e:
            logger.debug("_rpds_single_component: device FFT failed for timeseries %d: %s - falling back to host FFT.", idx, e)
            amp_host = np.asarray(_to_host_array(amp)) if USE_CUPY else np.asarray(amp)
            fft_host = np.fft.rfft(amp_host, **{k: v for k, v in settings.fft_settings.items() if k != "n"})
            fft = _to_xp_array(fft_host, dtype=xp.complex128)

        try:
            psd += (xp.abs(fft) ** 2).astype(xp.float64)
        except Exception:
            fft_h = np.asarray(_to_host_array(fft))
            psd_host = (np.abs(fft_h) ** 2).astype(np.float64)
            psd = _to_xp_array(np.asarray(_to_host_array(psd)) + psd_host, dtype=xp.float64)

    try:
        window_amp = xp.ones_like(_to_xp_array(ts_copy.amplitude, dtype=xp.float64))
    except Exception:
        window_amp = xp.ones(n, dtype=xp.float64)

    try:
        window_ts = Time_Series_CUDA(window_amp, dt_in_seconds=getattr(ts_copy, "dt_in_seconds", None))
        window_ts.window(*settings.window_type_and_width)
        w_arr = _to_xp_array(window_ts.amplitude, dtype=xp.float64)
        window_scaling_factor = xp.mean(w_arr ** 2)
    except Exception as e:
        logger.warning("_rpds_single_component: failed to construct/apply window; using fallback scaling factor 1.0. Error: %s", e)
        window_scaling_factor = xp.asarray(1.0, dtype=xp.float64)

    try:
        wsf_scalar = float(window_scaling_factor.item() if hasattr(window_scaling_factor, "item") else xp.asnumpy(window_scaling_factor))
    except Exception:
        try:
            wsf_scalar = float(xp.asnumpy(window_scaling_factor))
        except Exception:
            wsf_scalar = 1.0

    if (not np.isfinite(wsf_scalar)) or (wsf_scalar <= 1e-12):
        logger.warning("_rpds_single_component: computed window_scaling_factor is non-finite or too small (%s). Using 1.0 to avoid division by zero.", wsf_scalar)
        wsf_scalar = 1.0

    psd = psd / float(wsf_scalar)

    try:
        n_samples_ref = int(getattr(first_ts, "n_samples", getattr(first_ts, "amplitude").size))
        fs_ref = float(getattr(first_ts, "fs", 1.0 / getattr(first_ts, "dt_in_seconds", 1.0)))
    except Exception:
        n_samples_ref = n
        fs_ref = 1.0 / (getattr(first_ts, "dt_in_seconds", 1.0))

    if n_samples_ref <= 0:
        logger.warning("_rpds_single_component: reference n_samples <= 0 (%s). Using n reference = %d", n_samples_ref, n)
        n_samples_ref = n

    try:
        psd = psd / float(n_samples_ref)
        psd = psd / float(fs_ref)
        psd = psd * 2.0
    except Exception:
        psd_host = np.asarray(_to_host_array(psd), dtype=np.float64)
        psd_host = psd_host / float(n_samples_ref)
        psd_host = psd_host / float(fs_ref)
        psd_host = psd_host * 2.0
        psd = _to_xp_array(psd_host, dtype=xp.float64)

    try:
        count_ts = float(len(timeseries_list))
        if count_ts == 0:
            raise ValueError("_rpds_single_component: zero-length timeseries_list passed.")
        psd = psd / count_ts
    except Exception as e:
        logger.warning("_rpds_single_component: averaging failed: %s. Returning un-averaged PSD.", e)

    try:
        non_finite_mask = ~xp.isfinite(psd)
        if bool(non_finite_mask.any()):
            logger.warning("_rpds_single_component: PSD contains non-finite values; replacing with 0.0.")
            psd = xp.where(xp.isfinite(psd), psd, 0.0)
    except Exception:
        psd_h = np.asarray(_to_host_array(psd))
        if not np.all(np.isfinite(psd_h)):
            logger.warning("_rpds_single_component: PSD contains non-finite values (host); replacing with 0.0.")
            psd_h[~np.isfinite(psd_h)] = 0.0
        psd = _to_xp_array(psd_h, dtype=xp.float64)

    return psd


def rpsd(
    records,
    settings
):
    if not records:
        raise ValueError("rpsd:`records` is empty.")

    prepare_fft_settings(records, settings)

    if not hasattr(settings, "fft_settings"):
        settings.fft_settings = {}

    try:
        fft_frq = xp.fft.rfftfreq(settings.fft_settings.get("n", None), records[0].vt.dt_in_seconds)
    except Exception:
        fft_frq = _to_xp_array(np_rfftfreq(settings.fft_settings.get("n", None), records[0].vt.dt_in_seconds))

    try:
        dts = [rec.vt.dt_in_seconds for rec in records]
        if not all(abs(dt - dts[0]) < 1e-8 for dt in dts):
            logger.warning("rpsd: records have inconsistent dt values; results may be unreliable.")
    except Exception:
        pass

    psd_ns = _rpds_single_component([record.ns for record in records], settings)
    psd_ew = _rpds_single_component([record.ew for record in records], settings)
    psd_vt = _rpds_single_component([record.vt for record in records], settings)

    try:
        nfreq = int(fft_frq.size)
    except Exception:
        nfreq = int(_to_host_array(fft_frq).size)

    def _ensure_len(x, name):
        x = _to_xp_array(x, dtype=xp.float64)
        if x.size != nfreq:
            raise RuntimeError(f"rpsd: {name} PSD length ({x.size}) does not match frequency vector length ({nfreq}).")
        return x

    psd_ns = _ensure_len(psd_ns, "ns")
    psd_ew = _ensure_len(psd_ew, "ew")
    psd_vt = _ensure_len(psd_vt, "vt")

    fft_frq_out = fft_frq

    if getattr(settings, "smoothing", None) is not None:
        operator = settings.smoothing["operator"]
        bandwidth = settings.smoothing["bandwidth"]
        fcs = _to_xp_array(settings.smoothing["center_frequencies_in_hz"], dtype=xp.float64)

        spectra = xp.empty((3,nfreq), dtype=xp.float64)
        spectra[0] = psd_ns
        spectra[1] = psd_ew
        spectra[2] = psd_vt

        try:
            smooth_spectra = _maybe_cpu_call(SMOOTHING_OPERATORS[operator], fft_frq, spectra, fcs, bandwidth)
            smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)
            if smooth_spectra.shape != spectra.shape:
                raise RuntimeError(f"Smoothing operator returned shape {smooth_spectra.shape}; expected {spectra.shape}")
            fft_frq_out = fcs
            psd_ns, psd_ew, psd_vt = smooth_spectra[0], smooth_spectra[1], smooth_spectra[2]
        except Exception as e:
            logger.debug("rpsd: device smoothing failed or returned bad shape: %s - falling back to host smoothing.", e)
            fft_frq_h = _to_host_array(fft_frq)
            spectra_h = np.vstack((_to_host_array(psd_ns), _to_host_array(psd_ew), _to_host_array(psd_vt)))
            fcs_h = np.asarray(settings.smoothing["center_frequencies_in_hz"], dtype=np.float64)
            smooth_h = SMOOTHING_OPERATORS[operator](fft_frq_h, spectra_h, fcs_h, bandwidth)
            smooth_h = np.asarray(smooth_h, dtype=np.float64)
            if smooth_h.shape != spectra_h.shape:
                raise RuntimeError(f"Host smoothing operator returned shape {smooth_h.shape}; expected {spectra_h.shape}")
            smooth_spectra = _to_xp_array(smooth_h, dtype=xp.float64)
            fft_frq_out = _to_xp_array(fcs_h, dtype=xp.float64)
            psd_ns, psd_ew, psd_vt = smooth_spectra[0], smooth_spectra[1], smooth_spectra[2]


    def _clean_and_warn(arr, name):
        try:
            arr = _to_xp_array(arr, dtype=xp.float64)
            nonfinite = ~xp.isfinite(arr)
            if bool(nonfinite.any()):
                logger.warning("rpsd: %s PSD contains %d non-finite values; replacing with 0.0", name, int(nonfinite.sum()))
                arr = xp.where(xp.isfinite(arr), arr, 0.0)
        except Exception:
            arr_h = np.asarray(_to_host_array(arr))
            mask = ~np.isfinite(arr_h)
            if mask.any():
                logger.warning("rpsd: %s PSD (host) contains %d non-finite values; replacing with 0.0", name, int(mask.sum()))
                arr_h[mask] = 0.0
            arr = _to_xp_array(arr_h, dtype=xp.float64)
        return arr

    psd_ns = _clean_and_warn(psd_ns, "ns")
    psd_ew = _clean_and_warn(psd_ew, "ew")
    psd_vt = _clean_and_warn(psd_vt, "vt")

    if not USE_CUPY:
        fft_frq_host = _to_host_array(fft_frq_out)
        psd_ns_host = _to_host_array(psd_ns)
        psd_ew_host = _to_host_array(psd_ew)
        psd_vt_host = _to_host_array(psd_vt)
        return dict(
            ns=Psd(fft_frq_host, psd_ns_host),
            ew=Psd(fft_frq_host, psd_ew_host),
            vt=Psd(fft_frq_host, psd_vt_host)
        )
    else:
        return dict(
            ns=Psd(fft_frq_out, psd_ns),
            ew=Psd(fft_frq_out, psd_ew),
            vt=Psd(fft_frq_out, psd_vt)
        )


#=============================
# Diffuse-field processing
#=============================
def diffuse_field_hvsr_processing(
    records,
    settings
):
    prepare_fft_settings(records, settings)
    records, dt_with_count = prepare_records_with_inconsistent_dt(records, settings)

    if len(dt_with_count.keys()) > 1:
        msg = "You cannot use diffuse field processing with records with "
        msg += "dissimilar time steps. Try setting "
        msg += "'handle_dissimilar_time_steps_by' to "
        msg += "'keeping_smallest_time_step' or 'keeping_majority_time_step' "
        msg += "to only process those records with similar time steps."
        raise ValueError(msg)

    fcs = _to_xp_array(settings.smoothing["center_frequencies_in_hz"], dtype=xp.float64)
    check_nyquist_frequency(max(dt_with_count.keys()), fcs)

    try:
        fft_frq = xp.fft.rfftfreq(settings.fft_settings["n"], records[0].vt.dt_in_seconds)
    except Exception:
        fft_frq = _to_xp_array(np_rfftfreq(settings.fft_settings["n"], records[0].vt.dt_in_seconds))

    psd_ns = _rpds_single_component([record.ns for record in records], settings)
    psd_ew = _rpds_single_component([record.ew for record in records], settings)
    psd_vt = _rpds_single_component([record.vt for record in records], settings)

    operator, bandwidth = settings.smoothing["operator"], settings.smoothing["bandwidth"]
    spectra = xp.empty((2, fft_frq.size), dtype=xp.float64)
    spectra[0] = psd_ns + psd_ew
    spectra[1] = psd_vt
    smooth_spectra = _maybe_cpu_call(SMOOTHING_OPERATORS[operator], fft_frq, spectra, fcs, bandwidth)
    smooth_spectra = _to_xp_array(smooth_spectra, dtype=xp.float64)
    hor = smooth_spectra[0]
    ver = smooth_spectra[1]

    with xp.errstate(divide="ignore", invalid="ignore"):
        out = xp.sqrt(hor / ver)
        out = xp.where(xp.isfinite(out), out, xp.nan)

    return HvsrDiffuseField(_to_host_array(fcs) if not USE_CUPY else fcs,
                            _to_host_array(out) if not USE_CUPY else out,
                            meta={**records[0].meta, **settings.attr_dict})


#==========================
# registry & dispatch
#==========================
PROCESSING_METHODS = {
    "traditional": traditional_hvsr_processing,
    "azimuthal": lambda recs, s: azimuthal_hvsr_processing(recs, s),
    "diffuse_field": diffuse_field_hvsr_processing,
    "psd": rpsd,
}

def azimuthal_hvsr_processing(
    records,
    settings
):
    prepare_fft_settings(records, settings)
    single_azimuth_settings = HvsrTraditionalSingleAzimuthProcessingSettings(
        window_type_and_width=settings.window_type_and_width,
        smoothing=settings.smoothing,
        handle_dissimilar_time_steps_by=settings.handle_dissimilar_time_steps_by,
        fft_settings=settings.fft_settings,
    )
    hvsr_per_azimuth = []
    for azimuth in settings.azimuths_in_degrees:
        single_azimuth_settings.azimuth_in_degrees = azimuth
        hvsr = traditional_single_azimuth_hvsr_processing(records, single_azimuth_settings)
        hvsr_per_azimuth.append(hvsr)
    return HvsrAzimuthal(hvsr_per_azimuth, settings.azimuths_in_degrees, meta={**records[0].meta, **settings.attr_dict})

def process(
    records,
    settings
):
    func = PROCESSING_METHODS.get(settings.processing_method)
    if func is None:
        raise KeyError(f"Unknown processing method: {settings.processing_method}")
    return func(records, settings)



####################################
####################################
# Post-Processing
####################################
####################################
def _to_xp_array(arr, dtype=None):
    if USE_CUPY:
        if isinstance(arr, cp.ndarray):
            return arr.astype(dtype) if dtype is not None else arr
        return cp.asarray(np.asarray(arr), dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype)


def _to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    try:
        return np.asarray(arr)
    except Exception:
        return np.asarray(list(arr))


def _to_python_float(x):
    if x is None:
        return float("nan")
    if USE_CUPY:
        try:
            if isinstance(x, cp.generic):
                return float(cp.asnumpy(x))
            if isinstance(x, cp.ndarray) and x.ndim == 0:
                return float(cp.asnumpy(x).item())
        except Exception:
            pass
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return float(arr.item())
        if arr.size == 1:
            return float(arr.ravel()[0])
        raise TypeError("not a scalar")
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


def _mask_positive_finite_freq(
        freq,
        *arrays,
        require_min_len=3
):
    freq_host = _to_host_array(freq)
    try:
        freq_host = np.asarray(freq_host, dtype=float)
    except Exception:
        raise ValueError("Frequency axis could notbe converted to host array.")

    mask = np.isfinite(freq_host) & (freq_host > 0.0)
    if mask.sum() < require_min_len:
        raise ValueError(f"Insufficient positive finite frequency points ({mask.sum()}); cannot plot on log axis.")

    freq_masked = freq_host[mask]
    arrays_masked = []
    for a in arrays:
        a_h = _to_host_array(a)
        a_h = np.asarray(a_h)
        if a_h.ndim == 2 and a_h.shape[1] == freq_host.size:
            arrays_masked.append(a_h[:, mask])
        elif a_h.ndim == 1 and a_h.size == freq_host.size:
            arrays_masked.append(a_h[mask])
        else:
            try:
                arrays_masked.append(np.asarray(a_h)[mask])
            except Exception:
                raise ValueError("Array length does not match frequency axis; cannot mask/match.")
    return freq_masked, arrays_masked


#=============
# Constants
#=============
DEFAULT_KWARGS = {
    "individual_valid_hvsr_curve": {
        "linewidth": 0.3,
        "color": "#888888",
        "label": "Accepted HVSR Curve",
    },
    "individual_invalid_hvsr_curve": {
        "linewidth": 0.3,
        "color": "red",
        "label": "Rejected HVSR Curve",
    },
    "mean_hvsr_curve": {
        "linewidth": 1.3,
        "color": "black",
        "label": "Mean Curve",
    },
    "nth_std_mean_hvsr_curve": {
        "linewidth": 1.3,
        "color": "black",
        "label": r"$\pm$ 1 Std Curve",
        "linestyle": "--",
    },
    "nth_std_frequency_range_normal": {
        "color": "#ff8080",
        "label": "r$\mu_{fn} \pm \sigma_{fn}$",
    },
    "nth_std_frequency_range_lognormal": {
        "color": "#ff8080",
        "label": r"$(\mu_{ln,fn} \pm \sigma_{ln,fn})^*$",
    },
    "peak_mean_hvsr_curve": {
        "linestyle": "",
        "marker": "D",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc}$",
    },
    "peak_mean_hvsr_curve_azimuthal": {
        "linestyle": "",
        "marker": "D",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,az}$",
    },
    "peak_mean_hvsr_curve_azimuthal_2d": {
        "linestyle": "",
        "marker": "s",
        "markersize": 4,
        "markerfacecolor": "lightgreen",
        "markeredgewidth": 1,
        "markeredgecolor": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,\alpha}$",
    },
    "peak_mean_hvsr_curve_azimuthal_3d": {
        "marker": "s",
        "s": 16,
        "c": "lightgreen",
        "edgecolors": "black",
        "zorder": 4,
        "label": r"$f_{n,mc,\alpha}$",
    },
    "peak_individual_valid_hvsr_curve": {
        "linestyle": "",
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "white",
        "markeredgewidth": 0.5,
        "markeredgecolor": "black",
        "zorder": 2,
        "label": r"$f_{n,i,accepted}$",
    },
    "peak_individual_invalid_hvsr_curve": {
        "linestyle": "",
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "red",
        "markeredgewidth": 0.5,
        "markeredgecolor": "white",
        "zorder": 2,
        "label": r"$f_{n,i,rejected}$",
    }
}

HVSRPY_MPL_STYLE = {
    "axes.titlesize": 8,
    "lines.linewidth": 0.75,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "font.size": 8,
    "legend.handlelength": 1.5,
    "legend.columnspacing": 0.5,
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.framealpha": 1,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


#================================
# Plot helpers (GPU-aware)
#================================
"""
def _plot_individual_hvsr_curves(
    ax,
    hvsr,
    valid=True,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        raise NotImplementedError("Can only plot HvsrTraditional and HvsrAzimuthal objects.")

    default_kwargs = (DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy()
                      if valid else DEFAULT_KWARGS["individual_invalid_hvsr_curve"].copy())
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    first_labeled = True

    for hv in hvsrs:
        try:
            freq_host = _to_host_array(hv.frequency)
            freq_host = np.asarray(freq_host, dtype=float)
        except Exception:
            logger.debug("_plot_individual_hvsr_curves: invalid frequency axis for hv; skipping.")
            continue

        try:
            mask_host = _to_host_array(hv.valid_window_boolean_mask)
            mask_host = np.asarray(mask_host, dtype=bool)
        except Exception:
            mask_host = None

        if hv.amplitude is None:
            continue

        amp_all = _to_host_array(hv.amplitude)
        amp_all = np.asarray(amp_all, dtype=float)

        if mask_host is None:
            selected_rows = np.arange(amp_all.shape[0])
        else:
            if mask_host.shape[0] != amp_all.shape[0]:
                logger.debug("_plot_individual_hvsr_curves: mask length (%d) != amplitude rows (%d). Using all rows.",
                             mask_host.shape[0], amp_all.shape[0])
                selected_rows = np.arange(amp_all.shape[0])
            else:
                if valid:
                    selected_rows = np.where(mask_host)[0]
                else:
                    selected_rows = np.where(~mask_host)[0]

        if selected_rows.size == 0:
            continue

        for ridx in selected_rows:
            amplitude = amp_all[ridx, :]
            if amplitude is None:
                continue
            amp = np.asarray(amplitude, dtype=float)

            valid_freq_mask = np.isfinite(freq_host) & (freq_host > 0.0)
            valid_amp_mask = np.isfinite(amp)
            final_mask = valid_freq_mask & valid_amp_mask

            if not np.any(final_mask):
                logger.debug("_plot_individual_hvsr_curves: window %d has no finite points -> skip", ridx)
                continue

            f_plot = freq_host[final_mask]
            amp_plot = amp[final_mask]

            local_kwargs = dict(plot_kwargs)
            if "label" in local_kwargs:
                local_kwargs.pop("label")

            if first_labeled:
                ax.plot(f_plot, amp_plot, label="HVSR Curve", **local_kwargs)
                first_labeled = False
            else:
                ax.plot(f_plot, amp_plot, **{**local_kwargs, "label": None})

            try:
                ax.plot(f_plot, amp_plot, **local_kwargs)
            except Exception as e:
                logger.debug("_plot_individual_hvsr_curves: plotting failed for window %d: %s", ridx, e)
                continue

    return None
"""

def _plot_individual_hvsr_curves(
    ax,
    hvsr,
    valid=True,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        raise NotImplementedError("Can only plot HvsrTraditional and HvsrAzimuthal objects.")

    default_kwargs = (DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy()
                      if valid else DEFAULT_KWARGS["individual_invalid_hvsr_curve"].copy())
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    first_labeled = True

    for hv in hvsrs:
        try:
            freq_host = _to_host_array(hv.frequency)
            freq_host = np.asarray(freq_host, dtype=float)
        except Exception:
            logger.debug("_plot_individual_hvsr_curves: invalid frequency axis for hv; skipping.")
            continue

        try:
            mask_host = _to_host_array(hv.valid_window_boolean_mask)
            mask_host = np.asarray(mask_host, dtype=bool)
        except Exception:
            mask_host = None

        if hv.amplitude is None:
            continue

        amp_all = _to_host_array(hv.amplitude)
        if isinstance(amp_all, list):
            try:
                amp_all = np.vstack([
                    np.asarray(a, dtype=float) for a in amp_all if a is not None
                ])
            except Exception:
                logger.warning("_plot_individual_hvsr_curves: failed stacking amplitude list")
                continue
        amp_all = np.asarray(amp_all, dtype=float)
        if amp_all.ndim == 1:
            amp_all = amp_all[np.newaxis, :]

        print("==== DEBUG HVSR ====")
        print("Amplitude shape:", np.shape(amp_all))
        try:
            print("Amplitude min/max:", np.nanmin(amp_all), np.nanmax(amp_all))
        except:
            print("Amplitude min/max: ERROR")
        print("Any finite:", np.any(np.isfinite(amp_all)))

        if mask_host is None:
            selected_rows = np.arange(amp_all.shape[0])
        else:
            if mask_host.shape[0] != amp_all.shape[0]:
                logger.debug("_plot_individual_hvsr_curves: mask length (%d) != amplitude rows (%d). Using all rows.",
                             mask_host.shape[0], amp_all.shape[0])
                selected_rows = np.arange(amp_all.shape[0])
            else:
                if valid:
                    selected_rows = np.where(mask_host)[0]
                else:
                    selected_rows = np.where(~mask_host)[0]

        if selected_rows.size == 0:
            continue

        for ridx in selected_rows:
            amplitude = amp_all[ridx, :]
            if amplitude is None:
                continue
            amp = np.asarray(amplitude, dtype=float)

            valid_freq_mask = np.isfinite(freq_host) & (freq_host > 0.0)
            valid_amp_mask = np.isfinite(amp)
            final_mask = valid_freq_mask & valid_amp_mask

            if not np.any(final_mask):
                logger.debug("_plot_individual_hvsr_curves: window %d has no finite points -> skip", ridx)
                continue

            f_plot = freq_host[final_mask]
            amp_plot = amp[final_mask]

            local_kwargs = dict(plot_kwargs)
            if "label" in local_kwargs:
                local_kwargs.pop("label")

            if first_labeled:
                ax.plot(f_plot, amp_plot, label="HVSR Curve", **local_kwargs)
                first_labeled = False
            else:
                ax.plot(f_plot, amp_plot, **{**local_kwargs, "label": None})

            try:
                ax.plot(f_plot, amp_plot, **local_kwargs)
            except Exception as e:
                logger.debug("_plot_individual_hvsr_curves: plotting failed for window %d: %s", ridx, e)
                continue

    return None


"""
def _plot_individual_hvsr_curves(
    ax,
    hvsr,
    valid=True,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        raise NotImplementedError(
            "Can only plot HvsrTraditional and HvsrAzimuthal objects."
        )

    default_kwargs = (
        DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy()
        if valid else DEFAULT_KWARGS["individual_invalid_hvsr_curve"].copy()
    )
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    plot_kwargs = dict(plot_kwargs)
    plot_kwargs.pop("label", None)

    first_labeled = True

    for hv in hvsrs:
        try:
            freq_host = np.asarray(_to_host_array(hv.frequency), dtype=float).ravel()
        except Exception:
            logger.debug(
                "_plot_individual_hvsr_curves: invalid frequency axis for hv; skipping."
            )
            continue

        if hv.amplitude is None:
            continue

        try:
            amp_all = np.asarray(_to_host_array(hv.amplitude), dtype=float)
        except Exception:
            logger.debug(
                "_plot_individual_hvsr_curves: invalid amplitude array for hv; skipping."
            )
            continue

        if amp_all.ndim == 1:
            amp_all = amp_all[np.newaxis, :]

        if amp_all.size == 0:
            continue

        nfreq = freq_host.size
        namp = amp_all.shape[1]
        if nfreq != namp:
            n = min(nfreq, namp)
            logger.debug(
                "_plot_individual_hvsr_curves: frequency length (%d) != amplitude length (%d). "
                "Truncating to %d.",
                nfreq, namp, n
            )
            freq_use = freq_host[:n]
            amp_use = amp_all[:, :n]
        else:
            freq_use = freq_host
            amp_use = amp_all

        try:
            mask_host = np.asarray(_to_host_array(hv.valid_window_boolean_mask), dtype=bool)
        except Exception:
            mask_host = None

        if mask_host is None:
            selected_rows = np.arange(amp_use.shape[0])
        else:
            if mask_host.shape[0] != amp_use.shape[0]:
                logger.debug(
                    "_plot_individual_hvsr_curves: mask length (%d) != amplitude rows (%d). "
                    "Using all rows.",
                    mask_host.shape[0], amp_use.shape[0]
                )
                selected_rows = np.arange(amp_use.shape[0])
            else:
                selected_rows = np.where(mask_host)[0] if valid else np.where(~mask_host)[0]

        if selected_rows.size == 0:
            continue

        for ridx in selected_rows:
            amp = np.asarray(amp_use[ridx, :], dtype=float)
            final_mask = np.isfinite(freq_use) & (freq_use > 0.0) & np.isfinite(amp)

            if not np.any(final_mask):
                logger.debug(
                    "_plot_individual_hvsr_curves: window %d has no finite points -> skip",
                    ridx
                )
                continue

            f_plot = freq_use[final_mask]
            amp_plot = amp[final_mask]

            if f_plot.size == 0 or amp_plot.size == 0:
                continue

            if first_labeled:
                ax.plot(f_plot, amp_plot, label="HVSR Curve", **plot_kwargs)
                first_labeled = False
            else:
                ax.plot(f_plot, amp_plot, **plot_kwargs)

    return None
"""

def _plot_peak_individual_hvsr_curve(
    ax,
    hvsr,
    valid=True,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    elif isinstance(hvsr, HvsrDiffuseField):
        return None
    else:
        raise NotImplementedError("Can only plot HvsrTraditional and HvsrAzimuthal objects.")

    default_kwargs = (DEFAULT_KWARGS["peak_individual_valid_hvsr_curve"].copy()
                      if valid else DEFAULT_KWARGS["peak_individual_invalid_hvsr_curve"].copy())
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    first_labeled = True

    for hv in hvsrs:
        try:
            mask_host = _to_host_array(hv.valid_peak_boolean_mask)
            mask_host = np.asarray(mask_host, dtype=bool)
        except Exception:
            mask_host = None

        try:
            freq_all = _to_host_array(getattr(hv, "_main_peak_frq", None))
            amp_all = _to_host_array(getattr(hv, "_main_peak_amp", None))
            freq_all = np.asarray(freq_all, dtype=float) if freq_all is not None else np.array([], dtype=float)
            amp_all = np.asarray(amp_all, dtype=float) if amp_all is not None else np.array([], dtype=float)
        except Exception:
            logger.debug("_plot_peak_individual_hvsr_curve: failed to fetch peak arrays for hv; skipping.")
            continue

        if freq_all.size == 0 or amp_all.size == 0:
            continue

        if mask_host is None:
            sel_idx = np.arange(freq_all.size)
        else:
            if mask_host.shape[0] != freq_all.shape[0]:
                logger.debug("_plot_peak_individual_hvsr_curve: mask length (%d) != peak rows (%d); using all rows.",
                             mask_host.shape[0], freq_all.shape[0])
                sel_idx = np.arange(freq_all.size)
            else:
                sel_idx = np.where(mask_host)[0] if valid else np.where(~mask_host)[0]

        if sel_idx.size == 0:
            continue

        sel_freq = freq_all[sel_idx]
        sel_amp = amp_all[sel_idx]

        finite_mask = np.isfinite(sel_freq) & np.isfinite(sel_amp) & (sel_freq > 0.0)

        if not np.any(finite_mask):
            logger.debug("_plot_peak_individual_hvsr_curve: selected peaks contain no finite points; skipping.")
            continue

        f_plot = sel_freq[finite_mask]
        a_plot = sel_amp[finite_mask]

        local_kwargs = dict(plot_kwargs)
        if first_labeled:
            first_labeled = False
        else:
            local_kwargs.pop("label", None)

        try:
            ax.plot(f_plot, a_plot, **local_kwargs)
        except Exception as e:
            logger.debug("_plot_peak_individual_hvsr_curve: plotting failed: %s", e)
            continue

    return None


def _plot_peak_mean_hvsr_curve(
    ax,
    hvsr,
    distribution="lognormal",
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrAzimuthal):
        default_kwargs = DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal"].copy()
    else:
        default_kwargs = DEFAULT_KWARGS["peak_mean_hvsr_curve"].copy()

    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}

    f_peak, a_peak = hvsr.mean_curve_peak(distribution=distribution)

    try:
        f_host = _to_host_array(f_peak)
    except Exception:
        f_host = _to_python_float(f_peak)
    try:
        a_host = _to_host_array(a_peak)
    except Exception:
        a_host = _to_python_float(a_peak)

    try:
        if np.ndim(f_host) == 0:
            f_host = float(np.asarray(f_host).item())
        elif np.size(f_host) == 1:
            f_host = float(np.asarray(f_host).ravel()[0])
    except Exception:
        f_host = _to_python_float(f_host)

    try:
        if np.ndim(a_host) == 0:
            a_host = float(np.asarray(a_host).item())
        elif np.size(a_host) == 1:
            a_host = float(np.asarray(a_host).ravel()[0])
    except Exception:
        a_host = _to_python_float(a_host)

    if f_host is None or a_host is None:
        warnings.warn("Peak values are None. Skipping peak plot.")
        return
    if not np.isfinite(f_host) or not np.isfinite(a_host):
        warnings.warn("Peak values are not finite (None/NaN/Inf). Skipping peak plot.")
        return

    ax.plot(f_host, a_host, **plot_kwargs)


def _plot_mean_hvsr_curve(
    ax,
    hvsr,
    distribution="lognormal",
    plot_kwargs=None
):
    default_kwargs = DEFAULT_KWARGS["mean_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}
    freq = _to_host_array(hvsr.frequency)
    mean_curve = _to_host_array(hvsr.mean_curve(distribution=distribution))
    try:
        f_plot, (mean_plot,) = _mask_positive_finite_freq(freq, mean_curve)
    except ValueError as e:
        warnings.warn(f"Mean curve not plotted: {e}")
        return
    ax.plot(f_plot, mean_plot, **plot_kwargs)


"""
def _plot_nth_std_hvsr_curve(
    ax,
    hvsr,
    distribution="lognormal",
    n=1.,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrDiffuseField):
        return None
    default_kwargs = DEFAULT_KWARGS["nth_std_mean_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}
    freq = _to_host_array(hvsr.frequency)
    mean_curve = _to_host_array(hvsr.mean_curve(distribution=distribution))
    try:
        f_plot, (mean_plot,) = _mask_positive_finite_freq(freq, mean_curve)
    except ValueError as e:
        warnings.warn(f"Mean curve not plotted: {e}")
        return
    ax.plot(f_plot, mean_plot, **plot_kwargs)
"""


def _plot_nth_std_hvsr_curve(
    ax,
    hvsr,
    distribution="lognormal",
    n=1.,
    plot_kwargs=None
):
    if isinstance(hvsr, HvsrDiffuseField):
        return None
    default_kwargs = DEFAULT_KWARGS["nth_std_mean_hvsr_curve"].copy()
    plot_kwargs = default_kwargs if plot_kwargs is None else {**default_kwargs, **plot_kwargs}
    freq = _to_host_array(hvsr.frequency)
    nth_curve = _to_host_array(hvsr.nth_std_curve(n=n, distribution=distribution))
    try:
        f_plot, (nth_plot,) = _mask_positive_finite_freq(freq, nth_curve)
    except ValueError as e:
        warnings.warn(f"nth std curve not plotted: {e}")
        return
    ax.plot(f_plot, nth_plot, **plot_kwargs)


def _plot_nth_std_frequency_range(
    ax,
    hvsr,
    distribution="lognormal",
    n=1.,
    fill_kwargs=None
):
    if isinstance(hvsr, HvsrDiffuseField):
        return None
    default_kwargs = DEFAULT_KWARGS[f"nth_std_frequency_range_{distribution}"].copy()
    fill_kwargs = default_kwargs if fill_kwargs is None else {**default_kwargs, **fill_kwargs}
    _, y_max = ax.get_ylim()
    f_min = _to_python_float(hvsr.nth_std_fn_frequency(n=-n, distribution=distribution))
    f_max = _to_python_float(hvsr.nth_std_fn_frequency(n=+n, distribution=distribution))
    ax.fill([f_min, f_min, f_max, f_max], [0, 100, 100, 0], **fill_kwargs)
    ax.set_ylim((0, np.ceil(y_max)))


def _plot_resonance_pdf(
    ax,
    hvsr,
    distribution="lognormal",
    contour_kwargs=None
):
    mean_frequency = _to_python_float(hvsr.Mean_fn_Frequency(distribution=distribution))
    mean_amplitude = _to_python_float(hvsr.mean_fn_amplitude(distribution=distribution))
    cov = _to_host_array(hvsr.cov_fn(distribution=distribution))

    if not np.isfinite(mean_frequency) or not np.isfinite(mean_amplitude):
        warnings.warn("Mean frequency or amplitude not finite; skipping resonance PDF plot.")
        return
    if cov is None or not hasattr(cov, "__array__"):
        warnings.warn("Covariance matrix invalid or missing; skipping resonance PDF plot.")
        return
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        warnings.warn(f"Covariance has incorrect shape {cov.shape}; expected (2,2). Skipping.")
        return

    is_log = (distribution != "normal")
    if is_log:
        if mean_frequency <= 0 or mean_amplitude <= 0:
            warnings.warn("Non-positive mean for log-distribution; skipping resonance PDF plot.")
            return
        mu = np.array([np.log(mean_frequency), np.log(mean_amplitude)], dtype=float)
    else:
        mu = np.array([mean_frequency, mean_amplitude], dtype=float)

    eps = 1e-12
    cov = 0.5 * (cov + cov.T)
    det = np.linalg.det(cov)
    if not np.isfinite(det) or det <= 0:
        cov = cov + np.eye(2) * max(eps, abs(cov).max() * 1e-8)
        det = np.linalg.det(cov)
        if det <= 0 or not np.isfinite(det):
            warnings.warn("Covariance matrix singular or invalid after regularization; skipping PDF plot.")
            return

    inv_cov = np.linalg.inv(cov)
    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det))

    std_frequency = np.sqrt(max(cov[0, 0], 0.0))
    std_amplitude = np.sqrt(max(cov[1, 1], 0.0))
    if not np.isfinite(std_frequency) or not np.isfinite(std_amplitude):
        warnings.warn("Invalid standard deviations from covariance; skipping.")
        return

    f_lower = mu[0] - 3.0 * std_frequency
    f_upper = mu[0] + 3.0 * std_frequency
    a_lower = mu[1] - 3.0 * std_amplitude
    a_upper = mu[1] + 3.0 * std_amplitude

    if not (np.isfinite(f_lower) and np.isfinite(f_upper) and f_upper > f_lower):
        f_center = mu[0]
        f_lower, f_upper = f_center - 1e-3, f_center + 1e-3
    if not (np.isfinite(a_lower) and np.isfinite(a_upper) and a_upper > a_lower):
        a_center = mu[1]
        a_lower, a_upper = a_center - 1e-3, a_center + 1e-3

    nx = contour_kwargs.get("nx", 50) if contour_kwargs else 50
    ny = contour_kwargs.get("ny", 50) if contour_kwargs else 50
    x = np.linspace(f_lower, f_upper, nx)
    y = np.linspace(a_lower, a_upper, ny)
    X, Y = np.meshgrid(x, y)

    XY = np.stack((X.ravel(), Y.ravel()), axis=1)
    diff = XY - mu.reshape((1, 2))
    t = diff.dot(inv_cov) * diff
    mahal = np.sum(t, axis=1)

    pdf_vals = norm_const * np.exp(-0.5 * mahal)
    pdf_mat = pdf_vals.reshape(X.shape)

    if is_log:
        F = np.exp(X)
        A = np.exp(Y)
        F_safe = np.where(F <= 0.0, np.finfo(float).tiny, F)
        A_safe = np.where(A <= 0.0, np.finfo(float).tiny, A)
        pdf_mat = pdf_mat / (F_safe * A_safe)
        plot_X = F
        plot_Y = A
    else:
        plot_X = X
        plot_Y = Y

    pdf_mat = np.where(np.isfinite(pdf_mat), pdf_mat, 0.0)

    try:
        cmap = cm.get_cmap("Reds").copy()
        cmap.set_under("white")
    except Exception:
        cmap = cm.get_cmap("Reds")

    max_pdf = np.nanmax(pdf_mat)
    if not np.isfinite(max_pdf) or max_pdf <= 0.0:
        warnings.warn("PDF values are zero or invalid; skipping contour.")
        return

    default_contour_kwargs = dict(levels=5, cmap=cmap, vmin=max_pdf * 1e-6, linewidths=0.8, zorder=7)
    contour_kwargs = {} if contour_kwargs is None else dict(contour_kwargs)  # shallow copy
    contour_kwargs.pop("nx", None)
    contour_kwargs.pop("ny", None)
    contour_kwargs = {**default_contour_kwargs, **contour_kwargs}

    try:
        ax.contour(plot_X, plot_Y, pdf_mat, **contour_kwargs)
    except Exception as e:
        warnings.warn(f"Contour plotting failed: {e}")
        return


#============================
# Plotting
#============================
"""
def plot_single_panel_hvsr_curves(
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal",
    plot_valid_curves=True,
    plot_invalid_curves=False,
    plot_mean_curve=True,
    plot_frequency_std=True,
    plot_peak_mean_curve=True,
    plot_peak_individual_valid_curves=True,
    plot_peak_individual_invalid_curves=False,
    ax=None,
    subplots_kwargs=None,
):
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_subplots_kwargs = dict(figsize=(3.75, 2.5), dpi=150)
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, ax = plt.subplots(**subplots_kwargs)

    if plot_valid_curves:
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=True)

    if plot_invalid_curves:
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=False)

    if plot_mean_curve:
        _plot_mean_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc)
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc, n=+1)
        _plot_nth_std_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc, n=-1, plot_kwargs=dict(label=None))

    if plot_frequency_std:
        _plot_nth_std_frequency_range(ax=ax, hvsr=hvsr, distribution=distribution_fn, n=+1)

    if plot_peak_mean_curve:
        _plot_peak_mean_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc)

    if plot_peak_individual_valid_curves:
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=True)

    if plot_peak_individual_invalid_curves:
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=False)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    ax.legend(loc="upper right")

    if ax_was_none:
        return (fig, ax)
    else:
        return ax
"""


def plot_single_panel_hvsr_curves(
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal",
    plot_valid_curves=True,
    plot_invalid_curves=False,
    plot_mean_curve=True,
    plot_frequency_std=True,
    plot_peak_mean_curve=True,
    plot_peak_individual_valid_curves=True,
    plot_peak_individual_invalid_curves=False,
    ax=None,
    subplots_kwargs=None,
):
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_subplots_kwargs = dict(figsize=(3.75, 2.5), dpi=150)
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, ax = plt.subplots(**subplots_kwargs)

    if plot_valid_curves:
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=True)

    if plot_invalid_curves:
        _plot_individual_hvsr_curves(ax=ax, hvsr=hvsr, valid=False)

    if plot_mean_curve:
        _plot_mean_hvsr_curve(ax=ax, hvsr=hvsr, distribution=distribution_mc)
        _plot_nth_std_hvsr_curve(
            ax=ax,
            hvsr=hvsr,
            distribution=distribution_mc,
            n=+1
        )
        _plot_nth_std_hvsr_curve(
            ax=ax,
            hvsr=hvsr,
            distribution=distribution_mc,
            n=-1,
            plot_kwargs=dict(label=None)
        )

    if plot_frequency_std:
        _plot_nth_std_frequency_range(
            ax=ax,
            hvsr=hvsr,
            distribution=distribution_fn,
            n=+1
        )

    if plot_peak_mean_curve:
        _plot_peak_mean_hvsr_curve(
            ax=ax,
            hvsr=hvsr,
            distribution=distribution_mc
        )

    if plot_peak_individual_valid_curves:
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=True)

    if plot_peak_individual_invalid_curves:
        _plot_peak_individual_hvsr_curve(ax=ax, hvsr=hvsr, valid=False)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    ax.legend(loc="upper right")

    if ax_was_none:
        return (fig, ax)
    else:
        return ax


def plot_seismic_recordings_3c(
    srecords,
    valid_window_boolean_mask=None,
    axs=None,
    subplots_kwargs=None,
    normalize=True
):
    axs_was_none = False
    if axs is None:
        axs_was_none = True
        default_subplots_kwargs = dict(nrows=3, figsize=(4, 4.5), dpi=150,
                                       sharex=True, sharey=True,
                                       gridspec_kw=dict(hspace=0.5))
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, axs = plt.subplots(**subplots_kwargs)

    if len(axs) != 3:
        raise ValueError(f"axs is of length {len(axs)}, must be 3.")

    if isinstance(srecords, SeismicRecording3C):
        srecords = [srecords]

    if valid_window_boolean_mask is None:
        valid_window_boolean_mask = [True] * len(srecords)

    if len(valid_window_boolean_mask) != len(srecords):
        raise ValueError("length of valid_window_boolean_mask must match srecords length.")

    if normalize:
        normalization_factor = 0.0
        for component in ["ns", "ew", "vt"]:
            for srecord in srecords:
                amp = _to_host_array(getattr(srecord, component).amplitude)
                c_max = np.max(np.abs(amp))
                if c_max > normalization_factor:
                    normalization_factor = c_max
    else:
        normalization_factor = 1.0

    for ax, component in zip(axs, ["ns", "ew", "vt"]):
        start_time = 0.0
        for srecord, valid in zip(srecords, valid_window_boolean_mask):
            tseries = getattr(srecord, component)
            time = _to_host_array(tseries.time()) + start_time
            amp = _to_host_array(tseries.amplitude) / (normalization_factor if normalization_factor != 0 else 1.0)
            default_kwargs = DEFAULT_KWARGS["individual_valid_hvsr_curve"].copy() if valid else DEFAULT_KWARGS[
                "individual_invalid_hvsr_curve"].copy()
            default_kwargs["label"] = None
            ax.plot(time, amp, **default_kwargs)
            start_time = time[-1]

        ax.set_title(f"{component.upper()} Recording")
        ax.set_ylabel("Normalized\nAmplitude" if normalize else "Amplitude\n(Counts)")
        ax.set_xlim(0, time[-1])

    axs[-1].set_xlabel("Time (s)")

    if normalize:
        for ax in axs:
            ax.set_ylim(-1, 1)

    if axs_was_none:
        return (fig, axs)
    else:
        return axs


"""
def plot_pre_and_post_rejection(
    srecords,
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal"
):
    if not isinstance(hvsr, HvsrTraditional):
        raise NotImplementedError("Can only plot HvsrTraditional results.")

    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    gs = fig.add_gridspec(nrows=6, ncols=6)

    ax0 = fig.add_subplot(gs[0:2, 0:3])
    ax1 = fig.add_subplot(gs[2:4, 0:3])
    ax2 = fig.add_subplot(gs[4:6, 0:3])
    ax3 = fig.add_subplot(gs[0:3, 3:6])
    ax4 = fig.add_subplot(gs[3:6, 3:6])

    plot_seismic_recordings_3c(srecords, valid_window_boolean_mask=hvsr.valid_window_boolean_mask, axs=(ax0, ax1, ax2))

    ax = ax3
    store_valid_window_boolean_mask = _to_host_array(hvsr.valid_window_boolean_mask).copy()
    store_valid_peak_boolean_mask = _to_host_array(hvsr.valid_peak_boolean_mask).copy()
    hvsr.valid_window_boolean_mask = _to_xp_array(np.full_like(store_valid_window_boolean_mask, True, dtype=bool))
    hvsr.valid_peak_boolean_mask = _to_xp_array(np.full_like(store_valid_peak_boolean_mask, True, dtype=bool))
    plot_single_panel_hvsr_curves(hvsr, distribution_mc=distribution_mc, distribution_fn=distribution_fn,
                                  plot_peak_individual_valid_curves=True, plot_peak_mean_curve=True, ax=ax)
    ax.set_title("Before Rejection")
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax = ax4
    hvsr.valid_window_boolean_mask = _to_xp_array(store_valid_window_boolean_mask)
    hvsr.valid_peak_boolean_mask = _to_xp_array(store_valid_peak_boolean_mask)
    plot_single_panel_hvsr_curves(hvsr, distribution_mc=distribution_mc, distribution_fn=distribution_fn,
                                  plot_peak_mean_curve=True, plot_invalid_curves=True,
                                  plot_peak_individual_valid_curves=True, plot_peak_individual_invalid_curves=True,
                                  ax=ax)
    if ax4.get_legend() is not None:
        ax4.get_legend().remove()
    ax.set_title("After Rejection")

    axs = (ax0, ax3, ax1, ax4, ax2)
    for ax, letter in zip(axs, list("abcde")):
        text = ax.text(0.02, 0.97, f"({letter})", ha="left", va="top", transform=ax.transAxes)
        text.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.15))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=1, w_pad=1, rect=(0, 0.05, 1, 1))

    ax4.legend(loc="upper center", bbox_to_anchor=(-0.2, -0.23), ncols=4)

    return (fig, axs)
"""


def plot_pre_and_post_rejection(
    srecords,
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal"
):
    if not isinstance(hvsr, HvsrTraditional):
        raise NotImplementedError("Can only plot HvsrTraditional results.")

    fig = plt.figure(figsize=(7, 5.5), dpi=150)
    gs = fig.add_gridspec(nrows=6, ncols=6)

    ax0 = fig.add_subplot(gs[0:2, 0:3])
    ax1 = fig.add_subplot(gs[2:4, 0:3])
    ax2 = fig.add_subplot(gs[4:6, 0:3])
    ax3 = fig.add_subplot(gs[0:3, 3:6])
    ax4 = fig.add_subplot(gs[3:6, 3:6])

    plot_seismic_recordings_3c(
        srecords,
        valid_window_boolean_mask=hvsr.valid_window_boolean_mask,
        axs=(ax0, ax1, ax2)
    )

    store_valid_window_boolean_mask = _to_host_array(hvsr.valid_window_boolean_mask).copy()
    store_valid_peak_boolean_mask = _to_host_array(hvsr.valid_peak_boolean_mask).copy()

    try:
        # Before rejection
        hvsr.valid_window_boolean_mask = _to_xp_array(
            np.full_like(store_valid_window_boolean_mask, True, dtype=bool)
        )
        hvsr.valid_peak_boolean_mask = _to_xp_array(
            np.full_like(store_valid_peak_boolean_mask, True, dtype=bool)
        )
        plot_single_panel_hvsr_curves(
            hvsr,
            distribution_mc=distribution_mc,
            distribution_fn=distribution_fn,
            plot_peak_individual_valid_curves=True,
            plot_peak_mean_curve=True,
            ax=ax3
        )
        ax3.set_title("Before Rejection")
        if ax3.get_legend() is not None:
            ax3.get_legend().remove()

        # After rejection
        hvsr.valid_window_boolean_mask = _to_xp_array(store_valid_window_boolean_mask)
        hvsr.valid_peak_boolean_mask = _to_xp_array(store_valid_peak_boolean_mask)
        plot_single_panel_hvsr_curves(
            hvsr,
            distribution_mc=distribution_mc,
            distribution_fn=distribution_fn,
            plot_peak_mean_curve=True,
            plot_invalid_curves=True,
            plot_peak_individual_valid_curves=True,
            plot_peak_individual_invalid_curves=True,
            ax=ax4
        )
        ax4.set_title("After Rejection")
        if ax4.get_legend() is not None:
            ax4.get_legend().remove()

    finally:
        # Always restore original state
        hvsr.valid_window_boolean_mask = _to_xp_array(store_valid_window_boolean_mask)
        hvsr.valid_peak_boolean_mask = _to_xp_array(store_valid_peak_boolean_mask)

    axs = (ax0, ax3, ax1, ax4, ax2)
    for ax, letter in zip(axs, list("abcde")):
        text = ax.text(0.02, 0.97, f"({letter})", ha="left", va="top", transform=ax.transAxes)
        text.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.15))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=1, w_pad=1, rect=(0, 0.05, 1, 1))
    ax4.legend(loc="upper center", bbox_to_anchor=(-0.2, -0.23), ncols=4)

    return (fig, axs)


def summarize_hvsr_statistics(
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal"
):
    def safe_inverse(val):
        if val is None or np.isnan(val) or val == 0:
            return np.nan
        return 1 / val

    if isinstance(hvsr, (HvsrTraditional, HvsrAzimuthal)):
        if distribution_fn == "lognormal":
            fn_mean = _to_python_float(hvsr.Mean_fn_Frequency(distribution=distribution_fn))
            fn_std = _to_python_float(hvsr.std_fn_frequency(distribution=distribution_fn))
            fn_neg = _to_python_float(hvsr.nth_std_fn_frequency(-1, distribution=distribution_fn))
            fn_pos = _to_python_float(hvsr.nth_std_fn_frequency(+1, distribution=distribution_fn))

            an_mean = _to_python_float(hvsr.Mean_fn_Amplitude(distribution=distribution_fn))
            an_std = _to_python_float(hvsr.std_fn_amplitude(distribution=distribution_fn))
            an_neg = _to_python_float(hvsr.nth_std_fn_amplitude(-1, distribution=distribution_fn))
            an_pos = _to_python_float(hvsr.nth_std_fn_amplitude(+1, distribution=distribution_fn))

            data = np.array([
                [fn_mean, fn_std, fn_neg, fn_pos],
                [safe_inverse(fn_mean), fn_std, safe_inverse(fn_neg), safe_inverse(fn_pos)],
                [an_mean, an_std, an_neg, an_pos],
            ])

            columns = [
                "Exponentiated Lognormal Median (units)",
                "Lognormal Standard Deviation (log units)",
                "-1 Lognormal Standard Deviation (units)",
                "+1 Lognormal Standard Deviation (units)"
            ]

        elif distribution_fn == "normal":
            fn_mean = _to_python_float(hvsr.Mean_fn_Frequency(distribution=distribution_fn))
            fn_std = _to_python_float(hvsr.std_fn_frequency(distribution=distribution_fn))
            fn_neg = _to_python_float(hvsr.nth_std_fn_frequency(-1, distribution=distribution_fn))
            fn_pos = _to_python_float(hvsr.nth_std_fn_frequency(+1, distribution=distribution_fn))

            an_mean = _to_python_float(hvsr.mean_fn_amplitude(distribution=distribution_fn))
            an_std = _to_python_float(hvsr.std_fn_amplitude(distribution=distribution_fn))
            an_neg = _to_python_float(hvsr.nth_std_fn_amplitude(-1, distribution=distribution_fn))
            an_pos = _to_python_float(hvsr.nth_std_fn_amplitude(+1, distribution=distribution_fn))

            data = np.array([
                [fn_mean, fn_std, fn_neg, fn_pos],
                [np.nan, np.nan, np.nan, np.nan],
                [an_mean, an_std, an_neg, an_pos],
            ])

            columns = [
                "Mean (units)",
                "Standard Deviation (units)",
                "-1 Standard Deviation (units)",
                "+1 Standard Deviation (units)"
            ]
        else:
            raise NotImplementedError

        # Create Data Frame
        df = pd.DataFrame(
            data=data,
            columns=columns,
            index=[
                "Resonant Site Frequency, fn (Hz)",
                "Resonant Site Period, Tn (s)",
                "Resonance Amplitude, An"
            ]
        )

        # Caption from peak
        mc_f, mc_a = hvsr.mean_curve_peak(distribution=distribution_mc)
        caption = f"The peak of the mean curve is at {_to_python_float(mc_f):.3f} Hz with amplitude {_to_python_float(mc_a):.3f}."

        s = df.style.format(precision=3)
        s = s.set_caption(caption).set_table_styles([{
            'selector': 'caption',
            'props': 'caption-side: bottom; font-size:1.25em;'
        }], overwrite=False)

        with pd.option_context('display.max_colwidth', None):
            display(s)

    elif isinstance(hvsr, HvsrDiffuseField):
        mc_f, mc_a = hvsr.mean_curve_peak(distribution=distribution_mc)
        caption = f"The peak of the mean curve is at {_to_python_float(mc_f):.3f} Hz with amplitude {_to_python_float(mc_a):.3f}."
        print(caption)

    else:
        raise NotImplementedError


def _azimuthal_mesh_from_hvsr(
    hvsr,
    distribution_mc="lognormal"
):
    azimuths = [*hvsr.azimuths, 180.0]
    freq_host = _to_host_array(hvsr.frequency)
    mesh_frq, mesh_azi = np.meshgrid(freq_host, azimuths)
    mesh_amp = _to_host_array(hvsr.mean_curve_by_azimuth(distribution=distribution_mc))
    mesh_amp = np.vstack((mesh_amp, mesh_amp[0]))
    return mesh_frq, mesh_azi, mesh_amp


def plot_azimuthal_contour_2d(
    hvsr,
    distribution_mc="lognormal",
    plot_mean_curve_peak_by_azimuth=True,
    fig=None,
    ax=None,
    subplots_kwargs=None,
    contourf_kwargs=None
):
    if not isinstance(hvsr, HvsrAzimuthal):
        raise NotImplementedError("Can only plot HvsrAzimuthal results.")

    mesh_frq, mesh_azi, mesh_amp = _azimuthal_mesh_from_hvsr(hvsr, distribution_mc=distribution_mc)

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_subplots_kwargs = dict(figsize=(3.75, 3), dpi=150)
        if subplots_kwargs is None:
            subplots_kwargs = {}
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}
        fig, ax = plt.subplots(**subplots_kwargs)

    default_contourf_kwargs = dict(cmap=cm.plasma, levels=10)
    contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
    contourf_kwargs = {**default_contourf_kwargs, **contourf_kwargs}
    contour = ax.contourf(mesh_frq, mesh_azi, mesh_amp, **contourf_kwargs)

    ax.set_xscale("log")
    ax.set_xlim(_to_python_float(hvsr.frequency[0]), _to_python_float(hvsr.frequency[-1]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_yticks(np.arange(0, 180 + 30, 30))
    ax.set_ylim(0, 180)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    if np.max(mesh_amp) < 6.5:
        ticks = np.arange(0, 7, 1)
    elif np.max(mesh_amp) < 14:
        ticks = np.arange(0, 16, 2)
    else:
        ticks = np.arange(0, (int(np.max(mesh_amp) // 5) + 1) * 5, 5)
    plt.colorbar(contour, cax=cax, orientation="horizontal", ticks=ticks)
    cax.xaxis.set_ticks_position("top")

    if plot_mean_curve_peak_by_azimuth:
        fpeak, _ = hvsr.mean_curve_peak_by_azimuth(distribution=distribution_mc)
        ax.plot(_to_host_array(fpeak), _to_host_array(hvsr.azimuths),
                **DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal_2d"])
        ax.legend()

    if ax_was_none:
        return (fig, (ax, cax))


def plot_azimuthal_contour_3d(
    hvsr,
    distribution_mc="lognormal",
    ax=None,
    plot_mean_curve_peak_by_azimuth=True,
    camera_elevation=35,
    camera_azimuth=250,
    camera_distance=13
):
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig = plt.figure(figsize=(3.75, 5), dpi=150)
        ax = fig.add_subplot(projection="3d")

    mesh_frq, mesh_azi, mesh_amp = _azimuthal_mesh_from_hvsr(hvsr, distribution_mc=distribution_mc)

    ax.plot_surface(np.log10(mesh_frq), mesh_azi, mesh_amp, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0,
                    antialiased=False)
    for coord in list("xyz"):
        getattr(ax, f"{coord}axis").pane.fill = False
        getattr(ax, f"{coord}axis").pane.set_edgecolor('white')
    ax.set_xticks(np.log10(np.array([0.01, 0.1, 1, 10, 100])))
    ax.set_xticklabels(["$10^{" + str(x) + "}$" for x in range(-2, 3)])
    ax.set_xlim(np.log10((_to_python_float(hvsr.frequency[0]), _to_python_float(hvsr.frequency[-1]))))
    ax.view_init(elev=camera_elevation, azim=camera_azimuth)
    ax.dist = camera_distance
    ax.set_yticks(np.arange(0, 180 + 45, 45))
    ax.set_ylim(0, 180)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_zlabel("HVSR Amplitude")

    if plot_mean_curve_peak_by_azimuth:
        fpeak, apeak = hvsr.mean_curve_peak_by_azimuth(distribution=distribution_mc)
        fpeak = _to_host_array(np.concatenate([fpeak, fpeak[:1]]))
        azimuths = _to_host_array(np.array([*hvsr.azimuths, 180.]))
        apeak = _to_host_array(np.concatenate([apeak, apeak[:1]]))
        ax.scatter(np.log10(fpeak), azimuths, apeak * 1.05, **DEFAULT_KWARGS["peak_mean_hvsr_curve_azimuthal_3d"])
        ax.legend()

    if ax_was_none:
        return (fig, ax)

"""
def plot_azimuthal_summary(
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal",
    plot_mean_curve_peak_by_azimuth=True,
    plot_valid_curves=True,
    plot_invalid_curves=False,
    plot_mean_curve=True,
    plot_frequency_std=True,
    plot_peak_mean_curve=True,
    plot_peak_individual_valid_curves=True,
    plot_peak_individual_invalid_curves=False,
):
    fig = plt.figure(figsize=(6, 5), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=2, wspace=0.3,
                          hspace=0.2, width_ratios=(1.2, 0.8))
    ax0 = fig.add_subplot(gs[0:3, 0:1], projection='3d')
    ax1 = fig.add_subplot(gs[0:2, 1:2])
    ax2 = fig.add_subplot(gs[2:4, 1:2])
    fig.subplots_adjust(bottom=0.21)

    # Plot 3D contour
    plot_azimuthal_contour_3d(hvsr, distribution_mc=distribution_mc, ax=ax0,
                              plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth)

    # Plot 2D contour
    plot_azimuthal_contour_2d(hvsr, distribution_mc=distribution_mc,
                              plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth,
                              ax=ax1)
    ax1.set_xlabel("")
    ax1.set_xticks([])

    plot_single_panel_hvsr_curves(hvsr, distribution_mc=distribution_mc, distribution_fn=distribution_fn,
                                  plot_valid_curves=plot_valid_curves, plot_invalid_curves=plot_invalid_curves,
                                  plot_mean_curve=plot_mean_curve, plot_frequency_std=plot_frequency_std,
                                  plot_peak_mean_curve=plot_peak_mean_curve,
                                  plot_peak_individual_valid_curves=plot_peak_individual_valid_curves,
                                  plot_peak_individual_invalid_curves=plot_peak_individual_invalid_curves, ax=ax2)
    ax2.get_legend().remove()
    ax2.legend(loc="lower left", bbox_to_anchor=(-1.9, -0.1), ncols=2)

    if plot_peak_mean_curve:
        _plot_peak_mean_hvsr_curve(ax=ax2, hvsr=hvsr, distribution=distribution_mc)

    xs, ys = [0.15, 0.64, 0.64], [0.83, 0.83, 0.50]
    for x, y, letter in zip(xs, ys, list("abc")):
        text = fig.text(x, y, f"({letter})")
        text.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.15))

    return (fig, (ax0, ax1, ax2))
"""


def plot_azimuthal_summary(
    hvsr,
    distribution_mc="lognormal",
    distribution_fn="lognormal",
    plot_mean_curve_peak_by_azimuth=True,
    plot_valid_curves=True,
    plot_invalid_curves=False,
    plot_mean_curve=True,
    plot_frequency_std=True,
    plot_peak_mean_curve=True,
    plot_peak_individual_valid_curves=True,
    plot_peak_individual_invalid_curves=False,
):
    fig = plt.figure(figsize=(6, 5), dpi=150)
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        wspace=0.3,
        hspace=0.2,
        width_ratios=(1.2, 0.8)
    )
    ax0 = fig.add_subplot(gs[0:3, 0:1], projection='3d')
    ax1 = fig.add_subplot(gs[0:2, 1:2])
    ax2 = fig.add_subplot(gs[2:4, 1:2])
    fig.subplots_adjust(bottom=0.21)

    # Plot 3D contour
    plot_azimuthal_contour_3d(
        hvsr,
        distribution_mc=distribution_mc,
        ax=ax0,
        plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth
    )

    # Plot 2D contour
    plot_azimuthal_contour_2d(
        hvsr,
        distribution_mc=distribution_mc,
        plot_mean_curve_peak_by_azimuth=plot_mean_curve_peak_by_azimuth,
        ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_xticks([])

    # Plot single-panel HVSR curves
    plot_single_panel_hvsr_curves(
        hvsr,
        distribution_mc=distribution_mc,
        distribution_fn=distribution_fn,
        plot_valid_curves=plot_valid_curves,
        plot_invalid_curves=plot_invalid_curves,
        plot_mean_curve=plot_mean_curve,
        plot_frequency_std=plot_frequency_std,
        plot_peak_mean_curve=plot_peak_mean_curve,
        plot_peak_individual_valid_curves=plot_peak_individual_valid_curves,
        plot_peak_individual_invalid_curves=plot_peak_individual_invalid_curves,
        ax=ax2
    )

    # Plot peak mean curve before legend is created
    if plot_peak_mean_curve:
        _plot_peak_mean_hvsr_curve(
            ax=ax2,
            hvsr=hvsr,
            distribution=distribution_mc
        )

    # Rebuild legend after all artists are plotted
    legend = ax2.get_legend()
    if legend is not None:
        legend.remove()

    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(-1.9, -0.1),
        ncols=2
    )

    xs, ys = [0.15, 0.64, 0.64], [0.83, 0.83, 0.50]
    for x, y, letter in zip(xs, ys, list("abc")):
        text = fig.text(x, y, f"({letter})")
        text.set_bbox(dict(
            facecolor='white',
            edgecolor='none',
            boxstyle='round',
            pad=0.15
        ))

    return (fig, (ax0, ax1, ax2))


def plot_voronoi(
    valid_sensor_coordinates,
    valid_mean_fn,
    tesselation_vertices,
    boundary,
    ax=None,
    fig_kwargs=None
):
    ax_was_none = False
    if ax is None:
        ax_was_none = True
        default_fig_kwargs = dict(figsize=(3.5, 3.5), dpi=150)
        if fig_kwargs is None:
            fig_kwargs = {}
        fig_kwargs = {**default_fig_kwargs, **fig_kwargs}
        fig, ax = plt.subplots(**fig_kwargs)

    valid_mean_fn_host = _to_host_array(valid_mean_fn)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = cm.colors.Normalize(vmin=np.min(valid_mean_fn_host), vmax=np.max(valid_mean_fn_host))
    cmap = cm.autumn
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label="Resonant Frequency (Hz)")

    for _dat, vertices in zip(_to_host_array(valid_mean_fn), tesselation_vertices):
        ax.fill(vertices[:, 0], vertices[:, 1], facecolor=mpl.colors.rgb2hex(cmap(norm(_dat))[:3]), edgecolor="black",
                linewidth=0.5)

    coords = _to_host_array(valid_sensor_coordinates)
    ax.plot(coords[:, 0], coords[:, 1], markerfacecolor="cornflowerblue", marker="o", linestyle="",
            markeredgecolor="black", label="Sensor Location")

    closed_boundary = _to_host_array(boundary)
    closed_boundary = np.vstack((closed_boundary, closed_boundary[0, :]))
    ax.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="black", label="Boundary", linewidth=3)

    ax.set_xlabel("Relative Easting (m)")
    ax.set_ylabel("Relative Northing (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    if ax_was_none:
        return fig, ax


def _to_host(arr):
    try:
        return arr.get()
    except AttributeError:
        return np.asarray(arr)


def summarize_spatial_statistics(
    spatial_mean,
    spatial_stddev,
    spatial_distribution
):
    mean_host = _to_host(spatial_mean)
    std_host = _to_host(spatial_stddev)

    if spatial_distribution == "lognormal":
        data = np.array([
            [
                mean_host,
                std_host,
                np.exp(np.log(mean_host) - std_host),
                np.exp(np.log(mean_host) + std_host)
            ],
            [
                1 / mean_host,
                std_host,
                1 / np.exp(np.log(mean_host) - std_host),
                1 / np.exp(np.log(mean_host) + std_host)
            ],
        ])
        columns = [
            "Exponentiated Lognormal Median (units)",
            "Lognormal Standard Deviation (log units)",
            "-1 Lognormal Standard Deviation (units)",
            "+1 Lognormal Standard Deviation (units)",
        ]
    elif spatial_distribution == "normal":
        data = np.array([
            [mean_host, std_host, mean_host - std_host, mean_host + std_host],
            [np.nan, np.nan, np.nan, np.nan],
        ])
        columns = [
            "Mean (units)",
            "Standard Deviation (units)",
            "-1 Standard Deviation (units)",
            "+1 Standard Deviation (units)",
        ]
    else:
        raise ValueError(f"spatial_distribution={spatial_distribution} not recognized.")

    df = pd.DataFrame(
        data=data,
        columns=columns,
        index=["Resonant Site Frequency, fn (Hz)", "Resonant Site Period, Tn (s)"]
    )
    s = df.style.format(precision=3)
    with pd.option_context('display.max_colwidth', None):
        display(s)



####################################
####################################
# Window Rejection
####################################
####################################
def _to_xp_array(arr, dtype=None):
    if USE_CUPY:
        if isinstance(arr, cp.ndarray):
            return arr.astype(dtype) if dtype is not None else arr
        return cp.asarray(arr, dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype)


def _to_host_array(arr):
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _to_python_float(x):
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        if USE_CUPY and isinstance(x, cp.generic):
            return float(cp.asnumpy(x))
        try:
            return float(np.asarray(x))
        except Exception as e:
            raise TypeError(f"_to_python_float: cannot convert object to type {type(x)} to float") from e


#=============================
# STA/LTA window rejection
#=============================
# sta = Short Term Average
# lta = Long Term Average
def sta_lta_window_rejection(
    records: Iterable,
    sta_seconds: float = 1.0,
    lta_seconds: float = 30.0,
    min_sta_lta_ratio: float = 0.2,
    max_sta_lta_ratio: float = 2.5,
    components: Tuple[str, str, str] = ("ns", "ew", "vt"),
    hvsr: Optional[object] = None,
):

    passing_records: List = []
    valid_window_boolean_mask: List = []

    for rec_idx, record in enumerate(records):
        reject_record = False

        for component in components:
            timeseries = getattr(record, component)
            n_samples = int(timeseries.n_samples)
            dt = float(timeseries.dt_in_seconds)

            npts_in_sta = int(max(1, round(sta_seconds / dt)))
            if npts_in_sta <= 0:
                raise IndexError("sta_seconds must be > 0 and larger than dt.")

            if npts_in_sta > n_samples:
                msg = (
                    "sta_seconds must be shorter than record length; "
                    f"sta_seconds={sta_seconds}, record_length_seconds={timeseries.time()[-1]}"
                )
                raise IndexError(msg)

            n_sta_in_window = int(n_samples // npts_in_sta)
            if n_sta_in_window <= 0:
                reject_record = True
                break

            short_len = npts_in_sta * n_sta_in_window
            try:
                short_timeseries = _to_xp_array(timeseries.amplitude[:short_len], dtype=xp.float64)
            except Exception:
                short_timeseries = xp.asarray(timeseries.amplitude[:short_len], dtype=xp.float64)

            try:
                sta_input = sanitize_signal(short_timeseries, xp_lib=xp)
            except Exception:
                warnings.warn("sanitize_signal unavailable; using nan_to_num fallback for STA computation.", UserWarning)
                sta_input = xp.nan_to_num(short_timeseries, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                abs_short = xp.abs(sta_input)
                reshaped = abs_short.reshape((n_sta_in_window, npts_in_sta))
                sta_values = xp.mean(reshaped, axis=1)
            except Exception:
                sta_values = xp.array([xp.mean(abs_short[i * npts_in_sta:(i + 1) * npts_in_sta]) for i in range(n_sta_in_window)], dtype=xp.float64)

            npts_in_lta = int(max(1, round(lta_seconds / dt)))
            if npts_in_lta <= 0:
                raise IndexError("lta_seconds must be > 0 and larger than dt.")
            if npts_in_lta > n_samples:
                msg = (
                    "lta_seconds must be shorter than record length; "
                    f"lta_seconds={lta_seconds}, record_length_seconds={timeseries.time()[-1]}"
                )
                raise IndexError(msg)

            if short_len >= npts_in_lta:
                lta_segment = sta_input[:npts_in_lta]
            else:
                try:
                    full_amp = _to_xp_array(timeseries.amplitude, dtype=xp.float64)
                except Exception:
                    full_amp = xp.asarray(timeseries.amplitude, dtype=xp.float64)
                try:
                    full_amp_sanit = sanitize_signal(full_amp, xp_lib=xp)
                except Exception:
                    full_amp_sanit = xp.nan_to_num(full_amp, nan=0.0, posinf=0.0, neginf=0.0)
                lta_segment = full_amp_sanit[:npts_in_lta]

            lta = xp.mean(xp.abs(lta_segment))

            try:
                lta_finite = bool(xp.isfinite(lta))
            except Exception:
                lta_finite = np.isfinite(float(lta))

            if (not lta_finite) or (float(lta) <= EPS):
                reject_record = True
                break

            if not bool(xp.all(xp.isfinite(sta_values))):
                reject_record = True
                break

            ratios = sta_values / float(lta)

            try:
                max_ratio = float(xp.max(ratios))
                min_ratio = float(xp.min(ratios))
            except Exception:
                reject_record = True
                break

            if (max_ratio > max_sta_lta_ratio) or (min_ratio < min_sta_lta_ratio) or (not np.isfinite(max_ratio)) or (not np.isfinite(min_ratio)):
                reject_record = True
                break

        # end components loop

        if not reject_record:
            passing_records.append(record)
            valid_window_boolean_mask.append(True)
        else:
            valid_window_boolean_mask.append(False)

    if hvsr is not None:
        mask_xp = _to_xp_array(valid_window_boolean_mask, dtype=bool)
        if isinstance(hvsr, HvsrTraditional):
            hvsr.valid_window_boolean_mask = mask_xp
            hvsr.valid_peak_boolean_mask = mask_xp
        elif isinstance(hvsr, HvsrAzimuthal):
            for _hvsr in hvsr.hvsrs:
                _hvsr.valid_window_boolean_mask = mask_xp
                _hvsr.valid_peak_boolean_mask = mask_xp
        else:
            raise NotImplementedError("hvsr type not supported in sta_lta_window_rejection")

    return passing_records



#===================================
# Maximum-value window rejection
#===================================
def maximum_value_window_rejection(
    records: Iterable,
    maximum_value_threshold: float = 0.9,
    normalized: bool = True,
    components: Tuple[str, str, str] = ("ns", "ew", "vt"),
    hvsr: Optional[object] = None,
):

    records = list(records)
    n_records = len(records)

    maximum_values_xp = xp.empty(n_records, dtype=xp.float64)

    for idx, record in enumerate(records):
        component_max = xp.float64(0.0)
        found_nonfinite = False

        for component in components:
            timeseries = getattr(record, component)
            try:
                amp = _to_xp_array(timeseries.amplitude, dtype=xp.float64)
            except Exception:
                amp = xp.asarray(timeseries.amplitude, dtype=xp.float64)

            if not bool(xp.all(xp.isfinite(amp))):
                component_max = xp.inf
                found_nonfinite = True
                break

            try:
                val = xp.max(xp.abs(amp))
            except Exception:
                val = xp.asarray(np.max(np.abs(_to_host_array(amp))))

            if val > component_max:
                component_max = val

        maximum_values_xp[idx] = component_max

    if normalized:
        finite_mask = xp.isfinite(maximum_values_xp)
        try:
            any_finite = bool(xp.any(finite_mask))
        except Exception:
            any_finite = np.any(_to_host_array(finite_mask))

        if any_finite:
            try:
                finite_vals = maximum_values_xp[finite_mask]
                finite_max = float(xp.max(xp.abs(finite_vals)))
            except Exception:
                finite_max = float(np.max(np.abs(_to_host_array(maximum_values_xp)[np.isfinite(_to_host_array(maximum_values_xp))])))
            if finite_max > 0.0:
                maximum_values_xp = maximum_values_xp / finite_max
            else:
                pass
        else:
            warnings.warn("maximum_value_window_rejection: all maxima are non-finite; skipping normalization.", UserWarning)

    passing_records = []
    valid_window_boolean_mask = []

    maximum_values_host = _to_host_array(maximum_values_xp)

    for rec, maximum_value in zip(records, maximum_values_host):
        if np.isfinite(maximum_value) and (maximum_value < float(maximum_value_threshold)):
            passing_records.append(rec)
            valid_window_boolean_mask.append(True)
        else:
            valid_window_boolean_mask.append(False)

    if hvsr is not None:
        mask_xp = _to_xp_array(valid_window_boolean_mask, dtype=bool)
        if isinstance(hvsr, HvsrTraditional):
            hvsr.valid_window_boolean_mask = mask_xp
            hvsr.valid_peak_boolean_mask = mask_xp
        elif isinstance(hvsr, HvsrAzimuthal):
            for _hvsr in hvsr.hvsrs:
                _hvsr.valid_window_boolean_mask = mask_xp
                _hvsr.valid_peak_boolean_mask = mask_xp
        else:
            raise NotImplementedError("hvsr type not supported in maximum_value_window_rejection")

    return passing_records



#================================================
# Frequency-domain window rejection (wrapper)
#================================================
"""
def frequency_domain_window_rejection(
    hvsr,
    n: float = 2.0,
    max_iterations: int = 50,
    distribution_fn: str = "lognormal",
    distribution_mc: str = "lognormal",
    search_range_in_hz: Tuple[Optional[float], Optional[float]] = (None, None),
    find_peaks_kwargs: Optional[dict] = None,
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else:
        msg = "The frequency domain window rejection algorithm can only be applied to HvsrTraditional and HvsrAzimuthal objects."
        raise NotImplementedError(msg)

    try:
        hvsr.meta["performed window rejection"] = "FDWRA (robust)"
        hvsr.meta["window rejection algorithm arguments"] = dict(
            n=n,
            max_iterations=max_iterations,
            distribution_fn=distribution_fn,
            distribution_mc=distribution_mc,
            search_range_in_hz=search_range_in_hz,
            find_peaks_kwargs=find_peaks_kwargs,
        )
    except Exception:
        pass

    max_performed_iterations = 0
    for h in hvsrs:
        try:
            h.update_peaks_bounded(search_range_in_hz=search_range_in_hz, find_peaks_kwargs=find_peaks_kwargs)
        except Exception as e:
            warnings.warn(f"update_peaks_bounded raised an exception: {e} — proceeding, but peaks may be invalid.", UserWarning)

        iterations = frequency_domain_window_rejection(
            hvsr=h,
            n=n,
            max_iterations=max_iterations,
            distribution_fn=distribution_fn,
            distribution_mc=distribution_mc,
        )

        if iterations > max_performed_iterations:
            max_performed_iterations = iterations

    return max_performed_iterations
"""

"""
def frequency_domain_window_rejection(
    hvsr,
    n=2.0,
    max_iterations=50,
    distribution_fn="lognormal",
    distribution_mc="lognormal"
):
    try:
        amp = _to_host_array(hvsr.amplitude)
    except Exception:
        amp = None
    if amp is None:
        raise RuntimeError("Hvsr object contains no amplitude array accessible on host.")

    amp = np.asarray(amp, dtype=float)

    try:
        existing_mask = _to_host_array(hvsr.valid_window_boolean_mask)
        if existing_mask is None:
            existing_mask = np.ones(amp.shape[0], dtype=bool)
        else:
            existing_mask = np.asarray(existing_mask, dtype=bool)
            if existing_mask.size != amp.shape[0]:
                existing_mask = np.ones(amp.shape[0], dtype=bool)
    except Exception:
        existing_mask = np.ones(amp.shape[0], dtype=bool)

    mask = existing_mask.copy()

    if mask.sum() == 0:
        warnings.warn("frequency_domain_window_rejection: no valid windows to process (existing mask all False).", UserWarning)
        return 0

    iterations_done = 0

    for it in range(max_iterations):
        iterations_done += 1

        try:
            kept_amp = amp[mask, :]
        except Exception:
            warnings.warn("frequency_domain_window_rejection: shape issue selecting kept windows; aborting.", UserWarning)
            break

        if kept_amp.size == 0:
            warnings.warn("frequency_domain_window_rejection: no windows left after previous removals; restoring last mask and stopping.", UserWarning)
            mask = existing_mask.copy()
            iterations_done -= 1
            break

        mean_curve = np.nanmean(kept_amp, axis=0)
        std_curve = np.nanstd(kept_amp, axis=0)

        std_safe = np.where(np.isfinite(std_curve) & (std_curve > EPS), std_curve, EPS)

        normed = (amp - mean_curve[None, :]) / std_safe[None, :]
        sq = np.square(normed)
        with np.errstate(all="ignore"):
            score2 = np.nanmean(sq, axis=1)
        score2 = np.where(np.isfinite(score2), score2, np.inf)
        score = np.sqrt(score2)

        to_remove = (score > n)

        to_remove = to_remove & mask

        if not np.any(to_remove):
            break

        new_mask = mask.copy()
        new_mask[to_remove] = False

        if not np.any(new_mask):
            warnings.warn("frequency_domain_window_rejection: removal would eliminate all windows; aborting further removals.", UserWarning)
            break

        mask = new_mask
        existing_mask = mask.copy()

    # end iterations

    try:
        mask_xp = _to_xp_array(mask, dtype=bool)
        if isinstance(hvsr, HvsrTraditional):
            hvsr.valid_window_boolean_mask = mask_xp
            hvsr.valid_peak_boolean_mask = mask_xp
        elif isinstance(hvsr, HvsrAzimuthal):
            for _hvsr in hvsr.hvsrs:
                _hvsr.valid_window_boolean_mask = mask_xp
                _hvsr.valid_peak_boolean_mask = mask_xp
    except Exception:
        warnings.warn("frequency_domain_window_rejection: failed to write masks back to hvsr object.", UserWarning)

    if not np.any(mask):
        warnings.warn("frequency_domain_window_rejection: ended with zero kept windows; restoring original mask.", UserWarning)
        try:
            mask_xp = _to_xp_array(existing_mask, dtype=bool)
            if isinstance(hvsr, HvsrTraditional):
                hvsr.valid_window_boolean_mask = mask_xp
                hvsr.valid_peak_boolean_mask = mask_xp
            elif isinstance(hvsr, HvsrAzimuthal):
                for _hvsr in hvsr.hvsrs:
                    _hvsr.valid_window_boolean_mask = mask_xp
                    _hvsr.valid_peak_boolean_mask = mask_xp
        except Exception:
            pass

    return iterations_done
"""


def _frequency_domain_window_rejection_core(
    hvsr,
    n=2.0,
    max_iterations=50,
    search_range_in_hz=(None, None),
):
    try:
        amp = _to_host_array(hvsr.amplitude)
    except Exception:
        amp = None
    if amp is None:
        raise RuntimeError("Hvsr object contains no amplitude array accessible on host.")

    amp = np.asarray(amp, dtype=float)

    try:
        existing_mask = _to_host_array(hvsr.valid_window_boolean_mask)
        if existing_mask is None:
            existing_mask = np.ones(amp.shape[0], dtype=bool)
        else:
            existing_mask = np.asarray(existing_mask, dtype=bool)
            if existing_mask.size != amp.shape[0]:
                existing_mask = np.ones(amp.shape[0], dtype=bool)
    except Exception:
        existing_mask = np.ones(amp.shape[0], dtype=bool)

    mask = existing_mask.copy()

    if mask.sum() == 0:
        warnings.warn(
            "frequency_domain_window_rejection: no valid windows to process (existing mask all False).",
            UserWarning
        )
        return 0

    freq = _to_host_array(hvsr.frequency)
    freq = np.asarray(freq, dtype=float)

    fmin, fmax = search_range_in_hz
    if fmin is None:
        fmin = np.nanmin(freq[np.isfinite(freq)])
    if fmax is None:
        fmax = np.nanmax(freq[np.isfinite(freq)])

    band_mask = np.isfinite(freq) & (freq >= float(fmin)) & (freq <= float(fmax))
    if not np.any(band_mask):
        warnings.warn(
            "frequency_domain_window_rejection: empty frequency band after applying search_range_in_hz.",
            UserWarning
        )
        return 0

    iterations_done = 0

    for it in range(max_iterations):
        iterations_done += 1

        kept_amp = amp[mask, :][:, band_mask]
        if kept_amp.size == 0:
            warnings.warn(
                "frequency_domain_window_rejection: no windows left after previous removals; stopping.",
                UserWarning
            )
            iterations_done -= 1
            break

        # Robust center/spread per frequency bin
        center_curve = np.nanmedian(kept_amp, axis=0)
        mad_curve = 1.4826 * np.nanmedian(np.abs(kept_amp - center_curve[None, :]), axis=0)

        robust_sigma = np.where(np.isfinite(mad_curve) & (mad_curve > EPS), mad_curve, EPS)

        normed = (amp[:, band_mask] - center_curve[None, :]) / robust_sigma[None, :]
        sq = np.square(normed)
        with np.errstate(all="ignore"):
            score2 = np.nanmean(sq, axis=1)

        score2 = np.where(np.isfinite(score2), score2, np.inf)
        score = np.sqrt(score2)

        to_remove = (score > n) & mask

        if not np.any(to_remove):
            break

        new_mask = mask.copy()
        new_mask[to_remove] = False

        if not np.any(new_mask):
            warnings.warn(
                "frequency_domain_window_rejection: removal would eliminate all windows; aborting.",
                UserWarning
            )
            break

        mask = new_mask
        existing_mask = mask.copy()

    try:
        mask_xp = _to_xp_array(mask, dtype=bool)
        if isinstance(hvsr, HvsrTraditional):
            hvsr.valid_window_boolean_mask = mask_xp
            hvsr.valid_peak_boolean_mask = mask_xp
        elif isinstance(hvsr, HvsrAzimuthal):
            for _hvsr in hvsr.hvsrs:
                _hvsr.valid_window_boolean_mask = mask_xp
                _hvsr.valid_peak_boolean_mask = mask_xp
    except Exception:
        warnings.warn(
            "frequency_domain_window_rejection: failed to write masks back to hvsr object.",
            UserWarning
        )

    if not np.any(mask):
        warnings.warn(
            "frequency_domain_window_rejection: ended with zero kept windows; restoring original mask.",
            UserWarning
        )
        try:
            mask_xp = _to_xp_array(existing_mask, dtype=bool)
            if isinstance(hvsr, HvsrTraditional):
                hvsr.valid_window_boolean_mask = mask_xp
                hvsr.valid_peak_boolean_mask = mask_xp
            elif isinstance(hvsr, HvsrAzimuthal):
                for _hvsr in hvsr.hvsrs:
                    _hvsr.valid_window_boolean_mask = mask_xp
                    _hvsr.valid_peak_boolean_mask = mask_xp
        except Exception:
            pass

    return iterations_done


def frequency_domain_window_rejection(
    hvsr,
    n=2.0,
    max_iterations=50,
    distribution_fn="lognormal",
    distribution_mc="lognormal",
    search_range_in_hz=(None, None),
    find_peaks_kwargs=None,
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else:
        raise NotImplementedError(
            "The frequency domain window rejection algorithm can only be applied to HvsrTraditional and HvsrAzimuthal objects."
        )

    try:
        hvsr.meta["performed window rejection"] = "FDWRA (robust)"
        hvsr.meta["window rejection algorithm arguments"] = dict(
            n=n,
            max_iterations=max_iterations,
            distribution_fn=distribution_fn,
            distribution_mc=distribution_mc,
            search_range_in_hz=search_range_in_hz,
            find_peaks_kwargs=find_peaks_kwargs,
        )
    except Exception:
        pass

    max_performed_iterations = 0
    for h in hvsrs:
        try:
            h.Update_Peaks_Bounded(
                search_range_in_hz=search_range_in_hz,
                find_peaks_kwargs=find_peaks_kwargs
            )
        except Exception as e:
            warnings.warn(
                f"Update_Peaks_Bounded raised an exception: {e} — proceeding, but peaks may be invalid.",
                UserWarning
            )

        iterations = _frequency_domain_window_rejection_core(
            hvsr=h,
            n=n,
            max_iterations=max_iterations,
            search_range_in_hz=search_range_in_hz,
        )

        if iterations > max_performed_iterations:
            max_performed_iterations = iterations

    return max_performed_iterations



#================================
# Manual window rejection
#================================
def manual_window_rejection(
    hvsr,
    distribution_mc: str = "lognormal",
    distribution_fn: str = "lognormal",
    plot_mean_curve: bool = True,
    plot_frequency_std: bool = True,
    search_range_in_hz: Tuple[Optional[float], Optional[float]] = (None, None),
    find_peaks_kwargs: Optional[dict] = None,
    y_limit: Optional[float] = None,
    fig=None,
    ax=None,
):
    if isinstance(hvsr, HvsrTraditional):
        hvsrs = [hvsr]
    elif isinstance(hvsr, HvsrAzimuthal):
        hvsrs = hvsr.hvsrs
    else:
        raise NotImplementedError("manual_window_rejection only supports HvsrTraditional or HvsrAzimuthal")

    try:
        hvsr.meta["window rejection algorithm"] = "Manual after SESAME (2004)"
        hvsr.meta["window rejection algorithm arguments"] = dict(
            y_limit=y_limit,
            distribution_fn=distribution_fn,
            distribution_mc=distribution_mc,
            search_range_in_hz=search_range_in_hz,
            find_peaks_kwargs=find_peaks_kwargs,
        )
    except Exception:
        pass

    if find_peaks_kwargs is None:
        find_peaks_kwargs = {}
    try:
        hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz, find_peaks_kwargs=find_peaks_kwargs)
    except Exception as e:
        warnings.warn(f"update_peaks_bounded raised: {e} — continuing.", UserWarning)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax = plot_single_panel_hvsr_curves(
        hvsr=hvsr,
        distribution_mc=distribution_mc,
        distribution_fn=distribution_fn,
        plot_mean_curve=plot_mean_curve,
        plot_frequency_std=plot_frequency_std,
        ax=ax,
    )
    if y_limit is not None:
        ax.set_ylim((0, y_limit))

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.autoscale(enable=False)
    plot_continue_button(ax, upper_right_corner_relative=(0.11, 0.98), box_size_relative=(0.1, 0.08))
    fig.show()

    freq_host = _to_host_array(hvsr.frequency)
    if freq_host is None:
        raise RuntimeError("Could not fetch frequency axis from hvsr object.")
    freq_host = np.asarray(freq_host)

    while True:
        res = ginput_session(fig, ax, initial_adjustment=False, n_points=2,
                             ask_to_confirm_point=False, ask_to_continue=False)
        if not res or len(res) < 2:
            plt.close("all")
            break

        xs, ys = res
        try:
            x1, x2 = float(xs[0]), float(xs[1])
            y1, y2 = float(ys[0]), float(ys[1])
        except Exception:
            x1, x2 = float(np.min(xs)), float(np.max(xs))
            y1, y2 = float(np.min(ys)), float(np.max(ys))

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        selected_columns = np.logical_and(freq_host > x_min, freq_host < x_max)
        if not np.any(selected_columns):
            in_continue_box = False
            for _x, _y in zip([x1, x2], [y1, y2]):
                if is_absolute_point_in_relative_box(
                    ax=ax,
                    absolute_point=(_x, _y),
                    upper_right_corner_relative=(0.11, 0.98),
                    box_size_relative=(0.1, 0.08),
                ):
                    in_continue_box = True
                    break
            if in_continue_box:
                plt.close("all")
                break
            else:
                warnings.warn("Selected frequency range contains no frequencies. Try again.", UserWarning)
                continue

        was_any_marked = False

        for h in hvsrs:
            amp_host = _to_host_array(h.amplitude)
            if amp_host is None:
                warnings.warn("Could not read amplitude array for an hvsr instance; skipping.", UserWarning)
                continue
            amp_host = np.asarray(amp_host)

            if amp_host.ndim != 2:
                warnings.warn("Amplitude array has unexpected shape; skipping.", UserWarning)
                continue

            n_windows = amp_host.shape[0]

            try:
                mask_host = np.asarray(_to_host_array(h.valid_window_boolean_mask), dtype=bool)
                if mask_host.size != n_windows:
                    mask_host = np.ones(n_windows, dtype=bool)
            except Exception:
                mask_host = np.ones(n_windows, dtype=bool)

            try:
                peak_mask_host = np.asarray(_to_host_array(h.valid_peak_boolean_mask), dtype=bool)
                if peak_mask_host.size != n_windows:
                    peak_mask_host = mask_host.copy()
            except Exception:
                peak_mask_host = mask_host.copy()

            for idx in range(n_windows):
                if not mask_host[idx]:
                    continue

                row = amp_host[idx, selected_columns]
                if row.size == 0:
                    continue
                finite_mask = np.isfinite(row)
                if not np.any(finite_mask):
                    continue

                cond_mask = np.logical_and(row > y_min, row < y_max)
                cond_mask = np.logical_and(cond_mask, finite_mask)
                if np.any(cond_mask):
                    mask_host[idx] = False
                    peak_mask_host[idx] = False
                    was_any_marked = True

            try:
                h.valid_window_boolean_mask = _to_xp_array(mask_host, dtype=bool)
                h.valid_peak_boolean_mask = _to_xp_array(peak_mask_host, dtype=bool)
            except Exception:
                try:
                    h.valid_window_boolean_mask = mask_host.tolist()
                    h.valid_peak_boolean_mask = peak_mask_host.tolist()
                except Exception:
                    warnings.warn("Failed to write masks back to hvsr object for one instance.", UserWarning)

        try:
            hvsr.update_peaks_bounded(search_range_in_hz=search_range_in_hz, find_peaks_kwargs=find_peaks_kwargs)
        except Exception as e:
            warnings.warn(f"update_peaks_bounded after manual rejection failed: {e}", UserWarning)

        ax.clear()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.autoscale(enable=False)
        ax = plot_single_panel_hvsr_curves(
            hvsr=hvsr,
            distribution_mc=distribution_mc,
            distribution_fn=distribution_fn,
            plot_mean_curve=plot_mean_curve,
            plot_frequency_std=plot_frequency_std,
            ax=ax,
        )
        plot_continue_button(ax, upper_right_corner_relative=(0.11, 0.98), box_size_relative=(0.1, 0.08))
        fig.canvas.draw_idle()

        if not was_any_marked:
            in_continue_box = False
            for _x, _y in zip([x1, x2], [y1, y2]):
                if is_absolute_point_in_relative_box(
                    ax=ax,
                    absolute_point=(_x, _y),
                    upper_right_corner_relative=(0.11, 0.98),
                    box_size_relative=(0.1, 0.08),
                ):
                    in_continue_box = True
                    break
            if in_continue_box:
                plt.close("all")
                break
    return


















