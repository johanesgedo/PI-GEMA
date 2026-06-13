# PINNs Execution
# Copyright (C) 2026 Johanes Gedo Sea
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import cupy as cp
import os
import glob
import re
import matplotlib.pyplot as plt
from pigema.pinns.PINN_NFC_CUDA import PINN_Empirical_Instrument_Response, Calibration_Plotter, PINN_Instrument_Model
from pigema.pinns.PINN_NFC_CUDA import Ambient_Noise_Statistical_Model
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "training"
OUTPUT_TRAINING = (
    PROJECT_ROOT /
    "models" /
    "transfer_functions_streaming.npz"
)
OUTPUT_MODEL = (
    PROJECT_ROOT /
    "models" /
    "Training_Model"
)


OVERLAP = 0.5
NPERSEG = 2048
FS = 100
NFFT = 4096
MAXIMUM_FILES = 2000
LEARNING_RATE = 1e-3
LAMBDA_SMOOTH = 1e-4
LAMBDA_PHYSICS = 1e-4
FMIN = 0.1
FMAX = 20.0
EPOCHS = 2000


def to_numpy(x):
    if x is None:
        return None
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def numeric_sorted(glob_list):
    def keyfn(p):
        name = os.path.basename(p)
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else name
    return sorted(glob_list, key=keyfn)


def load_3col_timeseries_from_mixed_file(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, L in enumerate(f, start=1):
            s = L.strip()
            if not s:
                continue
            toks = s.split()
            nums = []
            for t in toks:
                try:
                    tt = t.replace(',', '')
                    val = float(tt)
                    nums.append(val)
                except Exception:
                    continue
            if len(nums) >= 3:
                z, n, e = nums[-3], nums[-2], nums[-1]
                rows.append([z, n, e])
            else:
                continue
    if len(rows) == 0:
        raise RuntimeError(f"No valid 3-column rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def chunk_3c_data(data3c, chunk_len, max_chunks=None):
    n = data3c.shape[0]
    nchunks = max(1, n // chunk_len)
    if max_chunks is not None:
        nchunks = min(nchunks, max_chunks)
    list3c = []
    for i in range(nchunks):
        s = i * chunk_len
        e = s + chunk_len
        seg = data3c[s:e, :]
        if seg.shape[0] < max(1, chunk_len // 4):
            continue
        N = seg[:,0].copy()
        E = seg[:,1].copy()
        Z = seg[:,2].copy()
        list3c.append((N, E, Z))
    if len(list3c) == 0:
        N = data3c[:,0].copy()
        E = data3c[:,1].copy()
        Z = data3c[:,2].copy()
        list3c = [(N, E, Z)]
    return list3c


def build_reference_streaming(
    data_dir,
    fs=FS,
    nfft=NFFT,
    fmin=FMIN,
    fmax=FMAX,
    default_nperseg=NPERSEG,
    overlap=OVERLAP,
    max_files=MAXIMUM_FILES,
    max_chunks_per_file=6,
    chunk_len=None,
    device=0,
    verbose=True
):
    ambient = None
    ambient = Ambient_Noise_Statistical_Model(fs=fs, nfft=nfft, fmin=fmin, fmax=fmax, nperseg=default_nperseg, overlap=overlap, device=device, verbose=verbose)
    freqs_band = ambient.Build_Frequency_Axis()
    band_len = freqs_band.size
    if verbose:
        print("Band length:", band_len)

    count = 0
    mean = np.zeros(band_len, dtype=np.float64)
    M2 = np.zeros(band_len, dtype=np.float64)

    files = numeric_sorted(glob.glob(os.path.join(data_dir, "*.txt")))[:max_files]

    for p in files:
        try:
            data3c = load_3col_timeseries_from_mixed_file(p)
        except Exception as e:
            print("Skip", p, ":", e); continue

        npts = data3c.shape[0]
        if chunk_len is None:
            nperseg = min(default_nperseg, max(256, npts // 8))
            if nperseg < 128:
                nperseg = min(128, npts)
            chunk_len_local = max(nperseg * 2, 1024)
        else:
            chunk_len_local = chunk_len

        recs = chunk_3c_data(data3c, chunk_len=chunk_len_local, max_chunks=max_chunks_per_file)
        if verbose:
            print(f"{os.path.basename(p)} -> {len(recs)} segments")

        for (N,E,Z) in recs:
            try:
                PSD_band_cp = ambient.Compute_PSD(N, E, Z)  # cupy array
            except Exception as e:
                print("Compute_PSD failed for a segment:", e)
                continue
            PSD_band = cp.asnumpy(PSD_band_cp).astype(np.float64)

            count += 1
            delta = PSD_band - mean
            mean += delta / count
            delta2 = PSD_band - mean
            M2 += delta * delta2

    if count < 1:
        raise RuntimeError("No PSD segments processed.")

    variance = M2 / (count - 1) if count > 1 else np.zeros_like(mean)
    std = np.sqrt(variance)

    S_ref_mean = cp.asarray(mean.astype(np.float32))
    S_ref_std = cp.asarray(std.astype(np.float32))
    if verbose:
        print("Built reference from", count, "segments")
    return ambient.freqs, S_ref_mean, S_ref_std, ambient


def run_batch_streaming_pipeline(
    data_dir,
    **kwargs
):
    fs = kwargs.get("fs", FS)
    nfft = kwargs.get("nfft", NFFT)
    fmin = kwargs.get("fmin", FMIN)
    fmax = kwargs.get("fmax", FMAX)
    device = kwargs.get("device", 0)
    verbose = kwargs.get("verbose", True)
    out_npz = kwargs.get("out_npz", "transfer_functions_streaming.npz")
    out_fig = kwargs.get("out_fig", "batch_streaming_dashboard.png")

    cp.cuda.Device(device).use()

    freqs_band, S_ref_mean, S_ref_std, ambient = build_reference_streaming(
        data_dir,
        fs=fs, nfft=nfft, fmin=fmin, fmax=fmax,
        default_nperseg=kwargs.get("nperseg", NPERSEG),
        overlap=kwargs.get("overlap", OVERLAP),
        max_files=kwargs.get("max_files", MAXIMUM_FILES),
        max_chunks_per_file=kwargs.get("max_chunks_per_file", 6),
        device=device, verbose=verbose
    )

    pinn = PINN_Instrument_Model(
        freqs=freqs_band,
        S_reference=S_ref_mean,
        lr=kwargs.get("lr", LEARNING_RATE),
        lambda_smooth=kwargs.get("lambda_smooth", LAMBDA_SMOOTH),
        lambda_physics=kwargs.get("lambda_physics", LAMBDA_PHYSICS),
        device=device,
        verbose=verbose
    )
    # pinn.Build_Network()
    # pinn.Train(epochs=kwargs.get("epochs", 500))

    pinn.Train_Model(epochs=kwargs.get("epochs", EPOCHS))

    files = numeric_sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if len(files) == 0:
        raise RuntimeError(f"No input files found in {data_dir}")

    rep_file = files[0]
    if verbose:
        print(f"[run_batch_streaming_pipeline] using representative file: {os.path.basename(rep_file)}")

    data3c = load_3col_timeseries_from_mixed_file(rep_file)
    N = data3c[:, 0]
    E = data3c[:, 1]
    Z = data3c[:, 2]

    pipeline = PINN_Empirical_Instrument_Response(
        fs=fs, nfft=nfft, fmin=fmin, fmax=fmax, device=device, verbose=verbose
    )
    pipeline.freqs = freqs_band
    pipeline.S_ref_mean = S_ref_mean
    pipeline.S_ref_std = S_ref_std
    pipeline.pinn_model = pinn

    calib = pipeline.Run_Calibration_Pipeline(N=N, E=E, Z=Z,
                                              alpha=kwargs.get("alpha", 0.5),
                                              regularization=kwargs.get("regularization", 1e-6))

    # H_pin_cp = pinn.Predict_Transfer_Function(return_numpy=False)

    H_pin = pinn.Predict_Transfer(return_numpy=True)
    H_pin_cp = cp.asarray(H_pin)
    H_pin_mag_cp = cp.abs(H_pin_cp).astype(cp.float32)

    H_emp_cp = calib["H_empirical"]
    H_reg_cp = calib["H_regularized"]

    plotter = Calibration_Plotter(
        freqs=cp.asnumpy(freqs_band),
        S_ref_mean=cp.asnumpy(S_ref_mean),
        S_ref_std=cp.asnumpy(S_ref_std),
        pinn_H=cp.asnumpy(H_pin_cp) if isinstance(H_pin_cp, cp.ndarray) else H_pin_cp,
        H_empirical=cp.asnumpy(H_emp_cp),
        H_regularized=cp.asnumpy(H_reg_cp),
        loss_history=pinn.loss_history,
        verbose=verbose
    )

    fig = plotter.Dashboard(figsize=(12, 8))
    plotter.save_figure(fig, out_fig)
    if verbose:
        print(f"[run_batch_streaming_pipeline] saved dashboard to {out_fig}")

    # Save results
    cp.savez(out_npz,
             freqs=cp.asnumpy(freqs_band),
             pinn_H=cp.asnumpy(H_pin_cp),  # complex -> numpy complex
             H_empirical=cp.asnumpy(H_emp_cp),
             H_regularized=cp.asnumpy(H_reg_cp),
             S_ref_mean=cp.asnumpy(S_ref_mean),
             S_ref_std=cp.asnumpy(S_ref_std))
    if verbose:
        print(f"[run_batch_streaming_pipeline] Train complete. Saved {out_npz}")

    f0, zeta = pinn.Physical_Parameters()

    return {
        "model": pinn,
        "f0": float(f0.item()),
        "zeta": float(zeta.item()),
        "saved_files": [out_npz, out_fig]
    }


def main():
    # data_path = "/home/johanesgedo/Project/Training_microtremor/PINN_NFC/47.txt"
    # data_path = "/home/johanesgedo/Project/Training_microtremor/CuHVSR3/Raw_Data_Microtremor"
    # data_path = DATA_PATH

    res = run_batch_streaming_pipeline(
        data_dir=DATA_PATH,
        fs=FS,
        nfft=NFFT,
        fmin=FMIN,
        fmax=FMAX,
        nperseg=NPERSEG,
        overlap=OVERLAP,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_physics=LAMBDA_PHYSICS,
        device=0,
        verbose=True,
        max_files=MAXIMUM_FILES,
        max_chunks_per_file=6,
        out_npz=OUTPUT_TRAINING,
        out_fig=OUTPUT_MODEL
    )

    print("Done. Outputs:", res["saved_files"])


if __name__ == "__main__":
    main()












