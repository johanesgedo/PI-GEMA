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
"""

import numpy as np
import cupy as cp
import os
import glob
import re
import matplotlib.pyplot as plt
from PINN_NFC_CUDA import PINN_Empirical_Instrument_Response, Calibration_Plotter, PINN_Instrument_Model

OVERLAP = 0.5
NPERSEG = 2048
FS = 100
NFFT = 4096
MAXIMUM_FILES = 180
LEARNING_RATE = 1e-3
LAMBDA_SMOOTH = 1e-4
LAMBDA_PHYSICS = 1e-4
FMIN = 0.1
FMAX = 20.0
EPOCHS = 2000

# Training All Data
OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/transfer_functions_streaming.npz"
OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Training_Model"
DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs"

# Training Geothermal Data
# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/RAW_DATA/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/RAW_DATA/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs"

# Training Demak Data
# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor"

# Training OCG Data
# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR6/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG"



# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs"

# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_Microtremor"

# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/Raw_Data_OCG"

# OUTPUT_TRAINING = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/RAW_DATA/transfer_functions_streaming.npz"
# OUTPUT_MODEL = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/RAW_DATA/Training_Model"
# DATA_PATH = "/home/johanesgedo/Project/Training_microtremor/CuHVSR5/Dataset/Hasil_Pengolahan/Gabungan_Data_Untuk_Latih_PINNs/RAW_DATA"


# Helper
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
        Z = seg[:,0].copy()
        N = seg[:,1].copy()
        E = seg[:,2].copy()
        list3c.append((Z, N, E))
    if len(list3c) == 0:
        Z = data3c[:,0].copy()
        N = data3c[:,1].copy()
        E = data3c[:,2].copy()
        list3c = [(Z, N, E)]
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
    ambient = __import__("PINN_NFC_CUDA", fromlist=["Ambient_Noise_Statistical_Model"]).Ambient_Noise_Statistical_Model(
        fs=fs, nfft=nfft, fmin=fmin, fmax=fmax, nperseg=default_nperseg, overlap=overlap, device=device, verbose=verbose
    )
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

        for (Z,N,E) in recs:
            try:
                PSD_band_cp = ambient.Compute_PSD(Z, N, E)  # cupy array
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

    #-----------------------------------------
    # Representative calibration + plotting
    #-----------------------------------------
    files = numeric_sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if len(files) == 0:
        raise RuntimeError(f"No input files found in {data_dir}")

    rep_file = files[0]
    if verbose:
        print(f"[run_batch_streaming_pipeline] using representative file: {os.path.basename(rep_file)}")

    data3c = load_3col_timeseries_from_mixed_file(rep_file)
    Z = data3c[:, 0]
    N = data3c[:, 1]
    E = data3c[:, 2]

    pipeline = PINN_Empirical_Instrument_Response(
        fs=fs, nfft=nfft, fmin=fmin, fmax=fmax, device=device, verbose=verbose
    )
    pipeline.freqs = freqs_band
    pipeline.S_ref_mean = S_ref_mean
    pipeline.S_ref_std = S_ref_std
    pipeline.pinn_model = pinn

    calib = pipeline.Run_Calibration_Pipeline(Z=Z, N=N, E=E,
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

    #-------------------
    # Save results
    #-------------------
    cp.savez(out_npz,
             freqs=cp.asnumpy(freqs_band),
             pinn_H=cp.asnumpy(H_pin_cp),  # complex -> numpy complex
             H_empirical=cp.asnumpy(H_emp_cp),
             H_regularized=cp.asnumpy(H_reg_cp),
             S_ref_mean=cp.asnumpy(S_ref_mean),
             S_ref_std=cp.asnumpy(S_ref_std))
    if verbose:
        print(f"[run_batch_streaming_pipeline] Train complete. Saved {out_npz}")

    f0, zeta = pinn.Physical_Pamarameters()

    return {
        "model": pinn,
        "f0": float(f0.item()),
        "zeta": float(zeta.item()),
        "saved_files": [out_npz, out_fig]
    }


if __name__ == "__main__":
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













