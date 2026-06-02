# Transfer Training of PINNs Results to create Calibrated Microtremor Data
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


from typing import Optional, List, Tuple, Any
import numpy as np
import os
import glob
import concurrent.futures
import functools
import traceback
import argparse
import shutil
import tempfile
from pathlib import Path

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_FUNCTION = (
    PROJECT_ROOT /
    "models" /
    "transfer_functions_streaming.npz"
)

INPUT_DATA = (
    PROJECT_ROOT /
    "data" /
    "microtremor"
)

OUTPUT_DATA = (
    PROJECT_ROOT /
    "data" /
    "calibrated"
)


DEFAULT_FS = 100.0


class Microtremor_Calibrator:

    def __init__(
        self,
        tf_path: str,
        use_gpu: bool = False,
        mode: str = "multiply",
        detrend: bool = True,
        window: bool = True
    ):
        if mode not in ("multiply", "divide"):
            raise ValueError("mode must be 'multiply' or 'divide'")
        self.tf_path = tf_path
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.mode = mode
        self.detrend = detrend
        self.window = window
        self.xp = cp if self.use_gpu else np
        self.Load_Transfer_Function(tf_path)

    def Load_Transfer_Function(
        self,
        npz_path: str
    ) -> None:
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        freq_keys = ['freq', 'f', 'freqs', 'frequencies', 'w']
        H_keys = ['H', 'TF', 'transfer', 'transfer_function', 'transfer_functions', 'transfer_functions_streaming']
        freqs = None
        H = None
        for k in freq_keys:
            if k in data:
                freqs = np.asarray(data[k])
                break
        for k in H_keys:
            if k in data:
                H = np.asarray(data[k])
                break
        if freqs is None or H is None:
            for k in keys:
                arr = np.asarray(data[k])
                if freqs is None and arr.ndim == 1 and np.isrealobj(arr) and arr.size > 1:
                    freqs = arr
                elif H is None and (np.iscomplexobj(arr) or arr.ndim == 1):
                    H = arr
                if freqs is not None and H is not None:
                    break
        if freqs is None or H is None:
            raise ValueError(f"Cannot find freq/H arrays in {npz_path}. Available keys: {keys}")
        if H.ndim == 2 and H.shape[1] == 2 and not np.iscomplexobj(H):
            mag = H[:, 0]; phase = H[:, 1]
            H = mag * np.exp(1j * phase)
        self.freqs_tf = freqs.astype(np.float64, copy=False)
        self.H_tf = H.astype(np.complex128, copy=False)

    def Load_File(
        self,
        path
    ):
        arr = np.loadtxt(path)

        if arr.ndim == 1:
            return None, arr.astype(np.float64), None

        elif arr.ndim == 2:
            if arr.shape[1] == 3:
                t = None
                x = arr.astype(np.float64)  # (n,3)
                fs = None

            elif arr.shape[1] == 4:
                t = arr[:, 0].astype(np.float64)
                x = arr[:, 1:].astype(np.float64)

                dt = np.median(np.diff(t))
                fs = 1.0 / dt if dt > 0 else None

            elif arr.shape[1] == 2:
                t = arr[:, 0].astype(np.float64)
                x = arr[:, 1].astype(np.float64)
                dt = np.median(np.diff(t))
                fs = 1.0 / dt if dt > 0 else None

            else:
                raise ValueError("Format tidak dikenali")

            return t, x, fs

    def Detrend_Linear(
        self,
        x: Any
    ) -> Tuple[Any, Any, Any]:
        xp = self.xp
        n = x.shape[0]
        t = xp.arange(n, dtype=x.dtype)
        t_mean = xp.mean(t)
        if x.ndim == 1:
            x_mean = xp.mean(x)
            num = xp.sum((t - t_mean) * (x - x_mean))
            den = xp.sum((t - t_mean) ** 2)
            slope = num / den if den != 0 else xp.array(0.0, dtype=x.dtype)
            intercept = x_mean - slope * t_mean
            trend = slope * t + intercept
            return x - trend, slope, intercept
        else:
            x_mean = xp.mean(x, axis=0)
            t_minus = (t - t_mean)[:, None]
            num = xp.sum(t_minus * (x - x_mean[None, :]), axis=0)
            den = xp.sum(t_minus ** 2)
            slope = num / den if den != 0 else xp.zeros_like(x_mean)
            intercept = x_mean - slope * t_mean
            trend = (slope[None, :] * t[:, None]) + intercept[None, :]
            return x - trend, slope, intercept

    def Interpolate_Transfer_Function_to_FFT_Bins(
        self,
        freqs_dst: np.ndarray
    ) -> Any:
        freqs_src = np.asarray(self.freqs_tf)
        H_src = np.asarray(self.H_tf)
        if self.use_gpu and isinstance(freqs_dst, cp.ndarray):
            freqs_dst_cpu = np.asnumpy(freqs_dst)
        else:
            freqs_dst_cpu = np.asarray(freqs_dst)
        real_interp = np.interp(freqs_dst_cpu, freqs_src, np.real(H_src), left=np.real(H_src[0]), right=np.real(H_src[-1]))
        imag_interp = np.interp(freqs_dst_cpu, freqs_src, np.imag(H_src), left=np.imag(H_src[0]), right=np.imag(H_src[-1]))
        H_interp_cpu = real_interp + 1j * imag_interp
        return self.xp.asarray(H_interp_cpu) if self.use_gpu else H_interp_cpu

    def Calibrate_Array(
        self,
        samples: Any,
        fs: float
    ) -> Any:
        xp = self.xp
        if self.use_gpu and not isinstance(samples, cp.ndarray):
            samples = cp.asarray(samples)
        elif not self.use_gpu and isinstance(samples, cp.ndarray):
            samples = np.asnumpy(samples)
        n = samples.shape[0]
        single_channel = (samples.ndim == 1)
        if self.detrend:
            x_nodtrend, slope, intercept = self.Detrend_Linear(samples)
        else:
            x_nodtrend = samples
            slope = intercept = None
        if self.window:
            w = xp.hanning(n)
            if single_channel:
                xw = x_nodtrend * w
            else:
                xw = x_nodtrend * w[:, None]
        else:
            xw = x_nodtrend
        if self.use_gpu:
            X = cp.fft.rfft(xw, axis=0)
            freqs = cp.fft.rfftfreq(n, 1.0 / fs)
        else:
            X = np.fft.rfft(xw, axis=0)
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
        H_interp = self.Interpolate_Transfer_Function_to_FFT_Bins(freqs)
        if (not self.use_gpu and H_interp.ndim == 1) or (self.use_gpu and isinstance(H_interp, cp.ndarray) and H_interp.ndim == 1):
            H_b = H_interp[:, None]
        else:
            H_b = H_interp
        if self.mode == "multiply":
            Y = X * H_b
        else:
            eps = 1e-12
            Y = X / (H_b + eps)
        if self.use_gpu:
            y_cal = cp.fft.irfft(Y, n=n, axis=0)
        else:
            y_cal = np.fft.irfft(Y, n=n, axis=0)
        if self.detrend:
            t_idx = xp.arange(n, dtype=y_cal.dtype)
            if single_channel:
                trend = slope * t_idx + intercept
                y_cal = y_cal + trend
            else:
                trend = (t_idx[:, None] * slope[None, :]) + intercept[None, :]
                y_cal = y_cal + trend
        return y_cal

    def Calibrate_File(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        fs: Optional[float] = None
    ) -> str:
        t, x, fs_inferred = self.Load_File(input_path)
        if fs is None:
            if fs_inferred is None:
                raise ValueError("Sampling rate not found; provide fs argument")
            fs = fs_inferred
        y_cal = self.Calibrate_Array(self.xp.asarray(x) if self.use_gpu else x, fs)
        y_cpu = cp.asnumpy(y_cal) if self.use_gpu else y_cal
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = base + "_calibrated.txt"
        if t is not None:
            if y_cpu.ndim == 1:
                out = np.column_stack((t, y_cpu))
                header = "time(s) amplitude"
            else:
                nchan = y_cpu.shape[1]
                ch_labels = [f"ch{i+1}" for i in range(nchan)]
                out = np.column_stack((t, y_cpu))
                header = "time(s) " + " ".join(ch_labels)
        else:
            if y_cpu.ndim == 1:
                out = y_cpu
                header = "amplitude"
            else:
                nchan = y_cpu.shape[1]
                ch_labels = [f"ch{i+1}" for i in range(nchan)]
                out = y_cpu
                header = " ".join(ch_labels)
        np.savetxt(output_path, out, fmt="%.6e", header=header)
        meta_path = os.path.splitext(output_path)[0] + "_meta.npz"
        meta = dict(fs=fs, tf_path=self.tf_path, mode=self.mode, detrend=self.detrend, window=self.window)
        if y_cpu.ndim == 2:
            meta['n_channels'] = int(y_cpu.shape[1])
            meta['channel_labels'] = ch_labels
        else:
            meta['n_channels'] = 1
            meta['channel_labels'] = ['ch1']
        np.savez(meta_path, **meta)
        return output_path


class Batch_Microtremor_Calibrator:

    def __init__(
        self,
        tf_path: str,
        use_gpu: bool = False,
        mode: str = "multiply",
        detrend: bool = True,
        window: bool = True
    ):
        self.tf_path = tf_path
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.mode = mode
        self.detrend = detrend
        self.window = window
        self._gpu_calibrator = None
        if self.use_gpu:
            self._gpu_calibrator = Microtremor_Calibrator(tf_path=tf_path, use_gpu=True, mode=mode, detrend=detrend, window=window)

    def Gather_Files(self, indir: str, pattern: str) -> List[str]:
        pat = os.path.join(indir, pattern)
        files = sorted(glob.glob(pat))
        return files

    @staticmethod
    def CPU_Processing(
        input_path: str,
        output_path: str,
        tf_path: str,
        mode: str,
        detrend: bool,
        window: bool,
        fs_override: Optional[float] = None,
        chunk_size: Optional[int] = None
    ) -> str:
        calibrator = Microtremor_Calibrator(tf_path=tf_path, use_gpu=False, mode=mode, detrend=detrend, window=window)
        if chunk_size is None:
            return calibrator.Calibrate_File(input_path, output_path=output_path, fs=fs_override)
        t, x, fs_inferred = calibrator.Load_File(input_path)
        if fs_override is None and fs_inferred is None:
            raise ValueError("fs unknown; provide fs_override when using chunking")
        fs = fs_override if fs_override is not None else fs_inferred
        n = x.shape[0]
        y_out = np.zeros_like(x)
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            seg = x[start:stop]
            seg = seg.astype(np.float64)
            y_seg = calibrator.Calibrate_Array(seg, fs)
            if isinstance(y_seg, (np.ndarray,)):
                y_seg_cpu = y_seg
            else:
                y_seg_cpu = cp.asnumpy(y_seg)
            y_out[start:stop] = y_seg_cpu
        if t is not None:
            out = np.column_stack((t, y_out))
            if y_out.ndim == 1:
                header = "time(s) amplitude"
            else:
                ch_labels = [f"ch{i+1}" for i in range(y_out.shape[1])]
                header = "time(s) " + " ".join(ch_labels)
        else:
            out = y_out
            if y_out.ndim == 1:
                header = "amplitude"
            else:
                ch_labels = [f"ch{i+1}" for i in range(y_out.shape[1])]
                header = " ".join(ch_labels)
        np.savetxt(output_path, out, fmt="%.6e", header=header)
        meta_path = os.path.splitext(output_path)[0] + "_meta.npz"
        meta = dict(fs=fs, tf_path=tf_path, mode=mode, detrend=detrend, window=window, chunk_size=chunk_size)
        if y_out.ndim == 2:
            meta['n_channels'] = int(y_out.shape[1])
            meta['channel_labels'] = ch_labels
        else:
            meta['n_channels'] = 1
            meta['channel_labels'] = ['ch1']
        np.savez(meta_path, **meta)
        return output_path

    def Processing_File_List(
        self,
        files: List[str],
        outdir: str,
        workers: Optional[int] = None,
        fs_override: Optional[float] = None,
        chunk_size: Optional[int] = None,
        overwrite: bool = False
    ) -> List[str]:
        os.makedirs(outdir, exist_ok=True)
        if len(files) == 0:
            print("No files to process.")
            return []

        tasks = []
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            outpath = os.path.join(outdir, base + "_calibrated.txt")
            if (not overwrite) and os.path.exists(outpath):
                continue
            tasks.append((f, outpath))

        processed = []
        if self.use_gpu:
            calibrator = self._gpu_calibrator
            iterator = tasks
            if _HAS_TQDM:
                iterator = tqdm(tasks, desc="GPU processing")
            for inp, outp in iterator:
                try:
                    calibrator.Calibrate_File(inp, output_path=outp, fs=fs_override)
                    processed.append(outp)
                except Exception as e:
                    print(f"[ERROR] {inp}: {e}")
                    traceback.print_exc()
        else:
            max_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
            if max_workers < 1:
                max_workers = 1
            worker = functools.partial(Batch_Microtremor_Calibrator.CPU_Processing,
                                       tf_path=self.tf_path,
                                       mode=self.mode,
                                       detrend=self.detrend,
                                       window=self.window,
                                       fs_override=fs_override,
                                       chunk_size=chunk_size)
            if _HAS_TQDM:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(worker, inp, outp): (inp, outp) for inp, outp in tasks}
                    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="CPU processing"):
                        inp, outp = futures[fut]
                        try:
                            res = fut.result()
                            processed.append(res)
                        except Exception as e:
                            print(f"[ERROR] {inp}: {e}")
                            traceback.print_exc()
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(worker, inp, outp) for inp, outp in tasks]
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            res = fut.result()
                            processed.append(res)
                        except Exception as e:
                            print(f"[ERROR] worker failed: {e}")
                            traceback.print_exc()
        return processed

    def Processing_Directory(
        self,
        indir: str,
        outdir: str,
        pattern: str = "*.txt",
        workers: Optional[int] = None,
        fs_override: Optional[float] = None,
        chunk_size: Optional[int] = None,
        overwrite: bool = False
    ) -> List[str]:
        os.makedirs(outdir, exist_ok=True)
        files = self.Gather_Files(indir, pattern)
        return self.Processing_File_List(files, outdir, workers=workers, fs_override=fs_override, chunk_size=chunk_size, overwrite=overwrite)


def fix_decimal_comma_in_dir(indir: str, backup: bool = True) -> List[str]:
    modified = []
    for path in sorted(glob.glob(os.path.join(indir, "*.txt"))):
        changed = False
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        if "," in data and not "." in data.splitlines()[0]:
            newdata = data.replace(",", ".")
            if newdata != data:
                if backup:
                    shutil.copy2(path, path + ".bak")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(newdata)
                modified.append(path)
    return modified

def validate_input_files(indir: str, pattern: str = "*.txt") -> List[str]:
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matching {pattern} in {indir}")
    return files

def execute_pipeline(
    tf_path: str,
    indir: str,
    outdir: str,
    pattern: str = "*.txt",
    use_gpu: bool = False,
    workers: int = None,
    fs: float = None,
    chunk: int = None,
    mode: str = "multiply",
    detrend: bool = True,
    window: bool = True,
    overwrite: bool = False,
    fix_decimal: bool = False
):
    # 1. Basic checks
    if not os.path.exists(tf_path):
        raise FileNotFoundError(f"Transfer function not found: {tf_path}")
    if not os.path.isdir(indir):
        raise FileNotFoundError(f"Input directory not found: {indir}")
    os.makedirs(outdir, exist_ok=True)

    # 2. Optional preprocessing: fix decimal commas
    if fix_decimal:
        print("[INFO] Fixing decimal commas in input directory (backups *.bak created)...")
        modified = fix_decimal_comma_in_dir(indir, backup=True)
        print(f"[INFO] Decimal-fix applied to {len(modified)} files.")

    # 3. Gather files
    files = validate_input_files(indir, pattern)
    print(f"[INFO] Found {len(files)} files to process (pattern={pattern}).")

    # 4. Build batcher
    batcher = Batch_Microtremor_Calibrator(
        tf_path=tf_path,
        use_gpu=use_gpu,
        mode=mode,
        detrend=detrend,
        window=window
    )

    # 5. Run processing
    if use_gpu:
        if workers is not None and workers > 1:
            print("[WARN] GPU mode with multiple OS processes may conflict. Recommended: single-process GPU (--workers 1).")
        processed = batcher.Processing_Directory(
            indir=indir,
            outdir=outdir,
            pattern=pattern,
            workers=1,
            fs_override=fs,
            chunk_size=chunk,
            overwrite=overwrite
        )
    else:
        processed = batcher.Processing_Directory(
            indir=indir,
            outdir=outdir,
            pattern=pattern,
            workers=workers,
            fs_override=fs,
            chunk_size=chunk,
            overwrite=overwrite
        )

    print(f"[DONE] Processed {len(processed)} files. Outputs in: {outdir}")
    return processed


def main():

    parser = argparse.ArgumentParser(
        description="Execute 3C microtremor calibration pipeline (AUTO MODE ENABLED)"
    )

    # Auto + Optional Arguments
    parser.add_argument("--tf", default=TRANSFER_FUNCTION, help=f"Path to transfer function (default: {TRANSFER_FUNCTION})")
    parser.add_argument("--indir", default=INPUT_DATA, help=f"Input directory (default: {INPUT_DATA})")
    parser.add_argument("--outdir", default=OUTPUT_DATA, help=f"Output directory (default: {OUTPUT_DATA})")
    parser.add_argument("--pattern", default="*.txt", help="Filename glob pattern (default '*.txt')")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy). Recommended single process")
    parser.add_argument("--workers", type=int, default=None, help="CPU workers (default: cpu_count-1)")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help=f"Sampling rate Hz (default: {DEFAULT_FS})")
    parser.add_argument("--chunk", type=int, default=None, help="Chunk size (optional, for large data)")
    parser.add_argument("--mode", choices=["multiply", "divide"], default="multiply")
    parser.add_argument("--no-detrend", action="store_true", help="Disable detrending")
    parser.add_argument("--no-window", action="store_true", help="Disable windowing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--fix-decimal", action="store_true", help="Fix comma decimal to dot")

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.tf):
        raise FileNotFoundError(f"Transfer function not found: {args.tf}")

    if not os.path.isdir(args.indir):
        raise FileNotFoundError(f"Input directory not found: {args.indir}")

    os.makedirs(args.outdir, exist_ok=True)

    # Check
    if args.fs is None:
        print("[WARNING] fs not provided → fallback to default:", DEFAULT_FS)
        args.fs = DEFAULT_FS

    # Info Log
    print("\n=== CONFIGURATION ===")
    print(f"TF Path     : {args.tf}")
    print(f"Input Dir   : {args.indir}")
    print(f"Output Dir  : {args.outdir}")
    print(f"GPU         : {args.gpu}")
    print(f"Workers     : {args.workers}")
    print(f"Chunk Size  : {args.chunk}")
    print(f"FS          : {args.fs}")
    print(f"Mode        : {args.mode}")
    print("=====================\n")

    # Execution
    try:
        execute_pipeline(
            tf_path=args.tf,
            indir=args.indir,
            outdir=args.outdir,
            pattern=args.pattern,
            use_gpu=args.gpu,
            workers=args.workers,
            fs=args.fs,
            chunk=args.chunk,
            mode=args.mode,
            detrend=(not args.no_detrend),
            window=(not args.no_window),
            overwrite=args.overwrite,
            fix_decimal=args.fix_decimal
        )

    except Exception as e:
        print("\n[ERROR] Pipeline failed:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()







