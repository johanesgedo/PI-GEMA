# Hybrid Processing of Microtremor Data using CuHVSR and BFWI
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


import sys
import pathlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from obspy import read, UTCDateTime, Trace, Stream
from scipy.signal import welch, resample
from scipy.interpolate import interp1d
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from scipy.spatial import ConvexHull

try:
    import cupy as cp
except Exception:
    cp = None

from pigema.cuhvsr import CuHVSR
from pigema.bfwi import BFWI
from pathlib import Path
# from config import CALIBRATED_OUTPUT


plt.style.use(CuHVSR.HVSRPY_MPL_STYLE)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOTS = [
    PROJECT_ROOT / "data" / "microtremor",
    PROJECT_ROOT / "data" / "calibrated"
]

case_id = "sample_001_calibrated"

output_dir = CALIBRATED_OUTPUT / case_id

output_dir.mkdir(
    parents=True,
    exist_ok=True
)



#-------------------------
# Helper utilities
#-------------------------
def ensure_exists(fnames):
    if isinstance(fnames, (list, tuple)) and len(fnames) > 0 and isinstance(fnames[0], (list, tuple)):
        for group in fnames:
            for f in group:
                if not pathlib.Path(f).exists():
                    raise FileNotFoundError(f"File {f} not found; check spelling.")
    else:
        if isinstance(fnames, (list, tuple)):
            for f in fnames:
                if not pathlib.Path(f).exists():
                    raise FileNotFoundError(f"File {f} not found; check spelling.")
        else:
            if not pathlib.Path(fnames).exists():
                raise FileNotFoundError(f"File {fnames} not found; check spelling.")


def to_host(arr):
    if arr is None:
        return None
    try:
        if hasattr(arr, "get") and callable(arr.get):
            return np.asarray(arr.get())
        if cp is not None and isinstance(arr, cp.ndarray):
            return np.asarray(arr.get())
        return np.asarray(arr)
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return None


def txt_to_miniseed(
    input_txt,
    output_mseed,
    station="STA",
    network="XX",
    sampling_rate=100.0,
    start_time="2025-10-01T00:00:00"
):
    data = np.loadtxt(input_txt, dtype=float)
    N, E, Z = data[:,0], data[:,1], data[:,2]

    start = UTCDateTime(start_time)

    trZ = Trace(data=Z)
    trZ.stats.station = station
    trZ.stats.network = network
    trZ.stats.channel = "BHZ"
    trZ.stats.starttime = start
    trZ.stats.sampling_rate = sampling_rate

    trN = Trace(data=N)
    trN.stats.station = station
    trN.stats.network = network
    trN.stats.channel = "BHN"
    trN.stats.starttime = start
    trN.stats.sampling_rate = sampling_rate

    trE = Trace(data=E)
    trE.stats.station = station
    trE.stats.network = network
    trE.stats.channel = "BHE"
    trE.stats.starttime = start
    trE.stats.sampling_rate = sampling_rate

    st = Stream([trZ, trN, trE])

    st.write(output_mseed, format="MSEED")
    print(f"MiniSEED file has been successfully saved: {output_mseed}")


def miniseed_to_csv(
    input_mseed,
    output_csv,
    chan_order=("BHE", "BHN", "BHZ")
):
    st = read(input_mseed)

    chan_map = {}
    for tr in st:
        ch = tr.stats.channel
        if ch not in chan_map:
            chan_map[ch] = tr

    src = {tr.stats.sampling_rate for tr in chan_map.values()}
    if len(src) > 1:
        warnings.warn(
            f"Multiple sampling rates found in miniSEED: {src}. Attempting to resample to first SR.",
            UserWarning
        )
        # target_sr = next(iter(src))
        target_sr = sorted(src)[0]
        for k, tr in list(chan_map.items()):
            if tr.stats.sampling_rate != target_sr:
                tr.interpolate(sampling_rate=target_sr, starttime=tr.stats.starttime,
                               npts=int(round(tr.stats.npts * target_sr / tr.stats.sampling_rate)))
                chan_map[k] = tr

    traces = []
    for ch in chan_order:
        if ch in chan_map:
            traces.append(chan_map[ch])
        else:
            warnings.warn(f"Channel {ch} missing in {input_mseed}. Filling with zeros.", UserWarning)
            traces.append(None)

    present_traces = [tr for tr in traces if tr is not None]
    if len(present_traces) == 0:
        raise ValueError("No requested channels present in input miniseed.")

    latest_start = max(tr.stats.starttime for tr in present_traces)
    earliest_end = min(tr.stats.endtime for tr in present_traces)
    if earliest_end <= latest_start:
        min_npts = min(tr.stats.npts for tr in present_traces)
        aligned_arrays = []
        for tr in traces:
            if tr is None:
                aligned_arrays.append(np.zeros(min_npts, dtype=float))
            else:
                arr = tr.data.astype(float)
                if arr.size >= min_npts:
                    aligned_arrays.append(arr[:min_npts])
                else:
                    pad = np.zeros(min_npts - arr.size, dtype=float)
                    aligned_arrays.append(np.concatenate([arr, pad]))
    else:
        dur = earliest_end - latest_start
        sr = present_traces[0].stats.sampling_rate
        npts = int(round(dur * sr)) + 1
        aligned_arrays = []
        for tr in traces:
            if tr is None:
                aligned_arrays.append(np.zeros(npts, dtype=float))
            else:
                tr_slice = tr.copy()
                try:
                    tr_slice.trim(starttime=latest_start, endtime=earliest_end, pad=False)
                    arr = tr_slice.data.astype(float)
                    if arr.size < npts:
                        pad = np.zeros(npts - arr.size, dtype=float)
                        arr = np.concatenate([arr, pad])
                    elif arr.size > npts:
                        arr = arr[:npts]
                    aligned_arrays.append(arr)
                except Exception:
                    arr = tr.data.astype(float)
                    if arr.size >= npts:
                        aligned_arrays.append(arr[:npts])
                    else:
                        pad = np.zeros(npts - arr.size, dtype=float)
                        aligned_arrays.append(np.concatenate([arr, pad]))

    col_names = [ch for ch in chan_order]
    df = pd.DataFrame({name: aligned_arrays[i] for i, name in enumerate(col_names)})
    df.to_csv(output_csv, index=False, header=False)
    print(f"Saved CSV to {output_csv} with columns {col_names} (missing channels filled with zeros).")


def sanitize_obs_array_from_csv(path):
    df = pd.read_csv(path, header=None)
    arr = df.values
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        ncols = arr.shape[1]
        if ncols == 3:
            return arr.T
        elif ncols == 2:
            warnings.warn("CSV has 2 columns -> padding third column with zeros.")
            padded = np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)])
            return padded.T
        elif ncols == 1:
            return arr.flatten()
        elif ncols > 3:
            warnings.warn("CSV has >3 columns -> using the first 3 columns.", UserWarning)
            return arr[:, :3].T
    raise ValueError("Unsupported data shape read from CSV.")


def safe_plot_lines_from_dict(
    ax,
    mean_curves,
    freq,
    smoothing_options=None
):
    plotted_any = False
    freq = np.asarray(freq, dtype=float)
    for label, curve in mean_curves.items():
        if curve is None:
            print(f"Skipping '{label}': no curve (None).")
            continue
        m = to_host(curve)
        if m is None:
            print(f"Skipping '{label}': conversion to host failed.")
            continue
        m = np.asarray(m, dtype=float)

        if m.size == 0 or not np.isfinite(m).any():
            print(f"Skipping '{label}': empty or all-NaN.")
            continue

        if m.size != freq.size:
            if smoothing_options and label in smoothing_options:
                src = smoothing_options[label].get("center_frequencies_in_hz", None)
                if src is not None:
                    try:
                        src = to_host(src)
                        src = np.asarray(src, dtype=float)
                        if src.size == m.size:
                            interp = interp1d(src, m, bounds_error=False, fill_value=np.nan, assume_sorted=True)
                            m = interp(freq)
                            m = np.asarray(m, dtype=float)
                            print(f"Interpolated '{label}' from {src.size} -> {freq.size}.")
                        else:
                            print(f"Skipping '{label}': cannot interp (src size {src.size} != curve {m.size}).")
                            continue
                    except Exception as e:
                        print(f"Skipping '{label}': interpolation failed: {e}")
                        continue
                else:
                    print(f"Skipping '{label}': length mismatch and no source freqs available.")
                    continue
            else:
                print(f"Skipping '{label}': length mismatch (curve {m.size} vs freq {freq.size}).")
                continue

        if not np.isfinite(m).any():
            print(f"Skipping '{label}': no finite values after interpolation.")
            continue

        ax.plot(freq, m, label=label, lw=1.5)
        plotted_any = True

    return plotted_any


def make_preproc(
    window_len=20,
    detrend='linear',
    orient_deg=0.0,
    filter_corner=(None, None)
):
    s = CuHVSR.HvsrPreProcessingSettings()
    s.detrend = detrend
    s.window_length_in_seconds = window_len
    if hasattr(s, "orient_to_degrees_from_north"):
        s.orient_to_degrees_from_north = orient_deg
    elif hasattr(s, "orient_to_degrees_From_north"):
        s.orient_to_degrees_From_north = orient_deg
    s.filter_corner_frequencies_in_hz = filter_corner
    s.ignore_dissimilar_time_step_warning = False
    return s


def make_traditional_proc(
    window_type=("tukey", 0.2),
    smoothing_op="konno_and_ohmachi",
    bandwidth=40,
    center_freqs=None,
    combine_method="geometric_mean",
    handle_dts="frequency_domain_resampling"
):
    p = CuHVSR.HvsrTraditionalProcessingSettings()
    p.window_type_and_width = window_type
    if center_freqs is None:
        center_freqs = np.geomspace(0.2, 50, 200)
    p.smoothing = dict(operator=smoothing_op, bandwidth=bandwidth,
                       center_frequencies_in_hz=np.asarray(center_freqs, dtype=float))
    p.method_to_combine_horizontals = combine_method
    p.handle_dissimilar_time_steps_by = handle_dts
    return p


def load_spatial_input_csv(csv_file):
    csv_file = Path(csv_file)
    if not csv_file.is_file():
        print(f"[WARNING] Spatial CSV not found: {csv_file}")
        return None
    df = pd.read_csv(csv_file)
    required_cols = {"x", "y", "mean", "std"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Spatial CSV missing columns: {sorted(missing)}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "mean", "std"])
    coords = df[["x", "y"]].to_numpy(dtype=np.float32)
    generator_means = df["mean"].to_numpy(dtype=np.float32)
    generator_stddevs = df["std"].to_numpy(dtype=np.float32)
    if coords.shape[0] < 3:
        print("[WARNING] Spatial HVSR requires at least 3 spatial points. Skipping.")
        return None
    return coords, generator_means, generator_stddevs


@dataclass
class CasePaths:
    root: Path
    case_id: str
    data_roots: list[Path] = field(default_factory=list)

    @property
    def case_dir(self):
        return self.root / self.case_id

    @property
    def input_dir(self):
        return self.case_dir / "input"

    @property
    def intermediate_dir(self):
        return self.case_dir / "intermediate"

    @property
    def results_dir(self):
        return self.case_dir / "results"

    def ensure(self):
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # def raw_txt(self):
    #     return self.case_dir / f"{self.case_id}.txt"

    def raw_txt(self):
        candidates = []
        candidates.append(self.case_dir / f"{self.case_id}.txt")
        for root in self.data_roots:
            candidates.append(root / f"{self.case_id}.txt")
            candidates.append(root / self.case_id / f"{self.case_id}.txt")
        for p in candidates:
            print(f"[CHECK] {p} -> {p.exists()}")
        for p in candidates:
            if p.exists():
                print(f"[INFO] Using data: {p}")
                return p
        raise FileNotFoundError(f"No valid file found for {self.case_id}")

    def mseed(self):
        return self.intermediate_dir / f"{self.case_id}_raw_to_mseed.mseed"

    def csv(self):
        return self.intermediate_dir / f"{self.case_id}_mseed_to_csv.csv"

    def hvsr_png(self, tag="traditional"):
        return self.results_dir / f"{self.case_id}_hvsr_{tag}.png"

    def bfwi_dir(self):
        return self.results_dir / f"{self.case_id}_bfwi_out"


paths = CasePaths(
    root=PROJECT_ROOT,
    case_id=case_id,
    data_roots=DATA_ROOTS
)
paths.ensure()



#=============================
# Main flow (sections 1-10)
#=============================
# if __name__ == "__main__":

def main():

    raw_txt = paths.raw_txt()
    if not raw_txt.exists():
        raise FileNotFoundError(
            f"Raw txt is not found: {raw_txt}\n"
            f"Put the source file in the location, or change the CasePaths.raw_txt() to match the data structure."
        )

    #-----------------------------------------
    # Convert data from .txt to .miniseed
    #-----------------------------------------
    txt_to_miniseed(
        input_txt=str(paths.raw_txt()),
        output_mseed=str(paths.mseed()),
        station="STA",
        network="XX",
        sampling_rate=100.0,
        start_time="2026-04-20T00:00:00"
    )

    #-------------------------------------
    # Section 1: Preprocessing example
    #-------------------------------------
    print("\n1) Quick preprocess + plot raw vs preprocessed")

    mseed_file = paths.mseed()
    fnames = [[str(mseed_file)]]
    ensure_exists(fnames)

    srecords = CuHVSR.Read(fnames)

    preproc = make_preproc(window_len=10, detrend="linear", orient_deg=0.0)
    preproc.psummary()

    srecords_pp = CuHVSR.preprocess(srecords, preproc)

    # Plot Raw Data
    print("Seismic recording: raw")
    fig_raw, axs_raw = CuHVSR.plot_seismic_recordings_3c(srecords)
    raw_plot_path = paths.results_dir / f"{paths.case_id}_raw_seismic.png"
    fig_raw.savefig(raw_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Raw plot saved: {raw_plot_path}")

    # Plot Preprocessed
    print("Seismic recordings: preprocessed")
    fig_pp, axs_pp = CuHVSR.plot_seismic_recordings_3c(srecords_pp)
    pp_plot_path = paths.results_dir / f"{paths.case_id}_preprocessed_seismic.png"
    fig_pp.savefig(pp_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Preprocessed plot saved: {pp_plot_path}")

    # Save metadata / sanity check
    try:
        print("\nQuick sanity check:")
        print(f"Number of records: {len(srecords_pp)}")
    except Exception as e:
        print("Warning during sanity check:", e)


    #------------------------------------
    # Section 2: Window rejection
    #------------------------------------
    print("\n2) HVSR Traditional Window Rejection")

    mseed_file = paths.mseed()
    fnames9 = [[str(mseed_file)]]
    ensure_exists(fnames9)

    print(f"[INFO] Using file: {mseed_file}")

    # preproc9 = make_preproc(window_len=10, detrend="constant")
    # proc9 = make_traditional_proc(center_freqs=np.geomspace(0.2,50,200))

    preproc9 = make_preproc(window_len=10, detrend="linear")
    proc9 = make_traditional_proc(center_freqs=np.geomspace(0.2, 20, 150))

    srecords9 = CuHVSR.Read(fnames9)
    srecords9 = CuHVSR.preprocess(srecords9, preproc9)

    hvsr9 = CuHVSR.process(srecords9, proc9)

    initial_windows = np.sum(hvsr9.valid_window_boolean_mask)
    print(f"[BEFORE] Valid windows: {initial_windows}")

    try:
        CuHVSR.frequency_domain_window_rejection(
            hvsr9,
            # n=2,
            # search_range_in_hz=(0.2, 10)
            n=1.5,
            search_range_in_hz=(0.2, 10)
        )
    except Exception as e:
        print("Warning: window rejection failed:", e)

    final_windows = np.sum(hvsr9.valid_window_boolean_mask)
    print(f"[AFTER] Valid window: {final_windows}")
    if final_windows == 0:
        raise RuntimeError("All windows rejected! Check threshold or data quality.")
    rejected = initial_windows - final_windows
    print(f"[INFO] Rejected windows: {rejected}")

    mfig, axs = CuHVSR.plot_pre_and_post_rejection(srecords9, hvsr9)
    plot_path = paths.results_dir / f"{paths.case_id}_window_rejection.png"
    mfig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Window rejection plot saved: {plot_path}")
    print("Script finished")

    #-----------------------------
    # Section 3: Spatial HVSR
    #-----------------------------
    print("\n3) Spatial HVSR")

    spatial_csv = paths.results_dir / f"{paths.case_id}_spatial_input.csv"
    spatial_data = load_spatial_input_csv(spatial_csv)

    if spatial_data is None:
        print("[INFO] Spatial HVSR skipped because valid multi-station input is not available.")
    else:
        coords, generator_means, generator_stddevs = spatial_data

        try:
            hull = ConvexHull(coords)
            boundary = coords[hull.vertices]
        except Exception as e:
            print(f"[WARNING] ConvexHull failed ({e}); using coords as boundary.")
            boundary = coords.copy()

        try:
            spatial = CuHVSR.HvsrSpatial(coords)

            generator_weights, valid_indices = spatial.spatial_weights(
                boundary,
                declustering_method="voronoi"
            )
            valid_indices = np.asarray(valid_indices, dtype=int)

            if valid_indices.size == 0:
                raise RuntimeError("No valid spatial indices after Voronoi declustering.")

            valid_coords = coords[valid_indices]

            tessellation_vertices, tessellation_sensor_indices = spatial.bounded_voronoi(boundary)

            spatial_mean, spatial_stddev, _ = CuHVSR.montecarlo_fn(
                generator_means[valid_indices],
                generator_stddevs[valid_indices],
                generator_weights,
                distribution_generators="lognormal",
                distribution_spatial="lognormal"
            )

            print("\n[Spatial HVSR Statistics]")
            CuHVSR.summarize_spatial_statistics(spatial_mean, spatial_stddev, "lognormal")

            fig = CuHVSR.plot_voronoi(
                valid_coords,
                np.exp(generator_means[valid_indices]),
                tessellation_vertices,
                boundary
            )

            plot_path = paths.results_dir / f"{paths.case_id}_spatial_hvsr.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"[INFO] Spatial HVSR plot saved: {plot_path}")

        except Exception as e:
            print(f"[WARNING] Spatial HVSR processing failed: {e}")



    #-----------------------------------
    # Section 4: HVSR IO checkpoint
    #-----------------------------------
    print("\n4) HVSR IO Check Point")

    hvsr_checkpoint = hvsr9

    try:
        hvsr_checkpoint.Update_Peaks_Bounded(search_range_in_hz=(0.2, 10))
    except Exception as e:
        print("Warning: Update_Peaks_Bounded failed:", e)

    hvsr_obj_path = paths.results_dir / f"{paths.case_id}_hvsr_io.csv"
    try:
        CuHVSR.Object_IO_CUDA.write_hvsr_object_to_file(
            hvsr_checkpoint,
            str(hvsr_obj_path)
        )
        print(f"HVSR object saved: {hvsr_obj_path}")
    except Exception as e:
        print("Warning: failed to save HVSR to disk:", e)

    try:
        hvsr_io = CuHVSR.Object_IO_CUDA.read_hvsr_object_from_file(
            str(hvsr_obj_path),
            allow_clean_nan=False
        )
        print("IO round-trip completed.")

        try:
            print("is_similar:", hvsr_checkpoint.is_similar(hvsr_io))
        except Exception as e:
            print("Warning: similarity check failed:", e)

        fig, ax = CuHVSR.plot_single_panel_hvsr_curves(
            hvsr_io,
            plot_valid_curves=True,
            plot_invalid_curves=False,
            plot_mean_curve=True,
            plot_frequency_std=False,
            plot_peak_mean_curve=True,
            plot_peak_individual_valid_curves=False,
            plot_peak_individual_invalid_curves=False,
        )
        ax.set_xlim(0.2, 20)
        ax.set_ylim(0, max(5.0, float(np.nanmax(to_host(hvsr_io.mean_curve()))) * 1.5))

        plot_path = paths.results_dir / f"{paths.case_id}_hvsr_io_checkpoint.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Plot saved: {plot_path}")

    except Exception as e:
        print("Warning: IO checkpoint read-back failed:", e)


    #-----------------------------------------------------------------------
    # Section 5: HVSR CLI (Common Line Interface) / Smoothing Comparisons
    #-----------------------------------------------------------------------
    print("\n5) HVSR Smoothing Comparisons (many operators)")

    smoothing_options = {
        "Parzen | b=0.5": dict(
            operator="parzen",
            bandwidth=0.5,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Savitzky and Golay (1964) | b=71": dict(
            operator="savitzky_and_golay",
            bandwidth=71,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Linear Rectangular | b=0.4": dict(
            operator="linear_rectangular",
            bandwidth=0.4,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Log Rectangular | b=0.07": dict(
            operator="log_rectangular",
            bandwidth=0.07,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Linear Triangular | b=0.4": dict(
            operator="linear_triangular",
            bandwidth=0.4,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Log Triangular | b=0.07": dict(
            operator="log_triangular",
            bandwidth=0.07,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
        "Konno and Ohmachi (1998) | b=40": dict(
            operator="konno_and_ohmachi",
            bandwidth=40,
            center_frequencies_in_hz=np.geomspace(0.2, 40, 128),
        ),
    }

    preproc_smooth = make_preproc(window_len=10, detrend="linear", orient_deg=0.0)
    preproc_smooth.psummary()

    srecords2 = CuHVSR.Read(fnames)
    srecords2 = CuHVSR.preprocess(srecords2, preproc_smooth)

    mean_curves = {}
    last_hvsr = None

    for label, sdict in smoothing_options.items():
        print(f"\nProcessing smoothing: {label}")
        proc = make_traditional_proc(
            center_freqs=sdict["center_frequencies_in_hz"],
            smoothing_op=sdict["operator"],
            bandwidth=sdict["bandwidth"],
        )
        try:
            hvsr = CuHVSR.process(srecords2, proc)
            last_hvsr = hvsr
            mean_curves[label] = hvsr.mean_curve()
        except Exception as e:
            print(f"Processing failed for '{label}': {e}")
            mean_curves[label] = None

    freq = None
    if last_hvsr is not None:
        freq = to_host(last_hvsr.frequency)

    if freq is None:
        any_sdict = next(iter(smoothing_options.values()))
        freq = np.asarray(any_sdict["center_frequencies_in_hz"], dtype=float)

    if last_hvsr is not None:
        print("\nStatistical Summary for HVSR Smoothing Comparison:")
        CuHVSR.summarize_hvsr_statistics(last_hvsr)

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)
    safe_plot_lines_from_dict(ax, mean_curves, freq, smoothing_options)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("HVSR Amplitude")
    ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.plot([], [], label="No valid mean curves")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plot_path = paths.results_dir / f"{paths.case_id}_hvsr_smoothing_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {plot_path}")

    if last_hvsr is not None:
        hvsr_obj_path = paths.results_dir / f"{paths.case_id}_hvsr_smoothing_comparison.csv"
        try:
            CuHVSR.Object_IO_CUDA.write_hvsr_object_to_file(
                last_hvsr,
                str(hvsr_obj_path),
            )
            print(f"HVSR object saved: {hvsr_obj_path}")
        except Exception as e:
            print("Warning: failed to save HVSR to disk:", e)


    #-------------------------------------
    # Section 6: HVSR Traditional
    #-------------------------------------
    print("\n6) HVSR Traditional")

    mseed_file = paths.mseed()
    fnames = [[str(mseed_file)]]
    ensure_exists(fnames)

    srecords6 = CuHVSR.Read(fnames)

    preproc6 = make_preproc(window_len=10, detrend="linear")
    preproc6.psummary()
    srecords6 = CuHVSR.preprocess(srecords6, preproc6)

    proc6 = make_traditional_proc(center_freqs=np.geomspace(0.2, 50, 200))
    hvsr6 = CuHVSR.process(srecords6, proc6)

    print("\nStatistical Summary for HVSR Traditional:")
    CuHVSR.summarize_hvsr_statistics(hvsr6)

    fig, ax = CuHVSR.plot_single_panel_hvsr_curves(
        hvsr6,
        plot_mean_curve=True,
        plot_frequency_std=False,
        plot_peak_mean_curve=True,
        plot_peak_individual_valid_curves=False,
        plot_peak_individual_invalid_curves=False,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plot_path = paths.results_dir / f"{paths.case_id}_hvsr_traditional.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {plot_path}")

    hvsr_obj_path = paths.results_dir / f"{paths.case_id}_hvsr_traditional.csv"
    try:
        CuHVSR.Object_IO_CUDA.write_hvsr_object_to_file(
            hvsr6,
            str(hvsr_obj_path),
        )
        print(f"HVSR object saved: {hvsr_obj_path}")
    except Exception as e:
        print("Warning: failed to save HVSR to disk:", e)


    #-------------------------------
    # Section 7: HVSR Azimuthal
    #-------------------------------
    print("\n7) HVSR Azimuthal")

    mseed_file = paths.mseed()

    fnames_az = [[str(mseed_file)]]
    ensure_exists(fnames_az)

    preproc4 = make_preproc(
        window_len=10,
        detrend="constant",
        orient_deg=0.0
    )

    proc4 = CuHVSR.HvsrAzimuthalProcessingSettings()
    proc4.window_type_and_width = ("tukey", 0.1)
    proc4.smoothing = dict(
        operator="konno_and_ohmachi",
        bandwidth=40,
        center_frequencies_in_hz=np.geomspace(0.2, 50, 200)
    )
    proc4.handle_dissimilar_time_steps_by = "frequency_domain_resampling"

    proc4.azimuths_in_degrees = np.arange(0, 180, 5)
    if hasattr(proc4, "psummary"):
        proc4.psummary()
    srecords4 = CuHVSR.Read(fnames_az)
    srecords4 = CuHVSR.preprocess(srecords4, preproc4)
    if len(srecords4) == 0:
        raise ValueError("No seismic records found for azimuthal processing.")
    hvsr4 = CuHVSR.process(srecords4, proc4)
    if hvsr4 is None:
        raise RuntimeError("HVSR azimuthal processing failed.")
    CuHVSR.summarize_hvsr_statistics(hvsr4)
    fig, axs = CuHVSR.plot_azimuthal_summary(
        hvsr4,
        plot_mean_curve=True,
        plot_frequency_std=False,
        plot_peak_mean_curve=True,
        plot_peak_individual_valid_curves=False,
        plot_peak_individual_invalid_curves=False
    )
    plot_path = paths.results_dir / f"{paths.case_id}_hvsr_azimuthal.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Azimuthal HVSR saved: {plot_path}")



    #------------------------------
    # Section 8: Diffuse-field
    #------------------------------
    print("\n8) HVSR Diffuse Field")

    mseed_file = paths.mseed()
    fnames = [[str(mseed_file)]]
    ensure_exists(fnames)

    srecords5 = CuHVSR.Read(fnames)

    preproc5 = make_preproc(window_len=10, detrend="linear")
    preproc5.psummary()

    srecords5 = CuHVSR.preprocess(srecords5, preproc5)

    proc5 = CuHVSR.HvsrDiffuseFieldProcessingSettings()
    proc5.window_type_and_width = ("tukey", 0.1)
    proc5.smoothing = dict(
        operator="log_rectangular",
        bandwidth=0.1,
        center_frequencies_in_hz=np.geomspace(0.2, 50, 256),
    )

    if hasattr(proc5, "psummary"):
        proc5.psummary()

    hvsr5 = CuHVSR.process(srecords5, proc5)

    print("\nStatistical Summary for HVSR Diffuse Field:")
    CuHVSR.summarize_hvsr_statistics(hvsr5)

    fig, ax = CuHVSR.plot_single_panel_hvsr_curves(hvsr5)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plot_path = paths.results_dir / f"{paths.case_id}_hvsr_diffuse_field.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {plot_path}")

    hvsr_obj_path = paths.results_dir / f"{paths.case_id}_hvsr_diffuse_field.csv"
    try:
        CuHVSR.Object_IO_CUDA.write_hvsr_object_to_file(
            hvsr5,
            str(hvsr_obj_path),
        )
        print(f"HVSR object saved: {hvsr_obj_path}")
    except Exception as e:
        print("Warning: failed to save HVSR to disk:", e)


    #---------------------------------------------
    # Section 9: Traditional + SESAME checks
    #---------------------------------------------
    print("\n9) HVSR Traditional SESAME checks")
    preproc7 = make_preproc(window_len=10, detrend="linear")
    proc7 = make_traditional_proc(center_freqs=np.geomspace(0.2, 50, 200))
    srecords7 = CuHVSR.Read(fnames)
    srecords7 = CuHVSR.preprocess(srecords7, preproc7)
    hvsr7 = CuHVSR.process(srecords7, proc7)

    # try:
    #     hvsr7.update_peaks_bounded(search_range_in_hz=(None, None))
    # except Exception as e:
    #     print("Warning: update_peaks_bounded failed:", e)

    try:
        # hvsr7.Update_Peaks_Bounded(search_range_in_hz=(None, None))
        hvsr7.Update_Peaks_Bounded(search_range_in_hz=(0.02, 1))
    except Exception as e:
        print("Warning: update_peaks_bounded failed:", e)

    if hasattr(CuHVSR, "Sesame_CUDA"):
        try:
            CuHVSR.Sesame_CUDA.reliability(
                windowlength=preproc7.window_length_in_seconds,
                passing_window_count=np.sum(hvsr7.valid_window_boolean_mask),
                frequency=hvsr7.frequency,
                mean_curve=hvsr7.mean_curve(distribution="lognormal"),
                std_curve=hvsr7.std_curve(distribution="lognormal"),
                search_range_in_hz=(None, None),
                verbose=1,
            )
        except Exception as e:
            print("Warning: Sesame reliability check failed:", e)

        try:
            CuHVSR.Sesame_CUDA.clarity(
                frequency=hvsr7.frequency,
                mean_curve=hvsr7.mean_curve(distribution="lognormal"),
                std_curve=hvsr7.std_curve(distribution="lognormal"),
                fn_std=hvsr7.std_fn_frequency(distribution="normal"),
                search_range_in_hz=(None, None),
                verbose=1,
            )
        except Exception as e:
            print("Warning: Sesame clarity check failed:", e)
    else:
        print("Warning: Sesame_CUDA is not available in current CuHVSR module.")

    CuHVSR.summarize_hvsr_statistics(hvsr7)
    fig, ax = CuHVSR.plot_single_panel_hvsr_curves(hvsr7)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    plot_path = paths.results_dir / f"{paths.case_id}_hvsr_sesame.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")



    #----------------------------------------
    # Section 10: Full Waveform Inversion
    #----------------------------------------
    def load_or_build_prior(
        cfg_path,
        hv_freqs=None,
        hv_values=None
    ):
        """
        if os.path.isfile(cfg_path):
            print(f"[BFWI] Loading prior from: {cfg_path}")
            with open(cfg_path, "r") as f:
                return json.load(f)
        """

        if os.path.isfile(cfg_path):
            print(f"[BFWI] Loading prior from: {cfg_path}")
            try:
                with open(cfg_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("[BFWI] Corrupted prior JSON detected → rebuilding.")
                os.remove(cfg_path)

        print("[BFWI] Prior config not found → building default prior")

        #--------------------------
        # Auto Prior Generation
        #--------------------------
        max_depth = 21.0
        dz_layer = 3.0
        layers = list(np.arange(dz_layer, max_depth + dz_layer, dz_layer))

        n_layer = len(layers) - 1

        vs_mean = np.linspace(80, 300, n_layer)
        vs_std = 0.2 * vs_mean
        # rho_mean = 1600 + 0.6 * vs_mean
        rho_mean = 1000 + 0.6 * vs_mean
        rho_std = 0.05 * rho_mean

        prior_cfg = {
            # "layer_thickness": layers,
            "layer_thickness": np.asarray(layers).tolist(),
            "vs_mean": vs_mean.tolist(),
            "vs_std": vs_std.tolist(),
            "rho_mean": rho_mean.tolist(),
            "rho_std": rho_std.tolist(),
            "dz": 0.5,
            "t_total": 1.0,
            "f0": 4.0,
            "fmin": 0.2,
            "fmax": 20.0
        }

        with open(cfg_path, "w") as f:
            json.dump(prior_cfg, f, indent=2)
        print(f"[BFWI] Auto prior saved → {cfg_path}")

        return prior_cfg


    def validate_1d_positive_array(
        name,
        arr,
        min_size=1
    ):
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size < min_size:
            raise ValueError(f"{name} must have at least {min_size} elements.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/inf.")
        if np.any(arr <= 0.0):
            raise ValueError(f"{name} must contain strictly positive values.")
        return arr


    def prepare_hvsr_data(
        hvsr_obj,
        fmin=0.2,
        fmax=20.0
    ):
        if hvsr_obj is None:
            raise ValueError("HVSR object is not available. Run HVSR processing first.")

        try:
            hv_values = to_host(hvsr_obj.mean_curve(distribution="lognormal"))
        except Exception:
            hv_values = to_host(hvsr_obj.mean_curve())

        hv_freqs = to_host(hvsr_obj.frequency)

        hv_values = validate_1d_positive_array("hv_values", hv_values, min_size=1)
        hv_freqs = validate_1d_positive_array("hv_freqs", hv_freqs, min_size=1)

        if hv_values.shape != hv_freqs.shape:
            raise ValueError(
                f"HVSR shape mismatch: hv_values={hv_values.shape}, hv_freqs={hv_freqs.shape}"
            )

        mask = np.isfinite(hv_values) & np.isfinite(hv_freqs) & (hv_values > 0.0) & (hv_freqs > 0.0)
        hv_values = hv_values[mask]
        hv_freqs = hv_freqs[mask]

        if hv_values.size == 0:
            raise ValueError("After cleaning, HVSR becomes empty.")

        mask = (hv_freqs >= fmin) & (hv_freqs <= fmax)
        hv_freqs = hv_freqs[mask]
        hv_values = hv_values[mask]

        if hv_values.size < 5:
            raise ValueError(
                f"Too few HVSR points in the {fmin}-{fmax} Hz band. "
                "Widen the band or improve preprocessing."
            )

        return hv_freqs, hv_values

    print("\n10) Full Waveform Inversion using precomputed HVSR")

    hvsr_obj = hvsr9
    prior_cfg_path = str(paths.results_dir / f"{paths.case_id}_prior_bfwi.json")

    hv_freqs, hv_values = prepare_hvsr_data(
        hvsr_obj,
        fmin=0.2,
        fmax=20.0
    )

    hv_values = hv_values / (np.max(hv_values) + 1e-12)

    map_outdir = str(paths.bfwi_dir())
    os.makedirs(map_outdir, exist_ok=True)

    hv_input_csv = str(paths.results_dir / f"{paths.case_id}_hvsr_input_for_fwi.csv")
    np.savetxt(
        hv_input_csv,
        np.column_stack([hv_freqs, hv_values]),
        delimiter=",",
        header="frequency_hz,hvsr",
        comments=""
    )
    print(f"[BFWI] HVSR input saved: {hv_input_csv}")

    prior_cfg = load_or_build_prior(
        prior_cfg_path,
        hv_freqs=hv_freqs,
        hv_values=hv_values
    )

    prior_layers = validate_1d_positive_array(
        "layer_thickness",
        prior_cfg["layer_thickness"],
        min_size=1
    )

    prior_vs_mean = validate_1d_positive_array(
        "vs_mean",
        prior_cfg["vs_mean"],
        min_size=1
    )

    prior_rho_mean = validate_1d_positive_array(
        "rho_mean",
        prior_cfg["rho_mean"],
        min_size=1
    )

    prior_vs_std = np.asarray(prior_cfg.get("vs_std", 0.2 * prior_vs_mean), dtype=float).ravel()
    prior_rho_std = np.asarray(prior_cfg.get("rho_std", 0.05 * prior_rho_mean), dtype=float).ravel()

    if prior_vs_std.shape != prior_vs_mean.shape:
        raise ValueError("vs_std must have the same shape as vs_mean.")
    if prior_rho_std.shape != prior_rho_mean.shape:
        raise ValueError("rho_std must have the same shape as rho_mean.")

    dz = float(prior_cfg.get("dz", 0.5))
    t_total = float(prior_cfg.get("t_total", 1.0))
    f0 = float(prior_cfg.get("f0", 4.0))
    fmin = float(prior_cfg.get("fmin", 0.2))
    fmax = float(prior_cfg.get("fmax", 20.0))

    mask = (hv_freqs >= fmin) & (hv_freqs <= fmax)
    hv_freqs = hv_freqs[mask]
    hv_values = hv_values[mask]

    if hv_values.size < 5:
        raise ValueError(f"Too few HVSR points after applying fmin={fmin}, fmax={fmax}.")

    hv_noise_std = float(1.4826 * np.median(np.abs(hv_values - np.median(hv_values))))
    if not np.isfinite(hv_noise_std) or hv_noise_std <= 0.0:
        hv_noise_std = 0.1

    # hv_values_proc = np.log(hv_values + 1e-12)
    # hv_noise_std = float(np.std(hv_values_proc) * 0.5)
    # hv_values = hv_values_proc

    prior_summary_path = str(paths.results_dir / f"{paths.case_id}_bfwi_prior_summary.json")
    with open(prior_summary_path, "w") as f:
        json.dump(
            {
                "layer_thickness": prior_layers.tolist(),
                "vs_mean": prior_vs_mean.tolist(),
                "vs_std": prior_vs_std.tolist(),
                "rho_mean": prior_rho_mean.tolist(),
                "rho_std": prior_rho_std.tolist(),
                "dz": dz,
                "t_total": t_total,
                "f0": f0,
                "fmin": fmin,
                "fmax": fmax,
                "hv_noise_std": hv_noise_std
            },
            f,
            indent=2
        )
    print(f"[BFWI] Prior summary saved: {prior_summary_path}")

    b = BFWI.Bayesian_Full_Waveform_Inversion(
        layer_thickness=prior_layers,
        layer_vs_true=prior_vs_mean,
        layer_rho=prior_rho_mean,
        dz=dz,
        t_total=t_total,
        f0=f0,
        use_gpu=True,
        dtype="float32",
        out_dir=map_outdir,
        seed=42
    )

    # Inversion settings
    mh_opts = {
        "niter": 100,
        "burnin": 20,
        "thin": 10,
        "proposal_fraction": 0.02,
        "auto_tune": True
    }

    map_opts = {
        "maxiter": 20,
        "disp": True
    }

    # Bayesian inversion
    inversion_kwargs = dict(
        layer_thickness=prior_layers,
        layer_vs_true=prior_vs_mean,
        layer_rho=prior_rho_mean,
        dz=dz,
        t_total=t_total,
        f0=f0,
        waveform_noise_std=0.02,
        hv_noise_std=hv_noise_std,
        use_gpu=True,
        map_opts=map_opts,
        mh_opts=mh_opts,
        out_dir=map_outdir,
        dtype="float32",
        obs_hvsr=hv_values,
        obs_freqs=hv_freqs
    )

    try:
        out = b.Bayesian_Inversion(
            **inversion_kwargs,
            hv_method="external"
        )
    except TypeError:
        out = b.Bayesian_Inversion(**inversion_kwargs)

    map_vs = out.get("map_params_vs", None)

    if map_vs is None:
        mp = out.get("map_params", None)
        if mp is not None:
            map_vs = np.asarray(mp)
        elif out.get("map_params_m", None) is not None:
            try:
                map_vs = np.exp(np.asarray(out.get("map_params_m")))
            except Exception:
                map_vs = np.asarray(out.get("map_params_m"))

    out["map_params_vs"] = map_vs

    if out.get("samples") is not None and out.get("param_in_log", False):
        out["samples_vs"] = np.exp(out["samples"])

    if out.get("map_params") is not None and out.get("param_in_log", False):
        out["map_params_vs"] = np.exp(out["map_params"])

    # Visualization
    my_hvsr_wrapper = b.Make_HVSR_Wrapper_FWD(out, param_in_log=True)

    BFWI.Plot_Inversion_Results(
        out,
        true_vs=prior_vs_mean,
        forward_hvsr_fn=my_hvsr_wrapper,
        samples_to_plot=1000,
        hv_sample_count=500
    )

    # plt.show()

    print("MAP (Vs) per-layer:", out.get("map_params_vs"))
    print("MAP (log-Vs) per-layer:", out.get("map_params_m"))
    print("Accept rate:", out.get("accept_rate"))
    print("Median per-layer Vs:", out.get("median_vs"))
    print("Keys in out:", out.keys())
    print("param_in_log:", out.get("param_in_log"))
    print("nsave (samples shape):", None if out.get("samples") is None else out["samples"].shape)
    print("MAP params:", out.get("map_params"))



if __name__ == "__main__":
    main()






















