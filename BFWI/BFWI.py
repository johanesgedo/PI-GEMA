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

Bayesian Full Waveform Inversion (BFWI)
"""


import os
import math
import time
import json
import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker
from scipy.optimize import minimize
from scipy.signal import welch
from typing import Tuple, Dict, Any, Dict, Sequence, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cupy as cp
    xp = cp
    use_cupy_global = True
except Exception:
    xp = np
    use_cupy_global = False


#====================
# Constants
#====================
PI = math.pi            # PI from math library
EPS = 1e-12             # Epsilon constant
TETHA = 1e-18           # Tetha constant
CFL = 0.45              # Courant-Fredrichs-Lewy constants
SANITIZE_INTERVAL = 50  # Sanitize frequency: how often to do per-step sanitization (reduce kernel-launch overhead)
LARGE_PENALTY = 1e30    # Helper: robust evaluation of negative log-posterior
TINY = 1e-300           # Denominator
V_CORRECTION = 1e-9     # Velocity correction
VP_SCALE = 1.73         # P-Wave velocity
N_ITER = 20000          # Number of iterations
BURN_IN = 2000          # Number of initial iterations discarded because the Markov chain is not yet stable (has not reached the target posterior distribution)
THIN = 10               # Thinning factor (subsampling) to reduce correlation between MCMC samples
PROPOSAL_FRACTION = 0.02    # Scale factor for Gaussian random walk
NUMBER_OF_SAMPLES = 100     # Number of sample plots
DZ = 0.5                # Depth direction (m)
T_TOTAL = 3.0           #
F0 = 4.0                #
WAVEFORM_NOISE = 0.02   #
HV_NOISE = 0.1          # Horizontal-to-Vertical Noise
PAD_DEPTH = 60.0        #
# Plot visualization
SAMPLES_TO_PLOT = 200   #
HV_SAMPLE_COUNT = 100   #
WAVEFORM_SAMPLE_COUNT = 50  #
THIN = 10               #

logger = logging.getLogger("Bayesian_Inversion")
logger.setLevel(logging.INFO)


def Plot_Inversion_Results(
    out,
    true_vs: Optional[Sequence[float]] = None,
    forward_hvsr_fn: Optional[callable] = None,
    forward_waveform_fn: Optional[callable] = None,
    samples_to_plot: int = SAMPLES_TO_PLOT,
    hv_sample_count: int = HV_SAMPLE_COUNT,
    waveform_sample_count: int = WAVEFORM_SAMPLE_COUNT,
    save_dir: Optional[str] = None,
    figsize_profile=(4, 8),
    verbose: bool = True,
):
    def _asarray(a):
        if a is None:
            return None
        try:
            return np.asarray(a)
        except Exception:
            return np.array(a)

    samples = _asarray(out.get("samples_vs", out.get("samples", None)))
    map_vs = _asarray(out.get("map_params_vs", out.get("map_params", None)))
    if map_vs is None and "map_params_m" in out and out.get("map_params_m") is not None:
        try:
            map_vs = np.exp(_asarray(out.get("map_params_m")))
        except Exception:
            map_vs = _asarray(out.get("map_params_m"))

    median_vs = _asarray(out.get("median_vs", None))
    lower_vs = _asarray(out.get("lower_vs", None))
    upper_vs = _asarray(out.get("upper_vs", None))

    hv_obs = _asarray(out.get("hv_obs", None))
    hv_map = _asarray(out.get("hv_map", None))
    freq_vec = _asarray(out.get("freq_vec", None))

    synth_map = _asarray(out.get("synth_map", None))
    obs_waveform = _asarray(out.get("obs_waveform", None))
    if obs_waveform is None and 'obs_surface_noisy' in out:
        obs_waveform = _asarray(out['obs_surface_noisy'])

    z = _asarray(out.get("z", None))
    layer_thickness = _asarray(out.get("layer_thickness", None))
    out_dir = save_dir if save_dir is not None else out.get("out_dir", ".")

    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    #--------------------------
    # Sanitize samples shape
    #--------------------------
    if samples is not None:
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if samples.ndim == 2 and samples.shape[0] < samples.shape[1] and samples.shape[0] > 1:
            if verbose:
                print("[plot] heuristic transpose of samples (rows < cols):", "before", samples.shape)
            samples = samples.T
            if verbose:
                print("[plot] samples.shape after transpose:", samples.shape)

    #--------------------------------------
    # Helpers: expand per-layer -> grid
    #--------------------------------------
    def Layer_to_Grid(layer_params):
        if layer_thickness is None or layer_params is None:
            return None
        layers = np.asarray(layer_thickness)
        dz_local = float(out.get("dz", 0.5))
        total_depth = float(np.sum(layers))
        nz_cells = int(np.ceil(total_depth / dz_local)) if dz_local > 0 else max(len(layer_params), 1)
        nz_cells = max(nz_cells, len(layer_params))
        grid = np.empty(nz_cells, dtype=float)
        cum = 0
        for th, val in zip(layers, layer_params):
            ncell = max(1, int(round(th / dz_local))) if dz_local > 0 else 1
            end = min(nz_cells, cum + ncell)
            grid[cum:end] = float(val)
            cum = end
        if cum < nz_cells:
            grid[cum:] = float(layer_params[-1])
        return grid


    # Diagnostics
    if verbose:
        print("[plot] samples shape:", None if samples is None else samples.shape)
        print("[plot] map_vs shape:", None if map_vs is None else np.asarray(map_vs).shape)
        print("[plot] median_vs shape:", None if median_vs is None else np.asarray(median_vs).shape)
        print("[plot] z length:", None if z is None else np.asarray(z).size)
        print("[plot] layer_thickness len:", None if layer_thickness is None else len(layer_thickness))

    # If median not present but samples exist, compute percentiles
    if median_vs is None and samples is not None and samples.size > 0:
        try:
            median_vs = np.median(samples, axis=0)
            lower_vs = np.percentile(samples, 5, axis=0)
            upper_vs = np.percentile(samples, 95, axis=0)
        except Exception:
            median_vs = None

    # Build default z if missing
    if z is None:
        if layer_thickness is not None:
            total_depth = float(np.sum(layer_thickness))
            dz = float(out.get("dz", 0.5))
            nz = int(round(total_depth / float(dz))) if dz > 0 else int(total_depth)
            nz = max(nz, 1)
            z = np.linspace(0.0, total_depth, nz, endpoint=False)
            if verbose:
                print(f"[plot] built z from layer_thickness: nz={nz}, dz={dz}")
        else:
            if median_vs is not None:
                z = np.arange(median_vs.size)
            elif map_vs is not None:
                z = np.arange(map_vs.size)

    # Infer whether samples are per-layer or per-cell
    nlayer = len(layer_thickness) if layer_thickness is not None else None
    is_layer_samples = False
    if samples is not None:
        ncols = samples.shape[1]
        if nlayer is not None and ncols == nlayer:
            is_layer_samples = True
        elif z is not None and ncols == z.size:
            is_layer_samples = False
        else:
            is_layer_samples = (ncols <= max(12, (nlayer or ncols)))

    if verbose:
        print(f"[plot] inferred is_layer_samples = {is_layer_samples}")

    #------------------------------
    # 1) Profile ensemble plot
    #------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize_profile, constrained_layout=True)
    if samples is not None and samples.size > 0:
        nplot = min(samples_to_plot, samples.shape[0])
        idx = np.linspace(0, samples.shape[0] - 1, nplot).astype(int)
        for ii in idx:
            samp = np.asarray(samples[ii])
            if is_layer_samples:
                try:
                    grid = Layer_to_Grid(samp)
                    depth = np.linspace(0.0, np.sum(layer_thickness), grid.size)
                    ax.step(grid, depth, where='post', color='C0', alpha=0.03, linewidth=0.8)
                except Exception as e:
                    if verbose:
                        print(f"[plot][WARN] cannot expand layer-sample idx={ii}: {e}")
            else:
                if z is None:
                    if verbose:
                        print(f"[plot][WARN] z is None but samples look per-cell; skipping sample idx={ii}")
                    continue
                if samp.size == z.size:
                    ax.step(samp, z, where='post', color='C0', alpha=0.03, linewidth=0.8)
                else:
                    if nlayer is not None and samp.size == nlayer:
                        try:
                            grid = Layer_to_Grid(samp)
                            depth = np.linspace(0.0, np.sum(layer_thickness), grid.size)
                            ax.step(grid, depth, where='post', color='C0', alpha=0.03, linewidth=0.8)
                        except Exception:
                            if verbose:
                                print(f"[plot][WARN] sample {ii} length mismatch; skipped")
                    else:
                        if verbose:
                            print(f"[plot][WARN] sample {ii} length {samp.size} != depth {z.size}; skipped")

    # Credible band + median
    try:
        if lower_vs is not None and upper_vs is not None:
            if is_layer_samples:
                med_grid = Layer_to_Grid(median_vs)
                p5_grid = Layer_to_Grid(lower_vs)
                p95_grid = Layer_to_Grid(upper_vs)
                depth = np.linspace(0.0, np.sum(layer_thickness), med_grid.size)
                ax.fill_betweenx(depth, p5_grid, p95_grid, step='post', color='C1', alpha=0.25, label='Credible Interval')
                ax.step(med_grid, depth, where='post', color='C1', linewidth=2.0, label='median')
            else:
                if median_vs is not None and z is not None and median_vs.size == z.size:
                    ax.fill_betweenx(z, lower_vs, upper_vs, color='C1', alpha=0.25, label='Credible Interval')
                    ax.step(median_vs, z, where='post', color='C1', linewidth=2.0, label='median')
                else:
                    if nlayer is not None and median_vs is not None and median_vs.size == nlayer:
                        med_grid = Layer_to_Grid(median_vs)
                        p5_grid = Layer_to_Grid(lower_vs)
                        p95_grid = Layer_to_Grid(upper_vs)
                        depth = np.linspace(0.0, np.sum(layer_thickness), med_grid.size)
                        ax.fill_betweenx(depth, p5_grid, p95_grid, step='post', color='C1', alpha=0.25, label='Credible Interval')
                        ax.step(med_grid, depth, where='post', color='C1', linewidth=2.0, label='median')
    except Exception as e:
        if verbose:
            print("[plot][WARN] credible band plotting error:", e)

    # MAP line
    if map_vs is not None:
        try:
            map_vs_arr = np.asarray(map_vs)
            if is_layer_samples and nlayer is not None and map_vs_arr.size == nlayer:
                map_grid = Layer_to_Grid(map_vs_arr)
                depth = np.linspace(0.0, np.sum(layer_thickness), map_grid.size)
                ax.step(map_grid, depth, where='post', color='k', linewidth=1.5, label='MAP')
            elif z is not None and map_vs_arr.size == z.size:
                ax.step(map_vs_arr, z, where='post', color='k', linewidth=1.5, label='MAP')
            elif nlayer is not None and map_vs_arr.size == nlayer:
                map_grid = Layer_to_Grid(map_vs_arr)
                depth = np.linspace(0.0, np.sum(layer_thickness), map_grid.size)
                ax.step(map_grid, depth, where='post', color='k', linewidth=1.5, label='MAP')
            else:
                if verbose:
                    print("[plot][WARN] MAP params length mismatch; skipping MAP line")
        except Exception as e:
            if verbose:
                print("[plot][WARN] MAP plotting failed:", e)

    # True markers
    if true_vs is not None:
        try:
            if layer_thickness is not None:
                layer_tops = np.concatenate([[0.0], np.cumsum(layer_thickness)[:-1]])
                ax.scatter(true_vs, layer_tops, marker='o', c='red', s=30, label='true (layer)')
            else:
                # Fallback: plot as line
                tv = np.asarray(true_vs)
                if tv.ndim == 1:
                    ax.plot(tv, np.linspace(0, 1, tv.size), '--', color='red', label='true')
        except Exception:
            if verbose:
                print("[plot][WARN] failed to plot true_vs")

    ax.invert_yaxis()
    ax.set_xlabel('Vs (m/s)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.legend()
    plt.title('Posterior Ensemble and MAP (Vs profile)')
    fpath = os.path.join(diag_dir, "Vs_Profile_Ensemble.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=200)
    if verbose:
        print("[plot] saved", fpath)
    plt.close(fig)

    #-------------------------------------
    # 2) Per-layer summary (error bars)
    #-------------------------------------
    try:
        if samples is not None and samples.size > 0:
            med_layer = np.median(samples, axis=0)
            p5_layer = np.percentile(samples, 5, axis=0)
            p95_layer = np.percentile(samples, 95, axis=0)
        else:
            med_layer = median_vs if median_vs is not None else map_vs
            p5_layer = lower_vs
            p95_layer = upper_vs

        if med_layer is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            x = np.arange(med_layer.size)
            yerr_low = (med_layer - p5_layer) if (p5_layer is not None and np.asarray(p5_layer).size == med_layer.size) else np.zeros_like(med_layer)
            yerr_high = (p95_layer - med_layer) if (p95_layer is not None and np.asarray(p95_layer).size == med_layer.size) else np.zeros_like(med_layer)
            ax.errorbar(x, med_layer, yerr=[yerr_low, yerr_high], fmt='o', capsize=4, label='Posterior (Credible Interval)')
            if true_vs is not None:
                tv = np.asarray(true_vs)
                if tv.size == med_layer.size:
                    ax.plot(x, tv, 'k--', label='true Vs (layer)')
            if map_vs is not None:
                mv = np.asarray(map_vs)
                if mv.size == med_layer.size:
                    ax.plot(x, mv, 'C1x', label='MAP', markersize=8)
            ax.set_xlabel('Layer index')
            ax.set_ylabel('Vs (m/s)')
            ax.grid(True)
            ax.legend()
            fpath = os.path.join(diag_dir, "Layer_Posteriors.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)
    except Exception as e:
        if verbose:
            print("[plot][WARN] layer posterior plotting failed:", e)

    #-------------------------------------------------------
    # 3) HVSR observation vs MAP and posterior envelope
    #-------------------------------------------------------
    if hv_obs is not None and freq_vec is not None:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.semilogy(freq_vec, hv_obs, label='HVSR Observation', linestyle='-', color='k')
            if hv_map is not None and np.asarray(hv_map).size == np.asarray(freq_vec).size:
                ax.semilogy(freq_vec, hv_map, label='HVSR Maximum a Posteriori', linestyle='--', color='C1')
            if forward_hvsr_fn is not None and samples is not None and samples.shape[0] > 0:
                nsamp = min(hv_sample_count, samples.shape[0])
                ids = np.linspace(0, samples.shape[0] - 1, nsamp).astype(int)
                hv_stack = []
                for ii in ids:
                    try:
                        params = samples[ii]
                        freqs_s, hv_s = forward_hvsr_fn(params)
                        hv_s = np.asarray(hv_s)
                        if not np.allclose(freqs_s, freq_vec):
                            hv_s = np.interp(freq_vec, freqs_s, hv_s, left=np.nan, right=np.nan)
                        hv_stack.append(hv_s)
                    except Exception:
                        continue
                if len(hv_stack) > 0:
                    hv_stack = np.vstack(hv_stack)
                    p5 = np.nanpercentile(hv_stack, 5, axis=0)
                    p95 = np.nanpercentile(hv_stack, 95, axis=0)
                    ax.fill_between(freq_vec, p5, p95, alpha=0.25, label='Posterior HVSR Credible Interval')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('HVSR')
            ax.grid(True)
            ax.legend()
            fpath = os.path.join(diag_dir, "HVSR_Observation_Vs_MAP.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print("[plot][WARN] HVSR plotting failed:", e)

    #--------------------------------------------------
    # 4) Waveform Posterior Predictive Check (PPC)
    #--------------------------------------------------
    try:
        obs_w = obs_waveform if obs_waveform is not None else out.get("obs_dev") or out.get("obs_surface_noisy")
        if obs_w is not None and (synth_map is not None or forward_waveform_fn is not None) and samples is not None and samples.shape[0] > 0:
            T = len(obs_w)
            tvec = np.arange(T) * out.get("dt", 1.0 / (len(obs_w) if len(obs_w) > 0 else 1.0))
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.plot(tvec, obs_w, color='k', label='obs', linewidth=1.2)
            nsamp = min(waveform_sample_count, samples.shape[0])
            ids = np.linspace(0, samples.shape[0] - 1, nsamp).astype(int)
            stack = []
            for ii in ids:
                try:
                    params = samples[ii]
                    if forward_waveform_fn is not None:
                        syn = np.asarray(forward_waveform_fn(params))
                    else:
                        syn = np.asarray(synth_map)
                    if syn.size == T:
                        ax.plot(tvec, syn, alpha=0.06, color='C0')
                        stack.append(syn)
                except Exception:
                    continue
            if len(stack) > 0:
                stack = np.vstack(stack)
                mean_syn = np.mean(stack, axis=0)
                p5 = np.percentile(stack, 5, axis=0)
                p95 = np.percentile(stack, 95, axis=0)
                ax.plot(tvec, mean_syn, color='C1', label='posterior mean synth')
                ax.fill_between(tvec, p5, p95, color='C1', alpha=0.25, label='Credible Interval Prediction')
            if synth_map is not None:
                if np.asarray(synth_map).size == T:
                    ax.plot(tvec, synth_map, color='C3', linestyle='--', label='MAP synth')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('surface')
            ax.legend()
            ax.grid(True)
            fpath = os.path.join(diag_dir, "waveform_ppc.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)

            if len(stack) > 0:
                rmses = np.sqrt(np.mean((stack - obs_w[None, :]) ** 2, axis=1))
                fig, ax = plt.subplots(1, 1, figsize=(5, 3))
                ax.hist(rmses, bins=30, alpha=0.8)
                if np.asarray(synth_map).size == T:
                    ax.axvline(np.sqrt(np.mean((synth_map - obs_w) ** 2)), color='C3', linestyle='--', label='MAP RMSE')
                ax.set_xlabel('RMSE (synthetic vs obs)')
                ax.set_ylabel('count')
                ax.grid(True)
                fpath = os.path.join(diag_dir, "rmse_hist.png")
                plt.tight_layout()
                plt.savefig(fpath, dpi=200)
                if verbose:
                    print("[plot] saved", fpath)
                plt.close(fig)
                if verbose:
                    print(f"[metric] MAP RMSE: {np.sqrt(np.mean((synth_map - obs_w) ** 2)):.4g}, sample RMSE mean: {np.mean(rmses):.4g}")
    except Exception as e:
        if verbose:
            print("[plot][WARN] waveform PPC failed:", e)

    #------------------------------------
    # 5) Trace plots and marginals
    #------------------------------------
    if samples is not None and samples.size > 0:
        try:
            nsamp, ndim = samples.shape
            max_plots = min(ndim, 12)
            ncols = 3
            nrows = int(np.ceil(max_plots / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
            axs = np.atleast_2d(axs)
            for i in range(max_plots):
                r = i // ncols; c = i % ncols
                ax1 = axs[r, c]
                ax1.plot(samples[:, i], alpha=0.7, linewidth=0.6)
                ax1.set_title(f"parameters {i}")
                ax1.grid(True, linestyle=':', linewidth=0.4)
            for j in range(max_plots, nrows * ncols):
                r = j // ncols; c = j % ncols
                axs[r, c].axis('off')
            fpath = os.path.join(diag_dir, "trace_plots.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)

            max_hist = min(ndim, 8)
            fig, axs = plt.subplots(2, 4, figsize=(16, 6))
            axs = axs.ravel()
            for i in range(max_hist):
                axs[i].hist(samples[:, i], bins=50, density=True, alpha=0.7)
                axs[i].axvline(np.median(samples[:, i]), color='C1', linestyle='--', label='median')
                if map_vs is not None and i < len(np.asarray(map_vs)):
                    try:
                        axs[i].axvline(np.asarray(map_vs)[i], color='k', label='MAP')
                    except Exception:
                        pass
                axs[i].set_title(f"parameters {i}")
                axs[i].legend()
                axs[i].grid(True)
            for j in range(max_hist, len(axs)):
                axs[j].axis('off')
            fpath = os.path.join(diag_dir, "marginals.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print("[plot][WARN] trace/marginal plotting failed:", e)

    #--------------------------
    # 6) Pairwise scatter
    #--------------------------
    if samples is not None and samples.size > 0:
        try:
            ndim = samples.shape[1]
            k = min(6, ndim)
            fig, axs = plt.subplots(k, k, figsize=(3 * k, 3 * k))
            for i in range(k):
                for j in range(k):
                    ax = axs[i, j]
                    if i == j:
                        ax.hist(samples[:, i], bins=40, color='C0', alpha=0.8)
                    else:
                        ax.scatter(samples[:, j], samples[:, i], s=3, alpha=0.3)
                    if i == k - 1:
                        ax.set_xlabel(f"{j}")
                    if j == 0:
                        ax.set_ylabel(f"{i}")
            fpath = os.path.join(diag_dir, "pairwise_scatter.png")
            plt.tight_layout()
            plt.savefig(fpath, dpi=200)
            if verbose:
                print("[plot] saved", fpath)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print("[plot][WARN] pairwise scatter plotting failed:", e)

    #----------------------------------------------
    # 7) Numeric summary when true values given
    #----------------------------------------------
    if true_vs is not None:
        try:
            true_vs = np.asarray(true_vs)
            if samples is not None and samples.size > 0:
                med = np.median(samples, axis=0)
                p5 = np.percentile(samples, 5, axis=0)
                p95 = np.percentile(samples, 95, axis=0)
                bias = med - true_vs
                rmse = np.sqrt(np.mean((med - true_vs) ** 2))
                coverage = np.mean((true_vs >= p5) & (true_vs <= p95))
                print("[summary] per-layer RMSE (median vs true):", rmse)
                print("[summary] average bias:", np.mean(bias))
                print("[summary] coverage (5-95%):", coverage)
                summary_txt = os.path.join(diag_dir, "summary.txt")
                with open(summary_txt, "w") as f:
                    f.write(f"RMSE_median_vs_true: {rmse}\n")
                    f.write(f"mean_bias: {np.mean(bias)}\n")
                    f.write(f"coverage_5_95: {coverage}\n")
                    f.write("per_layer_true median p5 p95 bias\n")
                    for i in range(true_vs.size):
                        f.write(f"{i} {true_vs[i]} {med[i]} {p5[i]} {p95[i]} {bias[i]}\n")
                if verbose:
                    print("[plot] saved numeric summary ->", summary_txt)
            else:
                if verbose:
                    print("[summary] true_vs provided but no samples to compute coverage")
        except Exception as e:
            if verbose:
                print("[plot][WARN] numeric summary failed:", e)
    if verbose:
        print("[plotting] diagnostics saved to:", diag_dir)




class Bayesian_Full_Waveform_Inversion:

    def __init__(
        self,
        layer_thickness: Sequence[float],
        layer_vs_true: Sequence[float],
        layer_rho: Sequence[float],
        dz: float = DZ,
        t_total: float = T_TOTAL,
        f0: float = F0,
        use_gpu: Optional[bool] = None,
        dtype: str = "float32",
        out_dir: str = "./Bayesian_FWI_out",
        seed: Optional[int] = None,
        sanitize_interval: int = SANITIZE_INTERVAL,
    ):
        # Validation
        if dz <= 0:
            raise ValueError("dz must be > 0")
        self.layer_thickness = np.asarray(layer_thickness, dtype=float)
        self.layer_vs_true = np.asarray(layer_vs_true, dtype=float)
        self.layer_rho = np.asarray(layer_rho, dtype=float)
        self.dz = float(dz)
        self.t_total = float(t_total)
        self.f0 = float(f0)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # Backend
        try:
            CUPY_AVAILABLE = bool(cp is not None)
        except Exception:
            CUPY_AVAILABLE = False

        self.use_gpu = bool(use_gpu) if use_gpu is not None else CUPY_AVAILABLE
        self.xp = cp if (self.use_gpu and CUPY_AVAILABLE) else np
        self.is_cupy = (self.xp is cp)

        # dtype
        self.float_dtype = self.xp.float32 if dtype == "float32" else self.xp.float64

        # Constants
        self.PI = PI
        self.EPS = EPS
        self.LARGE_PENALTY = LARGE_PENALTY

        # Build grid
        total_layer_depth = float(np.sum(self.layer_thickness))
        pad_depth = PAD_DEPTH
        model_depth = total_layer_depth + pad_depth
        self.nz = int(math.ceil(model_depth / self.dz)) + 1
        self.z = np.linspace(0.0, (self.nz - 1) * self.dz, self.nz)

        # Build Layer slices in cells (host)
        self.layer_cell_counts = [max(1, int(round(th / self.dz))) for th in self.layer_thickness]
        self.layer_slices = []
        cur = 0
        for c in self.layer_cell_counts:
            self.layer_slices.append((cur, min(cur + c, self.nz)))
            cur += c

        # Secondary Wave Velocity (Vs) true grid (host)
        halfspace_vs = float(self.layer_vs_true[-1])
        self.vs_grid_true = np.ones(self.nz, dtype=float) * halfspace_vs
        cur = 0
        for p, th in zip(self.layer_vs_true, self.layer_thickness):
            ncell = int(round(th / self.dz))
            if ncell <= 0:
                continue
            end = min(cur + ncell, self.nz)
            self.vs_grid_true[cur:end] = float(p)
            cur = end

        # Rho grid
        self.rho_grid = np.ones(self.nz, dtype=float) * float(self.layer_rho[-1])
        cur = 0
        for rho_val, th in zip(self.layer_rho, self.layer_thickness):
            ncell = int(round(th / self.dz))
            if ncell <= 0:
                continue
            end = min(cur + ncell, self.nz)
            self.rho_grid[cur:end] = float(rho_val)
            cur = end

        # Time step
        vs_max = np.max(self.vs_grid_true)
        self.dt = float(CFL * self.dz / (vs_max + V_CORRECTION))
        self.nt = int(math.ceil(self.t_total / self.dt))
        self.sanitize_interval = int(sanitize_interval)

        # Random Number Generators (RNG)
        self.seed = seed
        if self.is_cupy:
            try:
                self.rng = self.xp.random.default_rng(self.seed)
            except Exception:
                self.rng = None
        else:
            self.rng = np.random.default_rng(self.seed)

        # Preallocate device arrays
        self.src_i = int(max(1, round(0.5 * self.layer_thickness[0] / self.dz)))
        self.nfft = 1 << (int(np.ceil(np.log2(self.nt))) + 1)
        self.xp_vs = self.xp.asarray(self.vs_grid_true, dtype=self.float_dtype)
        self.xp_rho = self.xp.asarray(self.rho_grid, dtype=self.float_dtype)

        # FDTD buffers
        self.v = self.xp.zeros(self.nz, dtype=self.float_dtype)
        self.tau = self.xp.zeros(max(1, self.nz - 1), dtype=self.float_dtype)
        self.u = self.xp.zeros(self.nz, dtype=self.float_dtype)
        self.mu_half = self.xp.zeros(max(1, self.nz - 1), dtype=self.float_dtype)
        self.rho_inv = 1.0 / (self.xp_rho + self.EPS)

        # Sponge
        self.sponge = self.xp.zeros(self.nz, dtype=self.float_dtype)
        sponge_cells = max(20, min(80, self.nz // 8))
        if sponge_cells > 0:
            ramp = self.xp.linspace(0.0, 1.0, sponge_cells, dtype=self.float_dtype)
            self.sponge[-sponge_cells:] = 0.02 * (ramp * ramp)

        logger.info(f"Initialized Bayesian_FWI: use_gpu={self.use_gpu}, nz={self.nz}, nt={self.nt}, dt={self.dt:.3e}")


    def Precompute_Linear_Interpolation_Weights(
        self,
        x_src,
        x_target,
        xp,
        eps=EPS
    ):
        x_src = xp.asarray(x_src, dtype=xp.float64)
        x_target = xp.asarray(x_target, dtype=xp.float64)
        if x_src.size < 2:
            raise ValueError("x_src must contain at least 2 points for linear interpolation.")
        idx = xp.searchsorted(x_src, x_target, side="right") - 1
        idx = xp.clip(idx, 0, x_src.size - 2).astype(xp.int64)
        x0 = x_src[idx]
        x1 = x_src[idx + 1]
        denom = x1 - x0
        denom = xp.where(xp.abs(denom) > eps, denom, xp.asarray(eps, dtype=xp.float64))
        alpha = (x_target - x0) / denom
        alpha = xp.clip(alpha, 0.0, 1.0).astype(xp.float64)
        return idx, alpha


    def Interpolation_Linear_Precomputed(
        self,
        y_src,
        idx,
        alpha,
        xp,
        dtype=None
    ):
        y_src = xp.asarray(y_src, dtype=xp.float64)
        out = (1.0 - alpha) * y_src[idx] + alpha * y_src[idx+1]
        if dtype is None:
            return out
        return out.astype(dtype, copy=False)


    #=============================================
    # Forward SH Finite Difference Time Domain
    #=============================================
    def Forward_SH_FDTD(
        self,
        src_time: Optional["xp.ndarray"] = None,
        nt: Optional[int] = None,
        src_i: Optional[int] = None,
        vs: Optional["xp.ndarray"] = None,
        rho: Optional["xp.ndarray"] = None,
        dz: Optional[float] = None,
        dt: Optional[float] = None,
        inplace: bool = False,
        return_state: bool = False,
        sanitize_interval: Optional[int] = None,
        xp: Optional[object] = None,
    ) -> "xp.ndarray":

        # Backend
        xp = xp if xp is not None else getattr(self, "xp", __import__("numpy"))
        float_dtype = getattr(self, "float_dtype", xp.float32)

        # Scalar parameters
        nt = int(nt) if nt is not None else int(getattr(self, "nt", None))
        if nt is None or nt <= 0:
            raise ValueError("nt must be provided (positive integer) either as argument or self.nt")

        dz = float(dz) if dz is not None else getattr(self, "dz", None)
        if dz is None or abs(dz) < EPS:
            raise ValueError("dz must be provided (positive scalar) either as argument or self.dz")

        dt_x = float(dt) if dt is not None else getattr(self, "dt", None)
        if dt_x is None or dt_x <= 0:
            raise ValueError("dt must be provided (positive scalar) either as argument or self.dt")

        # Grid of secondary wave velocity (Vs) and rho (xp arrays)
        def as_xp_array(obj):
            if obj is None:
                return None
            return xp.asarray(obj, dtype=float_dtype)

        def Pick_First_Array(*candidates):
            for c in candidates:
                if c is not None:
                    return as_xp_array(c)
            return None

        vs_dev = Pick_First_Array(vs, getattr(self, "xp_vs", None), getattr(self, "vs", None))
        rho_dev = Pick_First_Array(rho, getattr(self, "xp_rho", None), getattr(self, "rho", None))

        # vs_dev = as_xp_array(vs) or as_xp_array(getattr(self, "xp_vs", None)) or as_xp_array(getattr(self, "vs", None))
        # rho_dev = as_xp_array(rho) or as_xp_array(getattr(self, "xp_rho", None)) or as_xp_array(getattr(self, "rho", None))

        if vs_dev is None or rho_dev is None:
            raise ValueError ("vs and rho must be available either via args or instance attributes (vs/xp_vs, rho/xp_rho).")

        nz = int(getattr(self, "nz", vs_dev.size)) if getattr(self, "nz", None) is not None else int(vs_dev.size)
        if nz is None or nz <= 0:
            nz = int(vs_dev.size)

        if int(vs_dev.size) != nz or int(rho_dev.size) != nz:
            raise ValueError("vs/rho length mismatch with nz")

        if src_time is None:
            src_time_dev = xp.zeros(nt, dtype=float_dtype)
        else:
            src_time_dev = xp.asarray(src_time, dtype=float_dtype)
            if src_time_dev.size < nt:
                pad = xp.zeros(nt - int(src_time_dev.size), dtype=float_dtype)
                src_time_dev = xp.concatenate([src_time_dev, pad])
            elif src_time_dev.size > nt:
                src_time_dev = src_time_dev[:nt]

        # Source index
        if src_i is None:
            src_idx = int(getattr(self, "src_i", 0))
        else:
            src_idx = int(src_i)
        if not (0 <= src_idx < nz):
            raise ValueError(f"src_i {src_idx} out of range (0..{nz - 1})")

        inv_dz = 1.0 / (dz if abs(dz) > EPS else EPS)

        # Allocate working buffers
        if inplace:
            v_dev = getattr(self, "v", None)
            tau_dev = getattr(self, "tau", None)
            u_dev = getattr(self, "u", None)
            mu_half_dev = getattr(self, "mu_half", None)
            rho_inv_dev = getattr(self, "rho_inv", None)
            sponge_dev = getattr(self, "sponge", None)
        else:
            v_dev = xp.array(getattr(self, "v", xp.zeros(nz, dtype=float_dtype)), dtype=float_dtype)
            tau_dev = xp.array(getattr(self, "tau", xp.zeros(max(1, nz - 1), dtype=float_dtype)), dtype=float_dtype)
            u_dev = xp.array(getattr(self, "u", xp.zeros(nz, dtype=float_dtype)), dtype=float_dtype)
            mu_half_default = 0.5 * (rho_dev[:-1] * (vs_dev[:-1] ** 2) + rho_dev[1:] * (vs_dev[1:] ** 2))
            mu_half_dev = xp.array(getattr(self, "mu_half", mu_half_default), dtype=float_dtype)
            rho_inv_dev = xp.array(getattr(self, "rho_inv", 1.0 / (rho_dev + EPS)), dtype=float_dtype)
            sponge_dev = xp.array(getattr(self, "sponge", xp.zeros(nz, dtype=float_dtype)), dtype=float_dtype)

        # Ensure arrays exist and have correct dtype
        v_dev = xp.asarray(v_dev, dtype=float_dtype)
        tau_dev = xp.asarray(tau_dev, dtype=float_dtype)
        u_dev = xp.asarray(u_dev, dtype=float_dtype)
        mu_half_dev = xp.asarray(mu_half_dev, dtype=float_dtype)
        rho_inv_dev = xp.asarray(rho_inv_dev, dtype=float_dtype)
        sponge_dev = xp.asarray(sponge_dev, dtype=float_dtype)

        # Sanitize arrays (Nan/Inf)
        v_dev[:] = xp.nan_to_num(v_dev, nan=0.0, posinf=1e12, neginf=-1e12)
        tau_dev[:] = xp.nan_to_num(tau_dev, nan=0.0, posinf=1e12, neginf=-1e12)
        u_dev[:] = xp.nan_to_num(u_dev, nan=0.0, posinf=1e12, neginf=-1e12)
        mu_half_dev[:] = xp.nan_to_num(mu_half_dev, nan=0.0, posinf=1e12, neginf=0.0)
        rho_inv_dev[:] = xp.nan_to_num(rho_inv_dev, nan=0.0, posinf=1e12, neginf=0.0)
        sponge_dev[:] = xp.nan_to_num(sponge_dev, nan=0.0, posinf=1.0, neginf=0.0)

        # Prepare outputs and temporaries
        surface_dev = xp.zeros(nt, dtype=float_dtype)

        dv = xp.empty(max(0, nz - 1), dtype=float_dtype)
        tmp_half = xp.empty(max(0, nz - 1), dtype=float_dtype)
        dtau_diff = xp.empty(max(0, nz - 2), dtype=float_dtype) if nz > 2 else xp.empty(0, dtype=float_dtype)
        u0 = xp.empty(nz, dtype=float_dtype)
        tmp_rho_u = xp.empty(nz, dtype=float_dtype)

        # Damping precompute
        damping = xp.maximum(0.0, 1.0 - sponge_dev * dt_x)

        # Check any non-finite
        # def any_nonfinite(arr):
        #     try:
        #         val = xp.any(~xp.isfinite(arr))
        #         return bool(val.item()) if hasattr(val, "item") else bool(val)
        #     except Exception:
        #         return bool(xp.any(~xp.isfinite(xp.asarray(arr))))

        # Sanitize interval
        sanitize_interval = int(sanitize_interval) if sanitize_interval is not None else int(getattr(self, "sanitize_interval", 50))

        # Precompute host-side source series once to avoid per-step device scalar sync
        # src_time_host = np.asarray(src_time_dev.get() if hasattr(src_time_dev, "get") else src_time_dev, dtype=np.float64)
        # src_rho_inv = float(rho_inv_dev[src_idx].item() if hasattr(rho_inv_dev[src_idx], "item") else rho_inv_dev[src_idx])
        src_time_arr = src_time if src_time is not None else src_time_dev
        # src_rho_inv = float(np.asarray(rho_inv_dev[src_idx]).item())
        src_rho_inv = rho_inv_dev[src_idx].item()

        # Time loop
        for it in range(int(nt)):
            if nz > 1:
                xp.subtract(v_dev[1:], v_dev[:-1], out=dv)
                if inv_dz != 1.0:
                    xp.multiply(dv, inv_dz, out=dv)

                xp.multiply(mu_half_dev, dv, out=tmp_half)
                if dt_x != 1.0:
                    xp.multiply(tmp_half, dt_x, out=tmp_half)
                xp.add(tau_dev, tmp_half, out=tau_dev)

                if nz > 2:
                    xp.subtract(tau_dev[1:], tau_dev[:-1], out=dtau_diff)
                    if inv_dz != 1.0:
                        xp.multiply(dtau_diff, inv_dz, out=dtau_diff)
                    u0[1:-1] = dtau_diff
                else:
                    u0[:] = 0.0

                # Boundaries
                u0[0] = (tau_dev[0] * inv_dz) if nz > 1 else 0.0
                u0[-1] = (-tau_dev[-1] * inv_dz) if nz > 1 else 0.0

            else:
                u0[:] = 0.0

            # Damping on velocity
            xp.multiply(v_dev, damping, out=v_dev)
            xp.multiply(rho_inv_dev, u0, out=tmp_rho_u)
            if dt_x != 1.0:
                xp.multiply(tmp_rho_u, dt_x, out=tmp_rho_u)
            xp.add(v_dev, tmp_rho_u, out=v_dev)

            # Inject source (body force)
            # src_amp = float(src_time_dev[it]) if it < src_time_dev.size else 0.0
            # src_amp = src_time_host[it] if it < src_time_host.size else 0.0
            src_amp = src_time_arr[it].item()
            if src_amp != 0.0:
                incr = dt_x * src_amp * float(rho_inv_dev[src_idx])
                v_dev[src_idx] = v_dev[src_idx] + incr

            # Integrate displacement
            xp.multiply(v_dev, dt_x, out=tmp_rho_u)
            xp.add(u_dev, tmp_rho_u, out=u_dev)

            # Record surface at node 0
            surface_dev[it] = u_dev[0]

            # Periodic sanitize
            # if sanitize_interval and (it % sanitize_interval == 0):
            if sanitize_interval and (it % sanitize_interval == 0) and it > 0:
                # if any_nonfinite(v_dev):
                    v_dev[:] = xp.nan_to_num(v_dev, nan=0.0, posinf=1e12, neginf=-1e12)
                # if any_nonfinite(tau_dev):
                    tau_dev[:] = xp.nan_to_num(tau_dev, nan=0.0, posinf=1e12, neginf=-1e12)
                # if any_nonfinite(u_dev):
                    u_dev[:] = xp.nan_to_num(u_dev, nan=0.0, posinf=1e12, neginf=-1e12)

        # Final sanitize
        surface_dev = xp.nan_to_num(surface_dev, nan=0.0, posinf=1e12, neginf=-1e12)

        # Write-back if inplace
        if inplace:
            self.v = v_dev
            self.tau = tau_dev
            self.u = u_dev

        # Return state if requested
        if return_state:
            return surface_dev, v_dev, tau_dev, u_dev

        return surface_dev


    #==============================
    # Vertical Impedance Proxy
    #==============================
    """
    def Vertical_Impedance_Proxy(
        self,
        freqs,
        layer_thickness,
        vs_layers,
        rho_layers,
        vp_scale=VP_SCALE,
        float_dtype=None,
        complex_dtype=None,
        eps=None,
        to_numpy: bool = False,
    ):
        # Backend
        xp = self.xp

        if float_dtype is None:
            float_dtype = getattr(self, "float_dtype", xp.float32)
        if complex_dtype is None:
            if getattr(self, "complex_dtype", None) is not None:
                complex_dtype = self.complex_dtype
            else:
                complex_dtype = xp.complex64 if xp is getattr(__import__("cupy"), "cupy", None) else np.complex128
        if eps is None:
            eps = getattr(self, "EPS", EPS)

        freqs_x = xp.asarray(freqs, dtype=float_dtype)
        vs_arr = xp.asarray(vs_layers, dtype=float_dtype)
        rho_arr = xp.asarray(rho_layers, dtype=float_dtype)
        h_arr = xp.asarray(layer_thickness, dtype=float_dtype)

        if freqs_x.size == 0 or vs_arr.size == 0:
            out = xp.empty_like(freqs_x, dtype=float_dtype)
            return out.get() if to_numpy else out

        omega = 2.0 * float(PI) * freqs_x

        vp = xp.asarray(vp_scale, dtype=float_dtype) * vs_arr
        vp_full = xp.concatenate([vp, vp[-1:]])
        rho_full = xp.concatenate([rho_arr, rho_arr[-1:]])
        h_full = xp.concatenate([h_arr, xp.asarray([0.0], dtype=float_dtype)])

        nlayer = int(vs_arr.size)
        nf = int(freqs_x.size)

        T00 = xp.ones(nf, dtype=complex_dtype)
        T01 = xp.zeros(nf, dtype=complex_dtype)
        T10 = xp.zeros(nf, dtype=complex_dtype)
        T11 = xp.ones(nf, dtype=complex_dtype)

        Zb = rho_full[-1] * vp_full[-1]
        Zb_c = (Zb + 0j).astype(complex_dtype)

        eps_c = complex(eps, 0.0)

        # Loop over layers
        for j in range(nlayer):
            vpj = vp_full[j]
            rhoj = rho_full[j]
            hj = h_full[j]

            k = omega / (vpj + eps)
            kh = k * hj

            cos_kh = xp.cos(kh)
            sin_kh = xp.sin(kh)

            Zj = rhoj * vpj

            M00 = cos_kh.astype(complex_dtype)

            denom_local = (1j * Zj) + eps_c
            M01 = (sin_kh.astype(complex_dtype) / denom_local)
            M10 = ((1j * Zj) * sin_kh).astype(complex_dtype)
            M11 = M00

            nT00 = T00 * M00 + T01 * M10
            nT01 = T00 * M01 + T01 * M11
            nT10 = T10 * M00 + T11 * M10
            nT11 = T10 * M01 + T11 * M11

            T00, T01, T10, T11 = nT00, nT01, nT10, nT11

        # Compute denominator
        denom = T00 + T01 * Zb_c
        denom_abs = xp.abs(denom)
        denom = xp.where(denom_abs < eps, denom + eps_c, denom)

        # Surface impedance and admittance (device)
        Zsurf = (T10 + T11 * Zb_c) / (denom + eps_c)
        adm = 1.0 / (Zsurf + eps_c)

        # Imaginary part -> Im{Gzz}
        ImGzz = xp.imag(adm)

        # Sanitize numeric issues
        finite_mask = xp.isfinite(ImGzz)
        ImGzz = xp.where(finite_mask, ImGzz, xp.asarray(0.0, dtype=float_dtype))
        ImGzz = xp.maximum(ImGzz, xp.asarray(0.0, dtype=float_dtype))

        # Floor to avoid downstream divide-by-zero
        floor_val = xp.asarray(1e-16, dtype=float_dtype)
        ImGzz = xp.clip(ImGzz, a_min=floor_val, a_max=None)

        # Mask invalid frequency inputs (NaN or negative)
        invalid_mask = ~xp.isfinite(freqs_x) | (freqs_x < 0.0)
        if xp.any(invalid_mask):
            ImGzz = xp.where(invalid_mask, floor_val, ImGzz)

        # Final cast and return
        out = ImGzz.astype(float_dtype, copy=False)
        if to_numpy:
            return xp.asnumpy(out) if hasattr(xp, "asnumpy") else np.asarray(out)
        return out
    """
    

    def Vertical_Impedance_Proxy(
        self,
        freqs,
        layer_thickness,
        vs_layers,
        rho_layers,
        vp_scale=VP_SCALE,
        float_dtype=None,
        complex_dtype=None,
        eps=None,
        to_numpy: bool = False,
    ):
        xp = self.xp

        if float_dtype is None:
            float_dtype = getattr(self, "float_dtype", xp.float32)

        if complex_dtype is None:
            if float_dtype == xp.float32:
                complex_dtype = xp.complex64
            else:
                complex_dtype = xp.complex128

        if eps is None:
            eps = getattr(self, "EPS", EPS)

        eps_f = float(eps)

        freqs_x = xp.asarray(freqs, dtype=float_dtype).ravel()
        vs_arr = xp.asarray(vs_layers, dtype=float_dtype).ravel()
        rho_arr = xp.asarray(rho_layers, dtype=float_dtype).ravel()
        h_arr = xp.asarray(layer_thickness, dtype=float_dtype).ravel()

        if freqs_x.size == 0 or vs_arr.size == 0:
            out = xp.empty_like(freqs_x, dtype=float_dtype)
            return xp.asnumpy(out) if to_numpy and hasattr(xp, "asnumpy") else out

        omega = (2.0 * float(PI)) * freqs_x
        nlayer = int(vs_arr.size)
        nf = int(freqs_x.size)

        vp_last = vp_scale * vs_arr[-1]
        rho_last = rho_arr[-1]
        Zb_c = xp.asarray(rho_last * vp_last, dtype=complex_dtype)

        eps_c = xp.asarray(complex(eps_f, 0.0), dtype=complex_dtype)
        zero_f = xp.asarray(0.0, dtype=float_dtype)
        floor_val = xp.asarray(1e-16, dtype=float_dtype)

        T00 = xp.ones(nf, dtype=complex_dtype)
        T01 = xp.zeros(nf, dtype=complex_dtype)
        T10 = xp.zeros(nf, dtype=complex_dtype)
        T11 = xp.ones(nf, dtype=complex_dtype)

        for j in range(nlayer):
            vpj = vp_scale * vs_arr[j]
            rhoj = rho_arr[j]
            hj = h_arr[j]
            Zj = rhoj * vpj
            kh = omega * hj / (vpj + eps_f)
            cos_kh = xp.cos(kh)
            sin_kh = xp.sin(kh)
            M00 = cos_kh.astype(complex_dtype, copy=False)
            M11 = M00
            denom_local = (1j * Zj) + eps_c
            sin_c = sin_kh.astype(complex_dtype, copy=False)
            M01 = sin_c / denom_local
            M10 = (1j * Zj) * sin_c
            nT00 = T00 * M00 + T01 * M10
            nT01 = T00 * M01 + T01 * M11
            nT10 = T10 * M00 + T11 * M10
            nT11 = T10 * M01 + T11 * M11
            T00, T01, T10, T11 = nT00, nT01, nT10, nT11

        denom = T00 + T01 * Zb_c
        denom = xp.where(xp.abs(denom) < eps_f, denom + eps_c, denom)

        Zsurf = (T10 + T11 * Zb_c) / (denom + eps_c)
        adm = 1.0 / (Zsurf + eps_c)
        ImGzz = xp.imag(adm)

        ImGzz = xp.where(xp.isfinite(ImGzz), ImGzz, zero_f)
        ImGzz = xp.maximum(ImGzz, zero_f)
        ImGzz = xp.clip(ImGzz, a_min=floor_val, a_max=None)

        invalid_mask = (~xp.isfinite(freqs_x)) | (freqs_x < 0.0)
        ImGzz = xp.where(invalid_mask, floor_val, ImGzz)

        out = ImGzz.astype(float_dtype, copy=False)
        if to_numpy:
            return xp.asnumpy(out) if hasattr(xp, "asnumpy") else np.asarray(out)
        return out


    #==========================
    # Likelihood / priors
    #==========================
    """
    def Waveform_Negative_Log_Likelihood(
        self,
        obs_x,
        synth_x,
        sigma_w,
        eps: Optional[float] = None,
        per_sample: bool = False,
        to_float: bool = False,
    ) -> "xp scalar or float":
        
        
        xp = self.xp
        if eps is None:
            eps = getattr(self, "EPS", 1e-12)

        accum_dtype = xp.float64 if (getattr(self, "use_high_precision", False) or xp is __import__("numpy")) else self.float_dtype

        obs = xp.asarray(obs_x)
        synth = xp.asarray(synth_x)

        try:
            sigma_w_f = float(xp.asarray(sigma_w)) if hasattr(xp, "asarray") and not isinstance(sigma_w,
                                                                                                float) else float(
                sigma_w)
        except Exception:
            # Fallback to eps
            sigma_w_f = float(eps)

        if not (sigma_w_f > 0.0 and sigma_w_f == sigma_w_f):  # check positive and not NaN
            # Make sure positive and finite
            sigma_w_f = max(abs(sigma_w_f) if isinstance(sigma_w_f, float) else eps, float(eps))

        mask = xp.isfinite(obs) & xp.isfinite(synth)

        try:
            n_valid = int(mask.sum().item()) if hasattr(mask.sum(), "item") else int(mask.sum())
        except Exception:
            try:
                n_valid = int(xp.asnumpy(mask.sum()))
            except Exception:
                n_valid = int(mask.sum())

        large_pen = getattr(self, "LARGE_PENALTY", 1e12)
        large_pen_x = xp.asarray(large_pen, dtype=accum_dtype)

        if n_valid == 0:
            return float(large_pen) if to_float else large_pen_x

        r = (obs - synth)[mask].astype(accum_dtype, copy=False) / float(sigma_w_f)

        r = xp.nan_to_num(r, nan=0.0, posinf=1e12, neginf=-1e12)

        s = 0.5 * xp.sum(r * r, dtype=accum_dtype)

        if per_sample:
            s = s / float(n_valid)

        s_out = s.astype(accum_dtype, copy=False)

        if to_float:
            try:
                return float(s_out.item()) if hasattr(s_out, "item") else float(xp.asnumpy(s_out))
            except Exception:
                return float(s_out)
        return s_out
    """


    def Waveform_Negative_Log_Likelihood(
        self,
        obs_x,      # Observed waveform
        synth_x,    # Synthetic waveform
        sigma_w,    # Measurement std (scalar or per-sample)
        eps: Optional[float] = None,    #
        per_sample: bool = False,       #
        to_float: bool = False,         #
        include_norm: bool = False,     # Include log-normalization
        return_grad: bool = False,      #
        clip_sigma_min: Optional[float] = None, #
        clip_sigma_max: Optional[float] = None, #
    ) -> "xp scalar or float or (xp scalar, xp array)":

        # Backend
        xp = self.xp
        if eps is None:
            eps = getattr(self, "EPS", 1e-12)

        accum_dtype = xp.float64 if (getattr(self, "use_high_precision", False) or xp is __import__("numpy")) else self.float_dtype

        obs = xp.asarray(obs_x)
        synth = xp.asarray(synth_x)

        if obs.shape != synth.shape:
            raise ValueError(f"obs and synth must have same shape (got {obs.shape} vs {synth.shape})")

        try:
            sigma_arr = xp.asarray(sigma_w)
        except Exception:
            sigma_arr = xp.asarray(float(eps), dtype=accum_dtype)

        is_scalar_sigma = (getattr(sigma_arr, "ndim", 0) == 0) or (sigma_arr.size == 1)

        mask = xp.isfinite(obs) & xp.isfinite(synth)

        try:
            n_valid = int(mask.sum().item()) if hasattr(mask.sum(), "item") else int(mask.sum())
        except Exception:
            try:
                n_valid = int(xp.asnumpy(mask.sum()))
            except Exception:
                n_valid = int(mask.sum())

        large_pen = getattr(self, "LARGE_PENALTY", 1e12)
        large_pen_x = xp.asarray(large_pen, dtype=accum_dtype)

        if n_valid == 0:
            if return_grad:
                grad_zero = xp.zeros_like(synth, dtype=accum_dtype)
                if to_float:
                    return float(large_pen), xp.asnumpy(grad_zero)
                return large_pen_x, grad_zero
            return float(large_pen) if to_float else large_pen_x

        if is_scalar_sigma:
            try:
                sigma_scalar = float(sigma_arr)
            except Exception:
                sigma_scalar = float(eps)
            if clip_sigma_min is not None:
                sigma_scalar = max(sigma_scalar, float(clip_sigma_min))
            else:
                sigma_scalar = max(sigma_scalar, float(eps))
            if clip_sigma_max is not None:
                sigma_scalar = min(sigma_scalar, float(clip_sigma_max))
            sigma_scalar = float(sigma_scalar) if (sigma_scalar == sigma_scalar and sigma_scalar > 0.0) else float(eps)
            sigma_masked = sigma_scalar
            sigma_squared_masked = float(sigma_scalar * sigma_scalar)
        else:
            if sigma_arr.shape != obs.shape:
                try:
                    sigma_arr = xp.broadcast_to(sigma_arr, obs.shape)
                except Exception:
                    raise ValueError("sigma_w array is not broadcastable to obs/synth shape")
            if clip_sigma_min is not None:
                sigma_arr = xp.maximum(sigma_arr, xp.asarray(float(clip_sigma_min), dtype=accum_dtype))
            else:
                sigma_arr = xp.maximum(sigma_arr, xp.asarray(float(eps), dtype=accum_dtype))
            if clip_sigma_max is not None:
                sigma_arr = xp.minimum(sigma_arr, xp.asarray(float(clip_sigma_max), dtype=accum_dtype))
            sigma_arr = xp.where(xp.isfinite(sigma_arr) & (sigma_arr > 0.0), sigma_arr, xp.asarray(float(eps), dtype=accum_dtype))
            sigma_masked = sigma_arr[mask].astype(accum_dtype, copy=False)
            sigma_squared_masked = sigma_masked * sigma_masked

        r = (obs - synth)[mask].astype(accum_dtype, copy=False)

        if is_scalar_sigma:
            r_scaled = r / float(sigma_scalar)
        else:
            r_scaled = r / sigma_masked

        # Sanitize
        r_scaled = xp.nan_to_num(r_scaled, nan=0.0, posinf=1e12, neginf=-1e12)

        quad = 0.5 * xp.sum(r_scaled * r_scaled, dtype=accum_dtype)

        # Normalization
        norm_term = xp.asarray(0.0, dtype=accum_dtype)
        if include_norm:
            if is_scalar_sigma:
                norm_term = 0.5 * float(n_valid) * math.log(2.0 * math.pi * (sigma_scalar * sigma_scalar))
                norm_term = xp.asarray(norm_term, dtype=accum_dtype)
            else:
                log_sigma = xp.log(sigma_masked)
                norm_term = 0.5 * xp.sum(math.log(2.0 * math.pi) + 2.0 * log_sigma, dtype=accum_dtype)

        nll_total = quad + norm_term

        if per_sample:
            nll_total = nll_total / float(n_valid)

        nll_out = nll_total.astype(accum_dtype, copy=False)

        if return_grad:
            if is_scalar_sigma:
                grad_masked = (((synth - obs)[mask].astype(accum_dtype, copy=False))) / (sigma_squared_masked)
            else:
                grad_masked = ((synth - obs)[mask].astype(accum_dtype, copy=False)) / (sigma_squared_masked)
            if per_sample:
                grad_masked = grad_masked / float(n_valid)
            grad_full = xp.zeros_like(synth, dtype=accum_dtype)
            grad_full[mask] = grad_masked
            if to_float:
                return float(nll_out.item() if hasattr(nll_out, "item") else xp.asnumpy(nll_out)), xp.asnumpy(grad_full)
            return nll_out, grad_full

        if to_float:
            return float(nll_out.item() if hasattr(nll_out, "item") else xp.asnumpy(nll_out))

        return nll_out



    #==============================================
    # HVSR Negative Log-Likelihood on Log-Domain
    #==============================================
    """
    def HVSR_Negative_Log_Likelihood(
        self,
        hv_obs_x,
        hv_syn_x,
        sigma_h,
        xp,
        eps=EPS
    ):

        # Backend
        hv_obs = xp.asarray(hv_obs_x)
        hv_syn = xp.asarray(hv_syn_x)

        sigma_h = float(sigma_h)
        if sigma_h <= 0.0 or not (sigma_h == sigma_h):
            sigma_h = max(sigma_h, float(eps))

        safe_obs = xp.where((xp.isfinite(hv_obs)) & (hv_obs > eps), hv_obs, eps).astype(xp.float64)
        safe_syn = xp.where((xp.isfinite(hv_syn)) & (hv_syn > eps), hv_syn, eps).astype(xp.float64)

        lobs = xp.log(safe_obs)
        ls = xp.log(safe_syn)

        d = (lobs - ls) / float(sigma_h)
        d = xp.nan_to_num(d, nan=0.0, posinf=1e12, neginf=-1e12)
        s = 0.5 * xp.sum(d * d, dtype=xp.float64)

        return s.astype(xp.float64)
    """


    def HVSR_Negative_Log_Likelihood(
        self,
        hv_obs_x,
        hv_syn_x,
        sigma_h,
        xp,
        eps=None,
        *,
        include_norm: bool = False,
        per_sample: bool = False,
        return_grad: bool = False,
        penalty_missing_lambda: float = 0.0,
        weights=None,
        robust: str = "gauss",
        nu: float = 4.0,
    ):

        if eps is None:
            eps = getattr(self, "EPS", 1e-12)

        # Convert inputs to xp arrays (device)
        hv_obs = xp.asarray(hv_obs_x)
        hv_syn = xp.asarray(hv_syn_x)

        if hv_obs.shape != hv_syn.shape:
            raise ValueError(f"hv_obs and hv_syn must have same shape, got {hv_obs.shape} vs {hv_syn.shape}")

        try:
            sigma_h_f = float(sigma_h)
        except Exception:
            sigma_h_f = float(eps)
        if not (sigma_h_f > 0.0 and sigma_h_f == sigma_h_f):
            sigma_h_f = max(float(eps), abs(sigma_h_f) if isinstance(sigma_h_f, float) else float(eps))

        acc_dtype = xp.float64

        mask = (xp.isfinite(hv_obs) & xp.isfinite(hv_syn) & (hv_obs > eps) & (hv_syn > eps))

        n_total = int(hv_obs.size)
        try:
            n_valid = int(mask.sum().item()) if hasattr(mask.sum(), "item") else int(mask.sum())
        except Exception:
            n_valid = int(xp.asnumpy(mask.sum()))

        large_pen = getattr(self, "LARGE_PENALTY", 1e12)
        large_pen_x = xp.asarray(large_pen, dtype=acc_dtype)

        if n_valid == 0:
            if return_grad:
                grad_zero = xp.zeros_like(hv_syn, dtype=acc_dtype)
                return (float(large_pen), grad_zero) if xp is __import__("numpy") else (large_pen_x, grad_zero)
            return float(large_pen) if xp is __import__("numpy") else large_pen_x

        obs_masked = hv_obs[mask].astype(acc_dtype, copy=False)
        syn_masked = hv_syn[mask].astype(acc_dtype, copy=False)

        lobs = xp.log(obs_masked)
        lsyn = xp.log(syn_masked)

        if weights is None:
            w_masked = xp.ones(n_valid, dtype=acc_dtype)
        else:
            w_arr = xp.asarray(weights)
            if w_arr.shape != hv_obs.shape:
                try:
                    w_arr = xp.broadcast_to(w_arr, hv_obs.shape)
                except Exception:
                    raise ValueError("weights not broadcastable to hv shape")
            w_masked = w_arr[mask].astype(acc_dtype, copy=False)
            w_masked = xp.maximum(w_masked, xp.asarray(0.0, dtype=acc_dtype))

        # Standardized residual in log-domain
        r = (lobs - lsyn) / float(sigma_h_f)
        # Sanitize
        r = xp.nan_to_num(r, nan=0.0, posinf=1e12, neginf=-1e12)

        # Compute likelihood
        if robust.lower() in ("gauss", "normal"):
            quad = 0.5 * xp.sum(w_masked * (r * r), dtype=acc_dtype)
            norm_term = xp.asarray(0.0, dtype=acc_dtype)
            if include_norm:
                lognorm = 0.5 * xp.sum(w_masked * (math.log(2.0 * math.pi) + 2.0 * math.log(float(sigma_h_f))), dtype=acc_dtype)
                norm_term = xp.asarray(lognorm, dtype=acc_dtype)
            nll_masked = quad + norm_term
            grad_lsyn_masked = (lsyn - lobs) * (w_masked / (sigma_h_f * sigma_h_f))
        elif robust.lower() in ("student", "t"):
            nu_f = float(nu)
            denom = 1.0 + (r * r) / nu_f
            coef = 0.5 * (nu_f + 1.0)
            nll_masked = xp.sum(w_masked * (coef * xp.log(denom)), dtype=acc_dtype)
            norm_term = xp.asarray(0.0, dtype=acc_dtype)
            if include_norm:
                const_part = 0.5 * math.log(nu_f * math.pi) + math.lgamma(nu_f / 2.0) - math.lgamma((nu_f + 1.0) / 2.0)
                norm_vec = const_part + xp.log(xp.asarray(float(sigma_h_f), dtype=acc_dtype))
                norm_term = xp.sum(w_masked * norm_vec, dtype=acc_dtype)
                norm_term = xp.asarray(norm_term, dtype=acc_dtype)
            grad_lsyn_masked = - (nu_f + 1.0) * r / (nu_f * denom * float(sigma_h_f))
            grad_lsyn_masked = grad_lsyn_masked * w_masked
        else:
            raise ValueError("robust must be 'gauss' or 'student'")

        if per_sample:
            nll_masked = nll_masked / float(n_valid)
            grad_lsyn_masked = grad_lsyn_masked / float(n_valid)

        # Missing-data penalty
        missing_count = n_total - n_valid
        missing_penalty = float(penalty_missing_lambda) * float(missing_count) if penalty_missing_lambda and (missing_count > 0) else 0.0

        nll_total = nll_masked + xp.asarray(missing_penalty, dtype=acc_dtype)

        # Build return scalar (xp)
        nll_out = xp.asarray(nll_total, dtype=acc_dtype)

        if return_grad:
            grad_masked = grad_lsyn_masked / syn_masked
            grad_full = xp.zeros_like(hv_syn, dtype=acc_dtype)
            grad_full = grad_full.astype(acc_dtype, copy=False)
            grad_full[mask] = grad_masked
            return (nll_out, grad_full)

        return nll_out


    def Gaussian_Prior_Negative_Log(
        self,
        params_x,
        prior_mean_x,
        prior_std_x,
        eps: Optional[float] = None,
        to_float: bool = False,
        per_param_weight: Optional["array-like"] = None
    ):
        # Backend
        xp = self.xp
        if eps is None:
            eps = getattr(self, "EPS", 1e-12)

        accum_dtype = xp.float64 if (not getattr(self, "use_gpu", False)) else self.float_dtype

        # Ensure arrays on backend
        p = xp.asarray(params_x)
        pm = xp.asarray(prior_mean_x)
        ps = xp.asarray(prior_std_x)

        if per_param_weight is not None:
            w = xp.asarray(per_param_weight)
        else:
            w = None

        try:
            p_b, pm_b, ps_b = xp.broadcast_arrays(p, pm, ps)
            if w is not None:
                w_b = xp.broadcast_to(w, p_b.shape)
            else:
                w_b = None
        except Exception:
            p_np = _np.asarray(p)
            pm_np = _np.asarray(pm)
            ps_np = _np.asarray(ps)
            try:
                p_b_np, pm_b_np, ps_b_np = _np.broadcast_arrays(p_np, pm_np, ps_np)
            except Exception:
                p_b_np = _np.atleast_1d(p_np)
                pm_b_np = _np.atleast_1d(pm_np)
                ps_b_np = _np.atleast_1d(ps_np)
            # Move back to xp
            p_b = xp.asarray(p_b_np, dtype=self.float_dtype)
            pm_b = xp.asarray(pm_b_np, dtype=self.float_dtype)
            ps_b = xp.asarray(ps_b_np, dtype=self.float_dtype)
            if w is not None:
                w_b = xp.asarray(_np.broadcast_to(_np.asarray(w), p_b_np.shape), dtype=self.float_dtype)
            else:
                w_b = None

        mask = xp.isfinite(p_b) & xp.isfinite(pm_b) & xp.isfinite(ps_b)
        if w_b is not None:
            mask &= xp.isfinite(w_b)

        try:
            n_valid = int(mask.sum().item()) if hasattr(mask.sum(), "item") else int(mask.sum())
        except Exception:
            try:
                n_valid = int(xp.asnumpy(mask.sum()))
            except Exception:
                n_valid = int(mask.sum())

        if n_valid == 0:
            zero_x = xp.asarray(0.0, dtype=accum_dtype)
            return float(zero_x) if to_float else zero_x

        p_v = p_b[mask].astype(accum_dtype, copy=False)
        pm_v = pm_b[mask].astype(accum_dtype, copy=False)
        ps_v = ps_b[mask].astype(accum_dtype, copy=False)

        ps_v = xp.maximum(xp.abs(ps_v), xp.asarray(eps, dtype=accum_dtype))

        r = (p_v - pm_v) / ps_v

        if w_b is not None:
            w_v = w_b[mask].astype(accum_dtype, copy=False)
            r = r * w_v

        # Sanitize
        r = xp.nan_to_num(r, nan=0.0, posinf=1e12, neginf=-1e12)

        s = 0.5 * xp.sum(r * r, dtype=accum_dtype)

        # Return xp scalar
        s_out = s.astype(accum_dtype, copy=False)
        if to_float:
            try:
                return float(s_out.item()) if hasattr(s_out, "item") else float(xp.asnumpy(s_out))
            except Exception:
                return float(s_out)
        return s_out


    #===========================
    # Linear Interpolation
    #===========================
    """
    def Linear_Interpolation(
        self,
        x_src,
        y_src,
        x_target,
        eps: Optional[float] = None,
        dtype: Optional["xp.dtype"] = None,
        to_numpy: bool = False,
    ):
        # Backend
        xp = self.xp
        if eps is None:
            eps = getattr(self, "EPS", 1e-12)
        if dtype is None:
            dtype = getattr(self, "float_dtype", xp.float64)

        # Ensure xp arrays on backend
        x_src = xp.asarray(x_src, dtype=xp.float64)
        y_src = xp.asarray(y_src, dtype=xp.float64)
        xq = xp.asarray(x_target, dtype=xp.float64)

        m = int(x_src.size)
        if m == 0:
            return xp.full_like(xq, xp.nan, dtype=dtype) if not to_numpy else xp.asnumpy(
                xp.full_like(xq, xp.nan, dtype=dtype))

        if m == 1:
            val = y_src[0]
            if not bool(xp.isfinite(val)):
                out = xp.full_like(xq, xp.nan, dtype=dtype)
            else:
                out = xp.full_like(xq, val.astype(dtype), dtype=dtype)
            return xp.asnumpy(out) if to_numpy and hasattr(xp, "asnumpy") else out

        try:
            is_sorted = bool(xp.all(x_src[1:] >= x_src[:-1]))
        except Exception:
            is_sorted = False

        if not is_sorted:
            order = xp.argsort(x_src)
            x_src = x_src[order]
            y_src = y_src[order]

        # Finite knots
        finite_mask = xp.isfinite(y_src)
        finite_idx = xp.where(finite_mask)[0]
        n_finite = int(finite_idx.size)

        if n_finite == 0:
            out = xp.full_like(xq, xp.nan, dtype=dtype)
            return xp.asnumpy(out) if to_numpy and hasattr(xp, "asnumpy") else out

        if n_finite == 1:
            const_val = y_src[finite_idx[0]].astype(dtype)
            out = xp.full_like(xq, const_val, dtype=dtype)
            return xp.asnumpy(out) if to_numpy and hasattr(xp, "asnumpy") else out

        # Knots from finite points
        xk = x_src[finite_idx]
        yk = y_src[finite_idx]

        def Interpolation_Knots(xk_local, yk_local, xt):
            idx = xp.searchsorted(xk_local, xt, side="right") - 1
            idx = xp.clip(idx, 0, int(xk_local.size) - 2)
            x0 = xk_local[idx]
            x1 = xk_local[idx + 1]
            y0 = yk_local[idx]
            y1 = yk_local[idx + 1]
            denom = x1 - x0
            tiny = xp.abs(denom) < float(eps)
            if xp.any(tiny):
                denom = xp.where(tiny, denom + float(eps), denom)
            t = (xt - x0) / denom
            return y0 * (1.0 - t) + y1 * t

        if not xp.all(finite_mask):
            nan_pos = xp.where(~finite_mask)[0]
            if nan_pos.size > 0:
                filled_vals = Interpolation_Knots(xk, yk, x_src[nan_pos])
                y_src_filled = x_src * 0.0 + y_src
                y_src_filled = y_src_filled.astype(xp.float64, copy=True)
                y_src_filled[nan_pos] = filled_vals
            else:
                y_src_filled = y_src.astype(xp.float64, copy=True)
        else:
            y_src_filled = y_src.astype(xp.float64, copy=True)

        yq = Interpolation_Knots(x_src, y_src_filled, xq)

        # Sanitize results
        yq = xp.where(xp.isfinite(yq), yq, xp.nan)

        try:
            out = yq.astype(dtype, copy=False)
        except Exception:
            out = yq

        if to_numpy and hasattr(xp, "asnumpy"):
            return xp.asnumpy(out)
        return out
    """


    def Linear_Interpolation(
        self,
        x_src,
        y_src,
        x_target,
        eps: Optional[float] = None,
        dtype: Optional["xp.dtype"] = None,
        to_numpy: bool = False,
    ):
        xp = self.xp

        if eps is None:
            eps = getattr(self, "EPS", 1e-12)
        if dtype is None:
            dtype = getattr(self, "float_dtype", xp.float64)

        def _to_scalar_float(v):
            try:
                if hasattr(v, "item"):
                    return float(v.item())
                return float(v)
            except Exception:
                try:
                    if hasattr(xp, "asnumpy"):
                        return float(xp.asnumpy(v).item())
                except Exception:
                    pass
                return float(v)

        def _maybe_to_numpy(arr):
            if to_numpy and hasattr(xp, "asnumpy"):
                return xp.asnumpy(arr)
            return arr

        # Ensure 1-D arrays
        x_src = xp.asarray(x_src, dtype=xp.float64).ravel()
        y_src = xp.asarray(y_src, dtype=xp.float64).ravel()
        xq = xp.asarray(x_target, dtype=xp.float64).ravel()

        m = int(x_src.size)
        if m == 0:
            out = xp.full_like(xq, xp.nan, dtype=dtype)
            return _maybe_to_numpy(out)

        if m == 1:
            val = y_src[0]
            if not bool(xp.isfinite(val).item() if hasattr(val, "item") else xp.isfinite(val)):
                out = xp.full_like(xq, xp.nan, dtype=dtype)
            else:
                out = xp.full_like(xq, xp.asarray(val, dtype=dtype), dtype=dtype)
            return _maybe_to_numpy(out)

        try:
            is_sorted = bool(xp.all(x_src[1:] >= x_src[:-1]).item())
        except Exception:
            is_sorted = False

        try:
            y_finite_all = bool(xp.all(xp.isfinite(y_src)).item())
        except Exception:
            y_finite_all = False

        if is_sorted and y_finite_all:
            x_src_sig = (
                int(x_src.size),
                _to_scalar_float(x_src[0]),
                _to_scalar_float(x_src[-1]),
                _to_scalar_float(x_src[1] - x_src[0]) if x_src.size > 1 else 0.0,
            )
            xq_sig = (
                int(xq.size),
                _to_scalar_float(xq[0]),
                _to_scalar_float(xq[-1]),
                _to_scalar_float(xq[1] - xq[0]) if xq.size > 1 else 0.0,
            )
            cache_key = (x_src_sig, xq_sig, str(dtype), _to_scalar_float(eps))

            cache = getattr(self, "_linear_interp_cache", None)
            if isinstance(cache, dict) and cache.get("key", None) == cache_key:
                idx = cache["idx"]
                t = cache["t"]
                yq = y_src[idx] * (1.0 - t) + y_src[idx + 1] * t
                yq = xp.where(xp.isfinite(yq), yq, xp.nan)
                try:
                    yq = yq.astype(dtype, copy=False)
                except Exception:
                    pass
                return _maybe_to_numpy(yq)

            idx = xp.searchsorted(x_src, xq, side="right") - 1
            idx = xp.clip(idx, 0, int(x_src.size) - 2)

            x0 = x_src[idx]
            x1 = x_src[idx + 1]
            denom = x1 - x0
            denom = xp.where(xp.abs(denom) < float(eps), float(eps), denom)

            t = (xq - x0) / denom
            t = xp.clip(t, 0.0, 1.0)

            yq = y_src[idx] * (1.0 - t) + y_src[idx + 1] * t
            yq = xp.where(xp.isfinite(yq), yq, xp.nan)

            try:
                yq = yq.astype(dtype, copy=False)
            except Exception:
                pass

            self._linear_interp_cache = {
                "key": cache_key,
                "idx": idx,
                "t": t,
            }
            return _maybe_to_numpy(yq)

        try:
            order = xp.argsort(x_src)
            x_src = x_src[order]
            y_src = y_src[order]
        except Exception:
            pass

        finite_mask = xp.isfinite(y_src)
        finite_idx = xp.where(finite_mask)[0]
        n_finite = int(finite_idx.size)

        if n_finite == 0:
            out = xp.full_like(xq, xp.nan, dtype=dtype)
            return _maybe_to_numpy(out)

        if n_finite == 1:
            const_val = xp.asarray(y_src[finite_idx[0]], dtype=dtype)
            out = xp.full_like(xq, const_val, dtype=dtype)
            return _maybe_to_numpy(out)

        xk = x_src[finite_idx]
        yk = y_src[finite_idx]

        if not bool(xp.all(finite_mask).item()):
            nan_pos = xp.where(~finite_mask)[0]
            if nan_pos.size > 0:
                idx = xp.searchsorted(xk, x_src[nan_pos], side="right") - 1
                idx = xp.clip(idx, 0, int(xk.size) - 2)
                x0 = xk[idx]
                x1 = xk[idx + 1]
                y0 = yk[idx]
                y1 = yk[idx + 1]
                denom = x1 - x0
                denom = xp.where(xp.abs(denom) < float(eps), float(eps), denom)
                t = (x_src[nan_pos] - x0) / denom
                filled_vals = y0 * (1.0 - t) + y1 * t
                y_src_filled = y_src.astype(xp.float64, copy=True)
                y_src_filled[nan_pos] = filled_vals
            else:
                y_src_filled = y_src.astype(xp.float64, copy=True)
        else:
            y_src_filled = y_src.astype(xp.float64, copy=True)

        # Interpolate to target
        idx = xp.searchsorted(x_src, xq, side="right") - 1
        idx = xp.clip(idx, 0, int(x_src.size) - 2)

        x0 = x_src[idx]
        x1 = x_src[idx + 1]
        y0 = y_src_filled[idx]
        y1 = y_src_filled[idx + 1]
        denom = x1 - x0
        denom = xp.where(xp.abs(denom) < float(eps), float(eps), denom)

        t = (xq - x0) / denom
        yq = y0 * (1.0 - t) + y1 * t
        yq = xp.where(xp.isfinite(yq), yq, xp.nan)

        try:
            yq = yq.astype(dtype, copy=False)
        except Exception:
            pass

        return _maybe_to_numpy(yq)


    #=================================================================================================
    # Local Markov-Chain Monte-Carlo (MCMC) (Metropolis-Hastings) around Maximum A Posteriori (MAP)
    #=================================================================================================
    """
    def Local_Markov_Chain_Monte_Carlo(
        self,
        start_params,
        niter,
        neglogpost_callable,
        args=(),
        proposal_scale=None,
        thin=THIN,
        burnin=0,
        xp: Optional[object] = None,
        dtype: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        # Backend
        xp_backend = xp if xp is not None else getattr(self, "xp", __import__("numpy"))
        is_cupy = hasattr(xp_backend, "asnumpy")

        p_host = np.asarray(start_params, dtype=float)
        ndim = int(p_host.size)

        if proposal_scale is None:
            proposal_scale = 0.02 * np.abs(p_host) + 0.05
        prop_scale_host = np.asarray(proposal_scale, dtype=float)
        if prop_scale_host.size == 1:
            prop_scale_host = np.full(ndim, float(prop_scale_host.item()), dtype=float)
        elif prop_scale_host.size != ndim:
            try:
                prop_scale_host = np.broadcast_to(prop_scale_host, (ndim,))
            except Exception:
                raise ValueError("proposal_scale must be scalar or shape (ndim,)")

        if dtype is None:
            dev_dtype = xp_backend.float32 if is_cupy else np.float64
        else:
            if dtype == "float32":
                dev_dtype = xp_backend.float32 if is_cupy else np.float32
            elif dtype == "float64":
                dev_dtype = xp_backend.float64 if is_cupy else np.float64
            else:
                raise ValueError("dtype must be 'float32' or 'float64'")

        # RNG setup
        rng_normal = None
        rng_uniform_scalar = None
        if is_cupy:
            try:
                rng = xp_backend.random.default_rng(seed)
                def rng_normal(shape):
                    return rng.standard_normal(size=shape).astype(dev_dtype)
                def rng_uniform_scalar():
                    u_dev = rng.random(size=())
                    return float(xp_backend.asnumpy(u_dev))
            except Exception:
                try:
                    rng = xp_backend.random.RandomState(seed)
                    def rng_normal(shape):
                        return xp_backend.asarray(rng.normal(size=shape)).astype(dev_dtype)
                    def rng_uniform_scalar():
                        return float(rng.rand())
                except Exception:
                    rng_np = np.random.default_rng(seed)
                    def rng_normal(shape):
                        return xp_backend.asarray(rng_np.standard_normal(size=shape)).astype(dev_dtype)
                    def rng_uniform_scalar():
                        return float(rng_np.random())
        else:
            rng_np = np.random.default_rng(seed)
            def rng_normal(shape):
                return rng_np.standard_normal(size=shape).astype(dev_dtype)
            def rng_uniform_scalar():
                return float(rng_np.random())

        # Prepare parameter arrays on device
        p_dev = xp_backend.asarray(p_host, dtype=dev_dtype)
        prop_scale_dev = xp_backend.asarray(prop_scale_host, dtype=dev_dtype)

        LARGE_PEN = getattr(self, "LARGE_PENALTY", self.EPS)
        EPS = getattr(self, "EPS", self.EPS)

        # Evaluator for negative log-posterior
        def Evaluate_Negative_Log(p_candidate):
            try:
                try:
                    val = neglogpost_callable(p_candidate, *args)
                except Exception:
                    try:
                        ph = xp_backend.asnumpy(p_candidate) if hasattr(xp_backend, "asnumpy") else np.asarray(p_candidate)
                    except Exception:
                        ph = np.asarray(p_candidate)
                    val = neglogpost_callable(ph, *args)
            except Exception as e:
                if verbose:
                    try:
                        msg = str(e)
                    except Exception:
                        msg = "exception"
                    print(f"[MH][WARN] neglog evaluation failed: {msg}. Using LARGE_PENALTY.")
                return float(LARGE_PEN)

            # Coerce result to python float
            try:
                if is_cupy:
                    if hasattr(val, "dtype") or hasattr(val, "__array__"):
                        try:
                            vhost = xp_backend.asnumpy(val).item() if xp_backend.asarray(val).size == 1 else \
                                xp_backend.asnumpy(val).ravel()[0]
                        except Exception:
                            vhost = float(val)
                    else:
                        vhost = float(val)
                else:
                    try:
                        vhost = float(np.asarray(val).item()) if np.asarray(val).size == 1 else float(val)
                    except Exception:
                        vhost = float(val)
            except Exception:
                vhost = float(LARGE_PEN)

            # Final numeric check
            if not np.isfinite(vhost):
                return float(LARGE_PEN)
            return float(vhost)

        # Initial evaluation
        neglog_curr = Evaluate_Negative_Log(p_dev)
        logp_curr = -neglog_curr

        accepts = 0

        # Sanity checks
        if niter <= burnin:
            raise ValueError("niter must be > burnin")
        if thin <= 0:
            raise ValueError("thin must be >= 1")

        # Compute number of saved samples
        nsave = max(0, (niter - burnin + (thin - 1)) // thin)
        if nsave > 0:
            if is_cupy:
                samples_dev = xp_backend.empty((nsave, ndim), dtype=dev_dtype)
            else:
                samples_dev = np.empty((nsave, ndim), dtype=np.float64)
        else:
            samples_dev = None

        save_idx = 0

        # Markov-Chain Monte-Carlo Loop
        for it in range(int(niter)):
            # Gaussian Random Walk
            try:
                z = rng_normal(shape=p_dev.shape)
                prop = p_dev + z * prop_scale_dev
            except Exception:
                z = np.random.normal(size=p_host.shape).astype(dev_dtype)
                prop = p_dev + xp_backend.asarray(z, dtype=dev_dtype) * prop_scale_dev

            # Evaluate proposal
            neglog_prop = Evaluate_Negative_Log(prop)
            logp_prop = -neglog_prop
            log_alpha = logp_prop - logp_curr

            # Sample uniform scalar
            try:
                u_val = rng_uniform_scalar()
                u_log = math.log(u_val + TINY)
            except Exception:
                u_val = float(np.random.rand())
                u_log = math.log(u_val + TINY)

            # Accept condition
            if u_log < log_alpha:
                p_dev = prop
                logp_curr = logp_prop
                neglog_curr = neglog_prop
                accepts += 1

            # Save if past burnin and thin
            if it >= burnin and ((it - burnin) % thin == 0):
                if nsave > 0:
                    if is_cupy:
                        samples_dev[save_idx, :] = p_dev
                    else:
                        samples_dev[save_idx, :] = np.asarray(p_dev, dtype=float)
                    save_idx += 1

            # Progress
            if verbose and (it + 1) % max(1, niter // 10) == 0:
                print(f"[MH] iter {it + 1}/{niter} accepted={accepts}/{it + 1} logp={logp_curr:.3f}")

        # Copy samples to host numpy
        if nsave > 0:
            if is_cupy:
                try:
                    samples_host = xp_backend.asnumpy(samples_dev)
                except Exception:
                    samples_host = np.array(samples_dev.get(), dtype=float)
            else:
                samples_host = np.asarray(samples_dev, dtype=float)
        else:
            samples_host = np.empty((0, ndim), dtype=float)

        accept_rate = float(accepts) / float(niter)
        return {"samples": samples_host, "accept_rate": accept_rate}
    """


    def Local_Markov_Chain_Monte_Carlo(
        self,
        start_params,
        niter,
        neglogpost_callable,
        args=(),
        proposal_scale=None,
        thin=THIN,
        burnin=0,
        xp: Optional[object] = None,
        dtype: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        xp_backend = xp if xp is not None else getattr(self, "xp", np)
        is_cupy = hasattr(xp_backend, "asnumpy")

        rng_cpu = np.random.default_rng(seed)
        rng_gpu = xp_backend.random.default_rng(seed) if is_cupy else None

        p_host = np.asarray(start_params, dtype=float)
        ndim = int(p_host.size)

        if proposal_scale is None:
            proposal_scale = 0.02 * np.abs(p_host) + 0.05

        prop_scale_host = np.asarray(proposal_scale, dtype=float)
        if prop_scale_host.size == 1:
            prop_scale_host = np.full(ndim, float(prop_scale_host.item()), dtype=float)
        elif prop_scale_host.size != ndim:
            prop_scale_host = np.broadcast_to(prop_scale_host, (ndim,))

        if dtype is None:
            dev_dtype = xp_backend.float32 if is_cupy else np.float64
        else:
            if dtype == "float32":
                dev_dtype = xp_backend.float32 if is_cupy else np.float32
            elif dtype == "float64":
                dev_dtype = xp_backend.float64 if is_cupy else np.float64
            else:
                raise ValueError("dtype must be 'float32' or 'float64'")

        curr_dev = xp_backend.asarray(p_host, dtype=dev_dtype)
        cand_dev = xp_backend.empty_like(curr_dev)
        noise_dev = xp_backend.empty_like(curr_dev)
        prop_scale_dev = xp_backend.asarray(prop_scale_host, dtype=dev_dtype)

        LARGE_PEN = getattr(self, "LARGE_PENALTY", self.EPS)
        TINY = getattr(self, "TINY", 1e-300)

        def Evaluate_Negative_Log(p_candidate):
            try:
                val = neglogpost_callable(p_candidate, *args)
            except Exception:
                return float(LARGE_PEN)

            try:
                if hasattr(val, "item"):
                    vhost = float(val.item())
                else:
                    vhost = float(val)
            except Exception:
                return float(LARGE_PEN)

            if not np.isfinite(vhost):
                return float(LARGE_PEN)

            return vhost

        # Initial evaluation
        neglog_curr = Evaluate_Negative_Log(curr_dev)
        logp_curr = -neglog_curr

        if niter <= burnin:
            raise ValueError("niter must be > burnin")
        if thin <= 0:
            raise ValueError("thin must be >= 1")

        nsave = max(0, (niter - burnin + (thin - 1)) // thin)
        samples_dev = xp_backend.empty((nsave, ndim), dtype=dev_dtype) if nsave > 0 else None
        save_idx = 0
        accepts = 0

        for it in range(int(niter)):
            # Proposal noise
            if is_cupy:
                noise_dev[...] = rng_gpu.standard_normal(size=curr_dev.shape, dtype=dev_dtype)
            else:
                noise_dev[...] = rng_cpu.standard_normal(size=curr_dev.shape).astype(dev_dtype)

            # cand = curr + noise * scale
            xp_backend.multiply(noise_dev, prop_scale_dev, out=cand_dev)
            xp_backend.add(curr_dev, cand_dev, out=cand_dev)

            # Evaluate proposal
            neglog_prop = Evaluate_Negative_Log(cand_dev)
            log_alpha = -neglog_prop - logp_curr

            # Accept/reject
            u_log = math.log(rng_cpu.random() + TINY)
            if u_log < log_alpha:
                curr_dev, cand_dev = cand_dev, curr_dev
                logp_curr = -neglog_prop
                neglog_curr = neglog_prop
                accepts += 1

            # Save samples
            if it >= burnin and ((it - burnin) % thin == 0) and nsave > 0:
                samples_dev[save_idx, :] = curr_dev
                save_idx += 1

            if verbose and (it + 1) % max(1, niter // 10) == 0:
                print(f"[MH] iter {it + 1}/{niter} accepted={accepts}/{it + 1} logp={logp_curr:.3f}")

        if nsave > 0:
            samples_host = xp_backend.asnumpy(samples_dev) if is_cupy else np.asarray(samples_dev, dtype=float)
        else:
            samples_host = np.empty((0, ndim), dtype=float)

        accept_rate = float(accepts) / float(niter)
        return {"samples": samples_host, "accept_rate": accept_rate}


    @staticmethod
    def choose_xp(self):
        return (cp if (self.use_gpu and use_cupy_global) else np), (self.use_gpu and use_cupy_global)


    @staticmethod
    def Layer_Params_to_Grid(params, nz_local, dz_local, layer_thickness_local):
        params = np.asarray(params, dtype=float)
        layer_thickness_local = np.asarray(layer_thickness_local, dtype=float)
        cells = [max(1, int(round(th / float(dz_local)))) for th in layer_thickness_local]
        total_cells = sum(cells)
        if total_cells != nz_local:
            cells[-1] += (nz_local - total_cells)
            if cells[-1] < 1:
                cells[-1] = 1
                diff = nz_local - sum(cells)
                i = len(cells) - 2
                while diff != 0 and i >= 0:
                    add = 1 if diff > 0 else -1
                    cells[i] += add
                    diff = nz_local - sum(cells)
                    i -= 1
        grid = np.empty(nz_local, dtype=float)
        idx = 0
        for i, n_cells in enumerate(cells):
            end = min(nz_local, idx + n_cells)
            val = float(params[i]) if i < params.size else float(params[-1])
            grid[idx:end] = val
            idx = end
        if idx < nz_local:
            grid[idx:] = float(params[-1] if params.size > 0 else 0.0)
        return grid


    def Model_Section(
        self,
        out_dir,
        samples,
        layer_thickness=None,
        median_vs=None,
        vs_grid_median=None,
        vs_grid_lower=None,
        vs_grid_upper=None,
        dz=0.5,
        nz=None,
        z=None,
        cmap='viridis',
        log_x=False,
        Nplot_max=300,
        save_name_prefix="Model_Section"
    ):
        # Validate inputs
        if layer_thickness is None:
            raise RuntimeError("layer_thickness must be provided to Model_Section.")
        layer_thickness = np.asarray(layer_thickness, dtype=float)
        nlayer = layer_thickness.size

        # Derive depth grid
        if z is None:
            total_depth = float(np.sum(layer_thickness))
            if nz is None:
                nz = max(1, int(round(total_depth / float(dz))))
            z = np.linspace(0.0, total_depth, nz, endpoint=False)
        else:
            nz = int(len(z))

        # Produce vs_grid_median / lower / upper
        if vs_grid_median is None:
            if median_vs is None:
                if samples is not None and getattr(samples, "size", 0) > 0:
                    median_vs = np.median(samples, axis=0)
                else:
                    raise RuntimeError("Need median_vs or samples to build vs_grid_median.")
            vs_grid_median = self.Layer_Params_to_Grid(median_vs)

        if vs_grid_lower is None or vs_grid_upper is None:
            if samples is not None and getattr(samples, "size", 0) > 0:
                p5 = np.percentile(samples, 5, axis=0)
                p95 = np.percentile(samples, 95, axis=0)
                vs_grid_lower = self.Layer_Params_to_Grid(p5) if vs_grid_lower is None else vs_grid_lower
                vs_grid_upper = self.Layer_Params_to_Grid(p95) if vs_grid_upper is None else vs_grid_upper
            else:
                vs_grid_lower = vs_grid_lower if vs_grid_lower is not None else vs_grid_median
                vs_grid_upper = vs_grid_upper if vs_grid_upper is not None else vs_grid_median

        # Compute layer tops for markers
        layer_tops = np.concatenate([[0.0], np.cumsum(layer_thickness)[:-1]])

        # Compute Vs30
        def compute_vs30(median_vs_local, layer_thickness_local, top_depth=30.0):
            if median_vs_local is None:
                return np.nan
            cum = 0.0
            vs_vals = []
            thickness = []
            for th, vs in zip(layer_thickness_local, median_vs_local):
                thf = float(th)
                take = float(np.clip(top_depth - cum, 0.0, thf))
                if take > 0:
                    vs_vals.append(float(vs))
                    thickness.append(take)
                    cum += take
                if cum >= top_depth:
                    break
            if len(vs_vals) == 0:
                return np.nan
            vs_vals = np.array(vs_vals)
            thickness = np.array(thickness)
            return np.sum(vs_vals * thickness) / np.sum(thickness)

        vs30 = compute_vs30(median_vs, layer_thickness)

        # Plotting
        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(constrained_layout=False, figsize=(9.0, 6.0))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.15], wspace=0.12)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_fan = fig.add_subplot(gs[:, 1])
        ax_cbar = fig.add_subplot(gs[:, 2])

        repeats = 12

        #--------------------
        # SYNC GRID SIZE
        #--------------------
        # Ensure arrays are numpy 1D
        vs_grid_median = np.asarray(vs_grid_median).ravel()
        if vs_grid_lower is not None:
            vs_grid_lower = np.asarray(vs_grid_lower).ravel()
        if vs_grid_upper is not None:
            vs_grid_upper = np.asarray(vs_grid_upper).ravel()

        if vs_grid_median.size != nz:
            old_nz = nz
            old_dz = dz
            nz = int(vs_grid_median.size)
            total_depth = float(np.sum(layer_thickness))
            dz = total_depth / float(nz) if total_depth > 0 else dz
            z = np.linspace(0.0, total_depth, nz, endpoint=False)

            print(f"[WARN] vs_grid_median length ({vs_grid_median.size}) != nz ({old_nz}).")
            print(f"[WARN] Adjusting nz->{nz}, dz:{old_dz:.4g}->{dz:.6g} and recomputing z to match vs_grid_median.")

            def Resample_to_nz(arr, target_nz, total_depth):
                if arr is None:
                    return None
                arr = np.asarray(arr).ravel()
                if arr.size == target_nz:
                    return arr
                z_old = np.linspace(0.0, total_depth, arr.size, endpoint=False)
                z_new = np.linspace(0.0, total_depth, target_nz, endpoint=False)
                return np.interp(z_new, z_old, arr)

            vs_grid_lower = Resample_to_nz(vs_grid_lower, nz, total_depth)
            vs_grid_upper = Resample_to_nz(vs_grid_upper, nz, total_depth)

        nz = vs_grid_median.size
        slab = np.tile(vs_grid_median.reshape(nz, 1), (1, repeats))
        X = np.linspace(0, 1, repeats + 1)
        Y = np.linspace(0.0, dz * nz, nz + 1)
        pcm = ax_main.pcolormesh(X, Y, slab, cmap=cmap, shading='auto',
                                 norm=Normalize(vmin=np.nanmin(vs_grid_lower),
                                                vmax=np.nanmax(vs_grid_upper)))
        ax_main.step(vs_grid_median, z, where='post', color='white', linewidth=2.0, zorder=3, alpha=0.95)
        for top in layer_tops:
            ax_main.hlines(top, 0.0, 1.0, colors='k', linewidth=0.6, alpha=0.35)
        cbar = fig.colorbar(pcm, cax=ax_cbar, orientation='vertical', pad=0.02)
        cbar.set_label('Vs (m/s)', fontsize=10)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(z.max() + dz * 0.5, -dz * 0.5)
        ax_main.set_xticks([])
        ax_main.set_ylabel('Depth (m)', fontsize=11)
        ax_main.set_title('Model (median) — blocky Vs', fontsize=12)
        ax_main.set_title('1D Vs Block Model')

        ax_profile = ax_main.inset_axes([0.10, 0.55, 0.8, 0.4])
        ax_profile.patch.set_alpha(0.0)
        if log_x:
            ax_profile.semilogx(vs_grid_median, z, color='black', linewidth=2.0)
        else:
            ax_profile.plot(vs_grid_median, z, color='black', linewidth=2.0)
        ax_profile.invert_yaxis()
        ax_profile.set_xlabel('Vs (m/s)', fontsize=8)
        ax_profile.set_ylabel('Depth (m)', fontsize=8)
        ax_profile.tick_params(axis='both', which='major', labelsize=8)
        if not np.isnan(vs30):
            ax_profile.axvline(vs30, color='tab:orange', linestyle='--', linewidth=1.0)
            ax_profile.text(0.98, 0.02, f"Vs30={vs30:.0f} m/s", transform=ax_profile.transAxes,
                            va='bottom', ha='right', fontsize=8, color='tab:orange',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Right panel
        # ax_fan.set_title('Posterior ensemble', fontsize=12)
        ax_fan.set_title('1D Vs Model')
        ax_fan.set_ylim(z.max() + dz * 0.5, -dz * 0.5)
        ax_fan.set_xlabel('Vs (m/s)', fontsize=10)
        if samples is not None and getattr(samples, "size", 0) > 0:
            nsamp = samples.shape[0]
            nplot = min(Nplot_max, nsamp)
            rng = np.random.default_rng(0)
            idx = rng.choice(nsamp, size=nplot, replace=False)
            for ii in idx:
                s = samples[ii]
                if s.size == nlayer:
                    sgrid = self.Layer_Params_to_Grid(s)
                elif s.size == nz:
                    sgrid = s
                else:
                    sgrid = self.Layer_Params_to_Grid(s[:nlayer], nz, dz, layer_thickness)
                ax_fan.step(sgrid, z, where='post', linewidth=0.6, alpha=0.05, color='tab:blue')

        ax_fan.fill_betweenx(z, vs_grid_lower, vs_grid_upper, step='post', color='tab:orange', alpha=0.28, label='Credible Interval')
        ax_fan.step(vs_grid_median, z, where='post', color='tab:orange', linewidth=2.0, label='median')
        if median_vs is not None:
            if median_vs.size == len(layer_tops):
                ax_fan.scatter(median_vs, layer_tops, color='k', s=30, edgecolors='white', zorder=4)
            else:
                layer_vs_repr = []
                cum_idx = 0
                for th in layer_thickness:
                    n_cells = max(1, int(round(th / float(dz))))
                    end = min(nz, cum_idx + n_cells)
                    layer_vs_repr.append(np.mean(median_vs[cum_idx:end]))
                    cum_idx = end
                layer_vs_repr = np.array(layer_vs_repr)
                ax_fan.scatter(layer_vs_repr, layer_tops, color='k', s=30, edgecolors='white', zorder=4)

        if log_x:
            ax_fan.set_xscale('log')
            ax_fan.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax_fan.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)))
        ax_fan.grid(axis='x', which='both', linestyle=':', linewidth=0.5)
        ax_fan.legend(loc='upper right', fontsize=9)

        # Save outputs
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{save_name_prefix}.png")
        pdf_path = os.path.join(out_dir, f"{save_name_prefix}.pdf")
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        return png_path, pdf_path


    def Bayesian_Inversion(
        self,
        layer_thickness: Sequence[float],
        layer_vs_true: Sequence[float],
        layer_rho: Sequence[float],
        dz: float = DZ,
        t_total: float = T_TOTAL,
        f0: float = F0,
        waveform_noise_std: float = WAVEFORM_NOISE,
        hv_noise_std: float = HV_NOISE,
        use_gpu: Optional[bool] = None,
        map_opts: Optional[dict] = None,
        mh_opts: Optional[dict] = None,
        out_dir: str = "./Bayesian_FWI_out",
        dtype: str = "float32",
        obs_waveforms: Optional[np.ndarray] = None,
        obs_dt: Optional[float] = None,
        obs_hvsr: Optional[np.ndarray] = None,
        obs_freqs: Optional[np.ndarray] = None,
        hv_method: str = "welch",
        welch_nperseg: Optional[int] = None,
    ) -> Dict[str, Any]:

        # Backend
        xp = getattr(self, "xp", None)
        is_cupy = getattr(self, "is_cupy", False)
        use_gpu_eff = bool(use_gpu) if use_gpu is not None else bool(getattr(self, "use_gpu", False))

        if xp is None:
            if use_gpu_eff:
                try:
                    import cupy as cp
                    xp = cp
                    is_cupy = True
                except Exception:
                    xp = np
                    is_cupy = False

        # Helper for device
        def to_host(arr):
            if is_cupy:
                try:
                    return xp.asnumpy(arr)
                except Exception:
                    import cupy as cp_local
                    return cp_local.asnumpy(arr)
            else:
                return np.asarray(arr)

        rng = np.random.default_rng(getattr(self, "seed", None))

        print("[RUN] Bayesian Full Waveform Inversion: backend:", getattr(xp, "__name__", str(xp)), "use_gpu:", is_cupy)

        os.makedirs(out_dir, exist_ok=True)

        #--------------------------------
        # Grid and layer mapping
        #--------------------------------
        total_layer_depth = float(np.sum(layer_thickness))
        pad_depth = float(getattr(self, "pad_depth", 60.0))
        model_depth = total_layer_depth + pad_depth
        nz = int(math.ceil(model_depth / dz)) + 1
        z = np.linspace(0.0, (nz - 1) * dz, nz)

        layer_cell_counts = [max(1, int(round(th / dz))) for th in layer_thickness]
        layer_slices = []
        cur = 0
        for c in layer_cell_counts:
            layer_slices.append((cur, min(cur + c, nz)))
            cur += c

        # True model grid arrays (host)
        halfspace_vs = float(layer_vs_true[-1])
        vs_grid_true = np.ones(nz, dtype=float) * halfspace_vs
        cur = 0
        for p, th in zip(layer_vs_true, layer_thickness):
            ncell = max(1, int(round(th / dz)))
            end = min(cur + ncell, nz)
            vs_grid_true[cur:end] = float(p)
            cur = end

        rho_grid = np.ones(nz, dtype=float) * float(layer_rho[-1])
        cur = 0
        for rho_val, th in zip(layer_rho, layer_thickness):
            ncell = max(1, int(round(th / dz)))
            end = min(cur + ncell, nz)
            rho_grid[cur:end] = float(rho_val)
            cur = end

        nlayer = len(layer_vs_true)

        #--------------------------------
        # Time-Stepping / source
        #--------------------------------
        vs_max = float(np.max(vs_grid_true))
        dt = float(CFL * dz / (vs_max + V_CORRECTION))
        nt = int(math.ceil(t_total / dt))
        time_vec = np.arange(nt) * dt

        def ricker(t):
            t0 = 1.5 / f0
            x = math.pi * f0 * (t - t0)
            return (1.0 - 2.0 * x * x) * math.exp(-x * x)

        src_time_host = np.array([ricker(tt) for tt in time_vec], dtype=float)
        maxabs = np.max(np.abs(src_time_host)) + 1e-15
        src_time_host /= maxabs
        src_i = int(max(1, round(0.5 * layer_thickness[0] / dz)))

        float_dtype = xp.float32 if dtype == "float32" and is_cupy else (xp.float32 if dtype == "float32" else xp.float64)
        complex_dtype = xp.complex64 if is_cupy and dtype == "float32" else (np.complex128 if not is_cupy else xp.complex128)

        # Move ricker to device
        src_time_dev = xp.asarray(src_time_host, dtype=float_dtype)

        #--------------------------------
        # Preallocate device arrays
        #--------------------------------
        vs_dev = xp.ones(nz, dtype=float_dtype)
        rho_dev = xp.asarray(rho_grid, dtype=float_dtype)

        v_dev = xp.zeros(nz, dtype=float_dtype)
        tau_dev = xp.zeros(max(1, nz - 1), dtype=float_dtype)
        u_dev = xp.zeros(nz, dtype=float_dtype)

        vs_true_dev = xp.asarray(vs_grid_true, dtype=float_dtype)
        mu_half_dev = 0.5 * (rho_dev[:-1] * vs_true_dev[:-1] * vs_true_dev[:-1] + rho_dev[1:] * vs_true_dev[1:] * vs_true_dev[
                                                                                                     1:])
        rho_inv_dev = 1.0 / (rho_dev + EPS)

        sponge_dev = xp.zeros(nz, dtype=float_dtype)
        sponge_cells = max(20, min(80, nz // 8))
        if sponge_cells > 0:
            ramp = xp.linspace(0.0, 1.0, sponge_cells, dtype=float_dtype)
            sponge_dev[-sponge_cells:] = 0.02 * (ramp * ramp)

        #-------------------
        # FFT parameters
        #-------------------
        n_samp = nt
        nfft = 1
        while nfft < n_samp:
            nfft <<= 1
        nfft <<= 1
        nfft_half = nfft // 2 + 1

        #----------------------------------------------
        # Prepare Observations (Three Branches)
        # - Precomputed HVSR
        # - Waveform(s) provided
        # - Fallback: synth from true model
        #-----------------------------------------------
        obs_surface_noisy = None
        obs_dev = None
        freq_vec_host = None
        freq_vec_dev = None
        hv_obs_dev = None
        hv_obs_host = None

        # Branch A: precomputed HVSR vector
        if obs_hvsr is not None and obs_freqs is not None:
            freq_vec_host = np.asarray(obs_freqs, dtype=float)
            freq_vec_dev = xp.asarray(freq_vec_host, dtype=float_dtype)
            hv_obs_host = np.asarray(obs_hvsr, dtype=float)
            # Normalization
            hv_obs_host = hv_obs_host / (np.median(hv_obs_host) + EPS)
            hv_obs_dev = xp.asarray(hv_obs_host, dtype=float_dtype)

        # Branch B: rawwaveform(s) provided
        elif obs_waveforms is not None:
            if obs_dt is None:
                raise ValueError("obs_dt (sampling interval in seconds) must be provided when obs_waveform is supplied.")
            obs_arr = np.asarray(obs_waveforms)
            fs = 1.0 / float(obs_dt)
            # Use Welch for stable PSD estimates
            if obs_arr.ndim == 1:
                f, Pzz = welch(obs_arr, fs=fs, nperseg=welch_nperseg or min(1024, obs_arr.size))
                hv_obs_host = np.sqrt(Pzz + TETHA)
                freq_vec_host = f
            elif obs_arr.ndim == 2 and obs_arr.shape[0] == 3:
                E, N, Z = obs_arr[0], obs_arr[1], obs_arr[2]
                f, Pxx_E = welch(E, fs=fs, nperseg=welch_nperseg or min(1024, E.size))
                _, Pxx_N = welch(N, fs=fs, nperseg=welch_nperseg or min(1024, N.size))
                _, Pzz = welch(Z, fs=fs, nperseg=welch_nperseg or min(1024, Z.size))
                Ph = 0.5 * (Pxx_E + Pxx_N)
                hv_obs_host = np.sqrt(Ph / (Pzz + TETHA))
                freq_vec_host = f
            else:
                raise ValueError("obs_waveforms must be 1D (T,) or 3xT (E,N,Z).")

            hv_obs_host = hv_obs_host * (1.0 + hv_noise_std * 0.05 * rng.standard_normal(size=hv_obs_host.shape))
            hv_obs_host = hv_obs_host / (np.median(hv_obs_host) + EPS)
            hv_obs_dev = xp.asarray(hv_obs_host, dtype=float_dtype)
            freq_vec_dev = xp.asarray(freq_vec_host, dtype=float_dtype)

            # Waveform misfit
            if abs(obs_dt - dt) < 1e-8:
                if obs_arr.ndim == 1:
                    obs_surface_noisy = obs_arr.astype(float)
                else:
                    obs_surface_noisy = obs_arr[2].astype(float)
                obs_dev = xp.asarray(obs_surface_noisy, dtype=float_dtype)
            else:
                print("[WARN] obs_dt != forward dt; waveform misfit will be disabled/ Use HVSR-only inversion or resample data.")

        # Branch C: Fallback to synth from true model
        else:
            print("[RUN] No obs provided -> synthesizing observed data from true model (fallback),")
            for (s, e), val in zip(layer_slices, layer_vs_true):
                vs_dev[s:e] = float(val)

            v_dev[:] = 0.0;
            tau_dev[:] = 0.0;
            u_dev[:] = 0.0

            # Forward
            surface_true_dev = self.Forward_SH_FDTD(src_time=src_time_dev, nt=nt, src_i=src_i,
                                                    vs=vs_dev, rho=rho_dev,
                                                    dz=dz, dt=dt, inplace=False,
                                                    return_state=False,
                                                    sanitize_interval=getattr(self, "sanitize_interval", None))
            obs_surface = to_host(surface_true_dev)

            obs_surface_noisy = obs_surface + waveform_noise_std * np.std(obs_surface) * rng.standard_normal(size=obs_surface.shape)
            obs_dev = xp.asarray(obs_surface_noisy, dtype=float_dtype)

            # Compute PSD on device
            win = xp.hanning(obs_dev.size).astype(float_dtype)
            pad = xp.zeros(nfft - obs_dev.size, dtype=float_dtype)
            x_in = xp.concatenate([obs_dev * win, pad])
            S_obs = xp.fft.rfft(x_in)
            PSD_obs = xp.abs(S_obs) ** 2 + TETHA
            fft_freqs = xp.fft.rfftfreq(nfft, dt)
            freq_mask = (fft_freqs > 0.2) & (fft_freqs <= 25.0)
            freq_vec_dev = fft_freqs[freq_mask]
            freq_vec_host = to_host(freq_vec_dev)

            # Evaluate vertical impedance on device for those freqs
            ImG_true_dev = self.Vertical_Impedance_Proxy(freq_vec_dev, layer_thickness, layer_vs_true,
                                                         layer_rho, vp_scale=VP_SCALE,
                                                         float_dtype=float_dtype,
                                                         complex_dtype=complex_dtype, eps=self.EPS)
            PSD_interp_dev = self.Linear_Interpolation(fft_freqs, PSD_obs, freq_vec_dev)
            hv_obs_dev = xp.sqrt(PSD_interp_dev / (ImG_true_dev + TETHA))
            med = float(xp.median(hv_obs_dev).item()) if is_cupy else float(np.median(to_host(hv_obs_dev)))
            hv_obs_dev = hv_obs_dev / (med + EPS)
            hv_obs_host = to_host(hv_obs_dev)

        if hv_obs_dev is None and ('hv_obs_host' in locals()):
            hv_obs_dev = xp.asarray(hv_obs_host, dtype=float_dtype)
            freq_vec_dev = xp.asarray(freq_vec_host, dtype=float_dtype)
        if freq_vec_dev is None and ('freq_vec_host' in locals()):
            freq_vec_dev = xp.asarray(freq_vec_host, dtype=float_dtype)

        if 'freq_vec_host' not in locals() and freq_vec_dev is not None:
            freq_vec_host = to_host(freq_vec_dev)

        #------------------------------------
        # Prior and initial guess (host)
        #------------------------------------
        prior_mean = np.array(layer_vs_true, dtype=float)
        prior_std = prior_mean * 0.25
        init = prior_mean * (1.0 + 0.1 * rng.standard_normal(size=prior_mean.shape))

        #-----------------------------------------
        # Precompute / Cache for MAP objective
        #-----------------------------------------
        self._posterior_cache = {}
        self._posterior_cache["n_samp"] = nt
        self._posterior_cache["nfft"] = nfft
        self._posterior_cache["dt"] = dt
        self._posterior_cache["freq_vec_dev"] = freq_vec_dev
        self._posterior_cache["layer_thickness"] = np.asarray(layer_thickness, dtype=float)
        self._posterior_cache["layer_rho"] = np.asarray(layer_rho, dtype=float)

        self._posterior_cache["win2"] = xp.hanning(nt).astype(float_dtype)

        self._posterior_cache["fft_freqs_dev"] = xp.fft.rfftfreq(nfft, dt)

        interp_idx, interp_alpha = self.Precompute_Linear_Interpolation_Weights(
            self._posterior_cache["fft_freqs_dev"],
            freq_vec_dev,
            xp=xp
        )
        self._posterior_cache["interp_idx"] = interp_idx
        self._posterior_cache["interp_alpha"] = interp_alpha

        self._posterior_cache["prior_mean_dev"] = xp.asarray(prior_mean, dtype=float_dtype)
        self._posterior_cache["prior_std_dev"] = xp.asarray(prior_std, dtype=float_dtype)

        if obs_surface_noisy is not None:
            self._posterior_cache["sigma_w_scaled"] = waveform_noise_std * (np.std(obs_surface_noisy) + EPS)
        else:
            self._posterior_cache["sigma_w_scaled"] = waveform_noise_std


        #-------------------------------------------------------------------
        # Preallocate working buffers (device) for Negative_Log_Posterior
        #-------------------------------------------------------------------
        vs_grid_dev_work = xp.ones(nz, dtype=float_dtype)
        v_work = xp.zeros(nz, dtype=float_dtype)
        tau_work = xp.zeros(max(1, nz - 1), dtype=float_dtype)
        u_work = xp.zeros(nz, dtype=float_dtype)
        mu_half_work = xp.empty(max(1, nz - 1), dtype=float_dtype)
        rho_inv_work = xp.asarray(rho_grid, dtype=float_dtype)

        self._posterior_cache["vs_grid_dev_work"] = vs_grid_dev_work
        self._posterior_cache["v_work"] = v_work
        self._posterior_cache["tau_work"] = tau_work
        self._posterior_cache["u_work"] = u_work

        # def Fill_Vs_Grid_Device_From_Parameters(params_host):
        #     last_val = float(params_host[-1])
        #     vs_grid_dev_work[:] = last_val
        #     for (s, e), val in zip(layer_slices, params_host):
        #         vs_grid_dev_work[s:e] = float(val)

        def Fill_Vs_Grid_Device_From_Parameters(params_in):
            params_dev_local = xp.asarray(params_in, dtype=float_dtype)
            last_val = params_dev_local[-1]
            vs_grid_dev_work[:] = last_val
            for (s,e), val in zip(layer_slices, params_dev_local):
                vs_grid_dev_work[s:e] = val

        #----------------------------------
        # Negative Log Posterior
        #----------------------------------
        """
        def Negative_Log_Posterior(
            params_host,
            waveform_noise_std_local,
            sigma_h_local,
            hv_weight_local,
            waveform_weight_local
        ):
            params_clamped = np.clip(params_host, 10.0, 10000.0)
            Fill_Vs_Grid_Device_From_Parameters(params_clamped)
            mu_half_work[:] = 0.5 * (rho_dev[:-1] * vs_grid_dev_work[:-1] * vs_grid_dev_work[:-1] +
                                     rho_dev[1:] * vs_grid_dev_work[1:] * vs_grid_dev_work[1:])
            v_work[:] = 0.0;
            tau_work[:] = 0.0;
            u_work[:] = 0.0

            # Forward
            surface_dev = self.Forward_SH_FDTD(src_time=src_time_dev, nt=nt, src_i=src_i,
                                               vs=vs_grid_dev_work, rho=rho_dev,
                                               dz=dz, dt=dt, inplace=False,
                                               return_state=False,
                                               sanitize_interval=getattr(self, "sanitize_interval", None))

            # Waveform Negative Log Likelihood
            # if (obs_dev is not None) and (waveform_weight_local > 0.0):
            #     synth_dev = surface_dev
            #     sigma_w_scaled = waveform_noise_std_local * (np.std(obs_surface_noisy) if obs_surface_noisy is not None else 1.0)
            #     nll_w = self.Waveform_Negative_Log_Likelihood(obs_dev, synth_dev, sigma_w_scaled, eps=EPS)
            #     nll_w_host = float(to_host(nll_w) if hasattr(nll_w, "__array__") or is_cupy else float(nll_w))
            #     nll_w_xp = xp.asarray(nll_w_host, dtype=float_dtype) if is_cupy else np.asarray(nll_w_host)
            # else:
            #     nll_w_xp = xp.asarray(0.0, dtype=float_dtype) if is_cupy else np.asarray(0.0)

            nll_w = xp.asarray(0.0, dtype=float_dtype)
            if (obs_dev is not None) and (waveform_weight_local > 0.0):
                nll_w = self.Waveform_Negative_Log_Likelihood(obs_dev, surface_dev, sigma_w_scaled, eps=EPS)

            # HVSR Negative Log
            # Compute PSD
            # win2 = xp.hanning(n_samp).astype(float_dtype)
            # pad2 = xp.zeros(nfft - n_samp, dtype=float_dtype)
            # x_in2 = xp.concatenate([surface_dev * win2, pad2])
            # S_synth = xp.fft.rfft(x_in2)

            win2 = xp.hanning(n_samp).astype(float_dtype)
            S_synth = xp.fft.rfft(surface_dev * win2, n=nfft)

            PSD_synth = xp.abs(S_synth) ** 2 + TETHA
            fft_freqs_dev = xp.fft.rfftfreq(nfft, dt)
            interp_idx, interp_alpha = self.Precompute_Linear_Interpolation_Weights(
                                            fft_freqs_dev,
                                            freq_vec_dev,
                                            xp=xp
                                        )

            # PSD_interp_dev2 = self.Linear_Interpolation(fft_freqs_dev, PSD_synth, freq_vec_dev)

            PSD_interp_dev2 = self.Interpolation_Linear_Precomputed(
                                    PSD_synth,
                                    interp_idx,
                                    interp_alpha,
                                    xp,
                                    dtype=float_dtype
                                )

            # Compute ImG for current parameters (device)
            ImG_dev = self.Vertical_Impedance_Proxy(freq_vec_dev, layer_thickness, params_clamped, layer_rho,
                                                    vp_scale=VP_SCALE, float_dtype=float_dtype,
                                                    complex_dtype=complex_dtype, eps=EPS)

            # hv_synth_dev = xp.sqrt(PSD_interp_dev2 / (ImG_dev + TETHA))
            # med2 = float(xp.median(hv_synth_dev).item()) if is_cupy else float(np.median(to_host(hv_synth_dev)))
            # hv_synth_dev = hv_synth_dev / (med2 + EPS)

            hv_synth_dev = xp.sqrt(PSD_interp_dev2 / (ImG_dev + TETHA))
            hv_synth_dev = hv_synth_dev / (xp.median(hv_synth_dev) + EPS)

            # HVSR misfit (device)
            nll_h = self.HVSR_Negative_Log_Likelihood(
                        hv_obs_dev,
                        hv_synth_dev,
                        sigma_h_local,
                        xp=xp,
                        eps=EPS
                    )
            # nll_h_host = float(to_host(nll_h)) if (is_cupy or hasattr(nll_h, "__array__")) else float(nll_h)
            # nll_h_xp = xp.asarray(nll_h_host, dtype=float_dtype) if is_cupy else np.asarray(nll_h_host)

            # Prior (device)
            params_dev = xp.asarray(params_clamped, dtype=float_dtype)
            prior_mean_dev = xp.asarray(prior_mean, dtype=float_dtype)
            prior_std_dev = xp.asarray(prior_std, dtype=float_dtype)
            nll_p = self.Gaussian_Prior_Negative_Log(
                        params_dev,
                        prior_mean_dev,
                        prior_std_dev,
                        eps=EPS
                    )
            # nll_p_host = float(to_host(nll_p)) if (is_cupy or hasattr(nll_p, "__array__")) else float(nll_p)
            # nll_p_xp = xp.asarray(nll_p_host, dtype=float_dtype) if is_cupy else np.asarray(nll_p_host)

            # Weighted sum (return host float)
            # neglog_total = float(to_host(waveform_weight_local * nll_w_xp + hv_weight_local * nll_h_xp + nll_p_xp))
            neglog_total = waveform_weight_local * nll_w + hv_weight_local * nll_h + nll_p
            neglog_total = xp.nan_to_num(neglog_total, nan=1e30, posinf=1e30, neginf=1e30)

            # Guard finite
            # if not np.isfinite(neglog_total):
            #     neglog_total = 1e30
            # return neglog_total

            return float(neglog_total.item() if hasattr(neglog_total, "item") else neglog_total)
        """

        layer_id = xp.zeros(nz, dtype=xp.int32)
        for i, (s,e) in enumerate(layer_slices):
            layer_id[s:e] = i

        params_dev_buffer = xp.empty(nlayer, dtype=float_dtype)

        fft_buffer = xp.zeros(nfft, dtype=float_dtype)

        cache = self._posterior_cache

        v_work = cache["v_work"]
        tau_work = cache["tau_work"]
        u_work = cache["u_work"]
        # vs_grid_dev_work = cache["vs_grid_dev_work"]

        def Negative_Log_Posterior(
            params_host,
            waveform_noise_std_local,
            sigma_h_local,
            hv_weight_local,
            waveform_weight_local
        ):
            xp_local = xp

            #=========================
            # PARAMETER
            #=========================
            xp.copyto(params_dev_buffer, xp.asarray(params_host, dtype=params_dev_buffer.dtype))
            # params_dev_tmp = xp.asarray(params_host, dtype=params_dev_buffer.dtype)
            # xp.copyto(params_dev_buffer, params_dev_tmp)
            params_dev = xp_local.clip(params_dev_buffer, 10.0, 10000.0)

            #=========================
            # VECTORIZE GRID MAPPING
            #=========================
            vs_grid_dev_work[:] = params_dev[layer_id]

            #=========================
            # RESET STATE (CHEAP)
            #=========================
            v_work.fill(0)
            tau_work.fill(0)
            u_work.fill(0)

            #=========================
            # FORWARD MODEL
            #=========================
            surface_dev = self.Forward_SH_FDTD(
                src_time=src_time_dev,
                nt=nt,
                src_i=src_i,
                vs=vs_grid_dev_work,
                rho=rho_dev,
                dz=dz,
                dt=dt,
                inplace=False,
                return_state=False,
                sanitize_interval=getattr(self, "sanitize_interval", None),
                xp=xp_local
            )

            #=========================
            # WAVEFORM MISFIT
            #=========================
            nll_w = xp_local.asarray(0.0, dtype=float_dtype)
            if (obs_dev is not None) and (waveform_weight_local > 0.0):
                nll_w = self.Waveform_Negative_Log_Likelihood(
                    obs_dev,
                    surface_dev,
                    cache["sigma_w_scaled"],
                    eps=EPS,
                    xp=xp_local
                )

            #=========================
            # FFT
            #=========================
            fft_buffer[:nt] = surface_dev * cache["win2"]
            fft_buffer[nt:] = 0

            S_synth = xp_local.fft.rfft(fft_buffer)
            PSD_synth = xp_local.abs(S_synth) ** 2 + TETHA

            PSD_interp_dev2 = self.Interpolation_Linear_Precomputed(
                PSD_synth,
                cache["interp_idx"],
                cache["interp_alpha"],
                xp_local,
                dtype=float_dtype
            )

            #=========================
            # IMPEDANCE
            #=========================
            ImG_dev = self.Vertical_Impedance_Proxy(
                cache["freq_vec_dev"],
                layer_thickness,
                params_dev,
                layer_rho,
                vp_scale=VP_SCALE,
                float_dtype=float_dtype,
                complex_dtype=complex_dtype,
                eps=EPS
            )

            #=========================
            # HVSR
            #=========================
            hv_synth_dev = xp_local.sqrt(PSD_interp_dev2 / (ImG_dev + TETHA))

            # 🔥 OPTIMASI BESAR (median → mean)
            hv_synth_dev /= (xp_local.mean(hv_synth_dev) + EPS)

            nll_h = self.HVSR_Negative_Log_Likelihood(
                hv_obs_dev,
                hv_synth_dev,
                sigma_h_local,
                xp=xp_local,
                eps=EPS
            )

            #=========================
            # PRIOR
            #=========================
            nll_p = self.Gaussian_Prior_Negative_Log(
                params_dev,
                cache["prior_mean_dev"],
                cache["prior_std_dev"],
                eps=EPS
            )

            #=========================
            # TOTAL
            #=========================
            neglog_total = waveform_weight_local * nll_w + hv_weight_local * nll_h + nll_p

            return xp_local.nan_to_num(
                neglog_total,
                nan=1e30,
                posinf=1e30,
                neginf=1e30
            )

        #---------------------------------------
        # Deterministic MAP (L-BFGS-B)
        #---------------------------------------
        if map_opts is None:
            map_opts = {"maxiter": 50,
                        "disp": True,
                        "ftol": 1e-3,
                        "gtol": 1e-3}
        bounds = [(10.0, 5000.0) for _ in range(nlayer)]
        print("[RUN] Running deterministic MAP (L-BFGS-B)...")

        # obj_wrapper = lambda p: Negative_Log_Posterior(p, waveform_noise_std, hv_noise_std, 1.0, 1.0)

        def obj_wrapper(p):
            val = Negative_Log_Posterior(
                    p,
                    waveform_noise_std,
                    hv_noise_std,
                    1.0,
                    1.0
                  )
            return float(val.get() if is_cupy else val)

        res = minimize(obj_wrapper, x0=init, method="L-BFGS-B", bounds=bounds, options=map_opts)
        map_params = res.x
        print("[MAP] success:", bool(res.success), "message:", res.message)
        print("[MAP] params:", map_params)

        # synth trace at MAP (host)
        Fill_Vs_Grid_Device_From_Parameters(map_params)
        mu_half_work[:] = 0.5 * (rho_dev[:-1] * vs_grid_dev_work[:-1] * vs_grid_dev_work[:-1] +
                                 rho_dev[1:] * vs_grid_dev_work[1:] * vs_grid_dev_work[1:])
        v_work[:] = 0.0;
        tau_work[:] = 0.0;
        u_work[:] = 0.0
        surface_map_dev = self.Forward_SH_FDTD(src_time=src_time_dev, nt=nt, src_i=src_i,
                                               vs=vs_grid_dev_work, rho=rho_dev,
                                               dz=dz, dt=dt, inplace=False,
                                               return_state=False,
                                               sanitize_interval=getattr(self, "sanitize_interval", None))
        synth_map_host = to_host(surface_map_dev)

        # HVSR at MAP (device)
        x_in3 = xp.concatenate(
            [xp.asarray(synth_map_host, dtype=float_dtype) * xp.hanning(synth_map_host.size).astype(float_dtype),
             xp.zeros(nfft - synth_map_host.size, dtype=float_dtype)])
        S_map = xp.fft.rfft(x_in3)
        PSD_map = xp.abs(S_map) ** 2 + TETHA
        PSD_map_interp = self.Linear_Interpolation(xp.fft.rfftfreq(nfft, dt), PSD_map, freq_vec_dev)
        ImG_true_dev = self.Vertical_Impedance_Proxy(freq_vec_dev, layer_thickness, map_params, layer_rho,
                                                     vp_scale=VP_SCALE,
                                                     float_dtype=float_dtype, complex_dtype=complex_dtype, eps=EPS)
        hv_map_dev = xp.sqrt(PSD_map_interp / (ImG_true_dev + TETHA))
        hv_map_dev = hv_map_dev / (float(xp.median(hv_map_dev).item()) + EPS)
        hv_map_host = to_host(hv_map_dev)

        #-----------------------------------------------
        # Local Markov-Chain Monte-Carlo around MAP
        #-----------------------------------------------
        sampler_fn = getattr(self, "Local_Markov_Chain_Monte_Carlo", None) or getattr(self, "Local_Metropolis_Hastings",
                                                                                      None)
        if sampler_fn is None:
            print("[WARN] No local sampler found on class; skipping MCMC sampling.")
            samples = np.empty((0, nlayer))
            accept_rate = 0.0
        else:
            if mh_opts is None:
                mh_opts = {"niter": N_ITER_DEFAULT,
                           "burnin": BURN_IN_DEFAULT,
                           "thin": THIN_DEFAULT,
                           "proposal_fraction": PROPOSAL_FRACTION_DEFAULT}
            proposal_scale = mh_opts["proposal_fraction"] * (1.0 * np.abs(map_params) + 1.0)
            print("[RUN] Running MCMC around MAP (GPU-aware)...")
            mh_out = sampler_fn(map_params, mh_opts["niter"], Negative_Log_Posterior,
                                args=(waveform_noise_std * (
                                    np.std(obs_surface_noisy) if obs_surface_noisy is not None else 1.0), hv_noise_std, 1.0, 1.0),
                                proposal_scale=proposal_scale, thin=mh_opts["thin"],
                                burnin=mh_opts["burnin"], xp=xp, dtype=dtype, seed=getattr(self, "seed", None),
                                verbose=True)
            samples = mh_out.get("samples", np.empty((0, nlayer)))
            accept_rate = mh_out.get("accept_rate", 0.0)

        #--------------------------------------
        # Posterior summary & cell profiles
        #--------------------------------------
        if getattr(samples, "size", 0):
            median_vs = np.median(samples, axis=0)
            lower_vs = np.percentile(samples, 5, axis=0)
            upper_vs = np.percentile(samples, 95, axis=0)
        else:
            median_vs = np.asarray(map_params, dtype=float)
            lower_vs = median_vs.copy()
            upper_vs = median_vs.copy()

        median_vs = np.asarray(median_vs, dtype=float)
        lower_vs = np.asarray(lower_vs, dtype=float)
        upper_vs = np.asarray(upper_vs, dtype=float)

        # Build full-cell profiles
        vs_grid_median = np.full(nz, float(median_vs[-1]), dtype=float)
        vs_grid_lower = np.full(nz, float(lower_vs[-1]), dtype=float)
        vs_grid_upper = np.full(nz, float(upper_vs[-1]), dtype=float)

        # vs_grid_median = Layer_Parameters_to_Grid(median_vs, nz, dz, layer_thickness)
        # vs_grid_lower = Layer_Parameters_to_Grid(p5, nz, dz, layer_thickness)
        # vs_grid_upper = Layer_Parameters_to_Grid(p95, nz, dz, layer_thickness)

        for (s, e), vm, vl, vu in zip(layer_slices, median_vs, lower_vs, upper_vs):
            s_i = max(0, int(s));
            e_i = min(nz, int(e))
            if e_i > s_i:
                vs_grid_median[s_i:e_i] = float(vm)
                vs_grid_lower[s_i:e_i] = float(vl)
                vs_grid_upper[s_i:e_i] = float(vu)

        # Save arrays
        np.save(os.path.join(out_dir, "vs_grid_median.npy"), vs_grid_median)
        np.save(os.path.join(out_dir, "vs_grid_lower.npy"), vs_grid_lower)
        np.save(os.path.join(out_dir, "vs_grid_upper.npy"), vs_grid_upper)
        np.savetxt(os.path.join(out_dir, "vs_grid_profile.csv"),
                   np.column_stack([z, vs_grid_median, vs_grid_lower, vs_grid_upper]),
                   header="depth(m),vs_median,vs_lower,vs_upper", delimiter=",")

        #--------------------------------------------------------
        # Plotting: profile, ensemble, HVSR, layer posteriors
        #--------------------------------------------------------
        if vs_grid_median is not None and z is not None:
            try:
                if len(vs_grid_median) != len(z):
                    old_idx = np.linspace(0.0, 1.0, len(vs_grid_median))
                    new_idx = np.linspace(0.0, 1.0, len(z))
                    vs_grid_median = np.interp(new_idx, old_idx, np.asarray(vs_grid_median, dtype=float))
                    if vs_grid_lower is not None:
                        vs_grid_lower = np.interp(new_idx, old_idx, np.asarray(vs_grid_lower, dtype=float))
                    if vs_grid_upper is not None:
                        vs_grid_upper = np.interp(new_idx, old_idx, np.asarray(vs_grid_upper, dtype=float))
                    if verbose:
                        print(f"[plot] resampled vs_grid_* from {len(old_idx)} -> {len(new_idx)} to match z")
            except Exception as e:
                if verbose:
                    print("[plot][WARN] failed to resample vs_grid to z:", e)

        try:
            plt.figure(figsize=(4, 8))
            plt.step(vs_grid_median, z, where='post', label='median Vs')
            plt.fill_betweenx(z, vs_grid_lower, vs_grid_upper, color='C0', alpha=0.25, label='Creedible Interval')
            layer_tops = np.concatenate([[0.0], np.cumsum(layer_thickness)[:-1]])
            plt.scatter(median_vs, layer_tops, marker='o', c='k', s=30, label='layer medians')
            plt.gca().invert_yaxis()
            plt.xlabel("Vs (m/s)");
            plt.ylabel("Depth (m)")
            plt.title("1D Vs profile (cell resolution)")
            plt.grid(True);
            plt.legend();
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "vs_profile_1d.png"), bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        """
        try:
            # Posterior layers
            plt.figure(figsize=(8, 4))
            x = np.arange(nlayer)
            plt.errorbar(x, median_vs, yerr=[median_vs - lower_vs, upper_vs - median_vs], fmt="o", capsize=4,
                         label="posterior (5-95%)")
            plt.plot(x, layer_vs_true, "k--", label="true Vs (layer)")
            plt.xlabel("layer");
            plt.ylabel("Vs (m/s)");
            plt.legend();
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, "posterior_layers.png"), bbox_inches="tight")
            plt.close()
        except Exception:
            pass
        """

        try:
            plt.figure(figsize=(8, 4))
            x = np.arange(nlayer)
            plt.errorbar(x, median_vs, yerr=[median_vs - lower_vs, upper_vs - median_vs], fmt="o", capsize=4,
                         label="Posterior (Credible Interval)")
            plt.plot(x, layer_vs_true, "k--", label="True Vs (layer)")
            plt.xlabel("Layer (m)");
            plt.ylabel("Vs (m/s)");
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, "posterior_layers.png"))

            plt.figure(figsize=(6, 4))
            plt.semilogy(freq_vec_host, (xp.asnumpy(hv_obs_dev) if is_cupy else np.asarray(hv_obs_dev)), label="hv_obs")
            plt.xlabel("Frequency (Hz)");
            plt.ylabel("HVSR");
            plt.legend();
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, "hv_map_vs_obs.png"), dpi=400)
        except Exception:
            pass

        """
        try:
            # HVSR comparison (host arrays)
            hv_plot = to_host(hv_obs_dev) if hv_obs_dev is not None else None
            if hv_plot is not None and freq_vec_host is not None:
                plt.figure(figsize=(6, 4))
                plt.semilogy(freq_vec_host, hv_plot, label="hv_obs")
                # overlay hv_map if available
                try:
                    plt.semilogy(freq_vec_host, hv_map_host, label="hv_map")
                except Exception:
                    pass
                plt.xlabel("Frequency (Hz)");
                plt.ylabel("HVSR");
                plt.legend();
                plt.grid(True)
                plt.savefig(os.path.join(out_dir, "hv_map_vs_obs.png"), bbox_inches="tight")
                plt.close()
        except Exception:
            pass
        """

        dz_plot = dz
        layers_thickness = layer_thickness

        if layers_thickness is None:
            raise RuntimeError("Cannot build Vs grid: 'layers' thickness array not found. Define `layers` or provide out['layer_thickness'].")

        total_depth = float(np.sum(layers_thickness))
        nz = int(round(total_depth / float(dz_plot)))
        nz = max(nz, 1)
        z = np.linspace(0.0, total_depth, nz, endpoint=False)

        def Layer_Parameters_to_Grid(params):
            grid = np.empty(nz, dtype=float)
            cum_idx = 0
            for i, th in enumerate(layers_thickness):
                n_cells = max(1, int(round(th / float(dz_plot))))
                end = min(nz, cum_idx + n_cells)
                val = float(params[i]) if i < len(params) else float(params[-1])
                grid[cum_idx:end] = val
                cum_idx = end
            if cum_idx < nz:
                grid[cum_idx:] = float(params[-1] if len(params) > 0 else 0.0)
            return grid

        if samples is not None and getattr(samples, "size", 0) > 0:
            samples = np.asarray(samples)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            med_layer = np.median(samples, axis=0)
            p5_layer = np.percentile(samples, 5, axis=0)
            p95_layer = np.percentile(samples, 95, axis=0)
        else:
            if map_params is None:
                raise RuntimeError("No samples and no map_params found in 'out'. Cannot plot ensemble.")
            med_layer = np.asarray(map_params)
            p5_layer = med_layer.copy()
            p95_layer = med_layer.copy()
            samples = None

        # Expand to full-cell grids
        med_grid = Layer_Parameters_to_Grid(med_layer)
        p5_grid = Layer_Parameters_to_Grid(p5_layer)
        p95_grid = Layer_Parameters_to_Grid(p95_layer)

        # Plotting
        plt.figure(figsize=(5, 10))
        ax = plt.gca()

        # Plot many sample profiles
        if samples is not None:
            nsamp = samples.shape[0]
            Nplot = min(300, nsamp)  # limit for speed/clarity
            rng = np.random.default_rng(0)
            idx = rng.choice(nsamp, size=Nplot, replace=False)
            for ii in idx:
                s_grid = Layer_Parameters_to_Grid(samples[ii])
                ax.step(s_grid, z, where='post', linewidth=0.6, alpha=0.06, color='tab:blue', zorder=1)

        ax.fill_betweenx(z, p5_grid, p95_grid, step='post', color='orange', alpha=0.25, label='Credible Interval', zorder=2)
        ax.step(med_grid, z, where='post', color='orange', linewidth=2.0, label='median', zorder=3)

        top_depths = []
        vals = []
        cum = 0.0
        for i, th in enumerate(layers_thickness):
            top_depths.append(cum)
            vals.append(float(med_layer[i] if i < len(med_layer) else med_layer[-1]))
            cum += float(th)
        ax.scatter(vals, top_depths, color='k', edgecolor='white', s=40, zorder=4, label='layer medians')

        ax.set_xlabel('Vs (m/s)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper right')
        plt.title('Ensemble of Posterior Profiles')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "vs_profile_ensemble.png"), dpi=400)

        png, pdf = self.Model_Section(out_dir=out_dir,
                                      samples=samples,
                                      layer_thickness=layer_thickness,
                                      median_vs=median_vs,
                                      vs_grid_median=vs_grid_median,
                                      vs_grid_lower=vs_grid_lower,
                                      vs_grid_upper=vs_grid_upper,
                                      dz=dz,
                                      nz=vs_grid_median,
                                      z=np.linspace(0.0, total_depth, nz, endpoint=False))

        #-------------------------------
        # Collect outputs
        #-------------------------------
        out = {
            "map_params": map_params,
            "samples": samples,
            "accept_rate": accept_rate,
            "median_vs": median_vs,
            "lower_vs": lower_vs,
            "upper_vs": upper_vs,
            "synth_map": synth_map_host,
            "hv_map": hv_map_host if 'hv_map_host' in locals() else None,
            "hv_obs": to_host(hv_obs_dev) if hv_obs_dev is not None else None,
            "freq_vec": freq_vec_host,
            "dz": dz,
            "dt": dt,
            "nt": nt,
            "z": z,
            "out_dir": out_dir,
            "res_optimizer": res,
            "vs_grid_median": vs_grid_median,
            "vs_grid_lower": vs_grid_lower,
            "vs_grid_upper": vs_grid_upper,
            "depth_cells": z,
            "layer_thickness": layer_thickness,
            "nlayer": len(layer_thickness),
        }

        # Metadata
        try:
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump({"use_gpu": bool(is_cupy), "map_success": bool(res.success), "accept_rate": float(accept_rate)}, f, indent=2)
        except Exception:
            pass
        return out


    def Make_HVSR_Wrapper_FWD(b, out, param_in_log=True):
        freq_vec = np.asarray(out['freq_vec'])
        layers_h = np.asarray(out.get('layer_thickness'))
        layers_rho = np.asarray(out.get('layer_rho'))
        tiny = getattr(b, "TETHA", 1e-12)

        def forward_hvsr_fn(params):
            params = np.asarray(params, dtype=float)
            if param_in_log:
                vs_layers = np.exp(params)
            else:
                vs_layers = params
            b.Fill_Vs_Grid_Device_From_Parameters(vs_layers)
            surf = b.Forward_SH_FDTD(src_time=b.src_time_dev, nt=b.nt, src_i=b.src_i,
                                     vs=b.vs_grid_dev_work, rho=b.rho_dev, dz=b.dz, dt=b.dt,
                                     inplace=False, return_state=False)
            freqs, hv = b.compute_hvsr_from_surface(surf, freq_vec)
            return freqs, hv

        return forward_hvsr_fn


















