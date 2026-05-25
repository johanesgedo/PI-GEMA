"""
All Rights Reserved

Copyright (c) 2026 Johanes Gedo Sea and Irfan Said

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

import os
import time
import types
import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    HAS_CUPY_FOR_PROF = True
except Exception:
    cp = None
    HAS_CUPY_FOR_PROF = False


class Elastic_Reverse_Time_Migration_2D_Profiler:

    DEFAULT_TARGETS = [
        "Elastic_TimeDomain_Propagation",
        "Apply_1D_Shifted_Convolution",
        "Absorbing_Boundary_Condition",
        "Cross_Correlation",
        "Image_Gaussian",
        "Preconditioning",
        "Elastic_Denoise",
        "Ricker_Wavelet",
        "Image_Denoise_Normalize_Cross_Correlation"
    ]

    def __init__(self, owner=None, enable=True, verbose=True, enable_by_default=None):
        if enable_by_default is not None:
            enable = bool(enable_by_default)
        self.owner = owner
        self.enable = enable
        # legacy alias: some code used self.enabled
        self.enabled = enable
        self.verbose = verbose
        self.samples = {}
        self.targets = list(self.DEFAULT_TARGETS)
        self.device_props = {}
        self.peak_flops_gflop_s = None
        self.mem_bandwidth_gb_s = None
        try:
            self.Probe_Device_Props()
        except Exception:
            pass
        # convenience aliases
        self._probe_device_props = self.Probe_Device_Props
        self.estimate = self.Estimate
        self.aggregate = self.Aggregate
        self.wrap_methods = self.Wrap_Methods
        self.add_sample = self.Add_Sample
        self.report = self.Report

    #-----------------
    # Device probing
    #-----------------
    def Probe_Device_Props(self):
        try:
            if HAS_CUPY_FOR_PROF and hasattr(cp, "cuda") and hasattr(cp.cuda, "runtime"):
                dev_id = 0
                if self.owner is not None and hasattr(self.owner, "device"):
                    try:
                        dev_id = int(getattr(self.owner, "device"))
                    except Exception:
                        dev_id = 0
                props = cp.cuda.runtime.getDeviceProperties(dev_id)
                # props may be dict-like or object with attributes; try both
                def g(k, default=0.0):
                    try:
                        return float(getattr(props, k))
                    except Exception:
                        try:
                            return float(props.get(k, default))
                        except Exception:
                            return float(default)
                mem_clock_khz = g('memoryClockRate', 0.0)
                bus_width = g('memoryBusWidth', 0.0)
                if mem_clock_khz and bus_width:
                    mem_clock_hz = mem_clock_khz * 1000.0
                    data_rate = 2.0
                    bandwidth_bps = mem_clock_hz * (bus_width / 8.0) * data_rate
                    self.mem_bandwidth_gb_s = float(bandwidth_bps) / 1e9
                multi_proc = int(g('multiProcessorCount', 0))
                major = int(g('major', 0))
                cores_per_sm_map = {8: 128, 7: 64, 6: 64, 5: 128}
                cores_per_sm = cores_per_sm_map.get(major, 64)
                cuda_cores = multi_proc * cores_per_sm
                clock_khz = g('clockRate', 0.0)
                if cuda_cores and clock_khz:
                    clock_hz = clock_khz * 1000.0
                    peak_flops = cuda_cores * clock_hz * 2.0
                    self.peak_flops_gflop_s = float(peak_flops) / 1e9
        except Exception:
            pass

    def Enable(self):
        self.enabled = True
        self.enable = True

    def Disable(self):
        self.enabled = False
        self.enable = False


    #------------------------------
    # Wrap methods for profiling
    #------------------------------
    """
    def Wrap_Methods(self, obj, method_names=None):
        if method_names is None:
            method_names = self.targets

        for name in method_names:
            if not hasattr(obj, name):
                continue
            orig = getattr(obj.__class__, name)
            if not callable(orig):
                continue

            def Make_Wrapper(mname, mfunc):
                def Wrapper(*args, __mfunc=mfunc, __mname=mname, **kwargs):
                    try:
                        est_flops, est_bytes, gridpoints = self.estimate(__mname, args, kwargs)
                    except Exception:
                        est_flops, est_bytes, gridpoints = 0.0, 0.0, 0

                    time_s = None
                    owner_has_cupy = bool(getattr(self.owner, "has_cupy", False))
                    if HAS_CUPY_FOR_PROF and hasattr(cp, "cuda") and owner_has_cupy:
                        try:
                            s = cp.cuda.Event(); e = cp.cuda.Event()
                            s.record()
                            res = __mfunc(*args, **kwargs)
                            e.record()
                            e.synchronize()
                            time_ms = cp.cuda.get_elapsed_time(s, e)
                            time_s = float(time_ms) / 1000.0
                        except Exception:
                            t0 = time.time()
                            res = __mfunc(*args, **kwargs)
                            time_s = time.time() - t0
                    else:
                        t0 = time.time()
                        res = __mfunc(*args, **kwargs)
                        time_s = time.time() - t0

                    try:
                        self.add_sample(__mname, est_flops, est_bytes, time_s, gridpoints)
                    except Exception:
                        pass
                    return res
                return Wrapper

            wrapped = Make_Wrapper(name, orig)
            try:
                bound = types.MethodType(wrapped, obj)
                setattr(obj, name, bound)
            except Exception:
                setattr(obj, name, wrapped)
    """


    def Wrap_Methods(self, obj, method_names=None):
        if method_names  is None:
            method_names = self.targets

        for name in method_names:
            orig = None
            try:
                orig = getattr(obj.__class__, name)
            except Exception:
                orig = None
            if orig is None:
                orig = getattr(obj, name, None)

            if not callable(orig):
                if self.verbose:
                    try:
                        self.log("[Profiler] skipping wrap - no method", name)
                    except Exception:
                        print("[Profiler] skipping wrap - no method", name)
                continue

            def Make_Wrapper(mname, mfunc):
                def Wrapper(*args, __mfunc=mfunc, __mname=mname, **kwargs):
                    try:
                        est_flops, est_bytes, gridpoints = self.estimate(__mname, args, kwargs)
                    except Exception:
                        est_flops, est_bytes, gridpoints = 0.0, 0.0, 0

                    time_s = None
                    owner_has_cupy = bool(getattr(self.owner, "has_cupy", False))
                    if HAS_CUPY_FOR_PROF and hasattr(cp, "cuda") and owner_has_cupy:
                        try:
                            s = cp.cuda.Event()
                            e = cp.cuda.Event()
                            s.record()
                            e.synchronize()
                            time_ms = cp.cuda.get_elapsed_time(s,e)
                            time_s = float(time_ms) / 1000.0
                        except Exception:
                            t0 = time.time()
                            res = __mfunc(*args, **kwargs)
                            time_s = time.time() - t0
                    else:
                        t0 = time.time()
                        res = __mfunc(*args, **kwargs)
                        time_s = time.time() - t0

                    try:
                        self.add_sample(__mname, est_flops, est_bytes, time_s, gridpoints)
                    except Exception:
                        pass
                    return res
                return Wrapper

            wrapped = Make_Wrapper(name, orig)
            try:
                bound = types.MethodType(wrapped, obj)
                setattr(obj, name, bound)
            except Exception:
                setattr(obj, name, wrapped)


    def Add_Sample(self, name, flops, bytes, time_s, gridpoints):
        if name not in self.samples:
            self.samples[name] = []
        self.samples[name].append({
            "flops": float(flops),
            "bytes": float(bytes),
            "time_s": float(time_s),
            "gridpoints": int(gridpoints)
        })


    #---------------------------
    # Estimation Heuristics
    #---------------------------
    def Estimate(self, name, args, kwargs):
        est_flops = 0.0
        est_bytes = 0.0
        gridpoints = 0

        def Find_2D_Array(a_list):
            for a in a_list:
                try:
                    import cupy as cpx
                    is_cp = (HAS_CUPY_FOR_PROF and isinstance(a, cpx.ndarray))
                except Exception:
                    is_cp = False
                if hasattr(a, "ndim") and getattr(a, "ndim", 0) == 2:
                    shp = tuple(a.shape)
                    return shp, is_cp
            return None, False

        argvals = list(args) + list(kwargs.values())

        # Elastic_TimeDomain_Propagation
        if name == "Elastic_TimeDomain_Propagation":
            shp, _ = Find_2D_Array(argvals)
            if shp is None:
                nx = int(kwargs.get("nx", 0) or (args[8] if len(args) > 8 else 0))
                nz = int(kwargs.get("nz", 0) or (args[10] if len(args) > 10 else 0))
            else:
                nx = int(shp[0]); nz = int(shp[1])
            gridpoints = nx * nz

            R = int(kwargs.get("fd_order_radius", 3))
            k = 2 * R + 1
            conv_flops_per_conv = 2.0 * k * gridpoints
            est_flops = 4.0 * conv_flops_per_conv + 30.0 * gridpoints
            est_bytes = gridpoints * 4.0 * 8.0
            return est_flops, est_bytes, gridpoints

        # Apply_1D_Shifted_Convolution
        if name == "Apply_1D_Shifted_Convolution":
            shp, _ = Find_2D_Array(argvals)
            kernel = None
            for v in argvals:
                if hasattr(v, "__len__") and not hasattr(v, "ndim"):
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 and all(np.isscalar(x) for x in v):
                        kernel = v
                        break
                    try:
                        import cupy as cpz
                        if isinstance(v, cpz.ndarray):
                            kernel = v
                            break
                    except Exception:
                        pass
            if shp is None:
                nx = int(kwargs.get("nx", 0))
                nz = int(kwargs.get("nz", 0))
            else:
                nx, nz = shp
            gridpoints = nx * nz
            k = (len(kernel) if kernel is not None else 3)
            est_flops = 2.0 * k * gridpoints
            est_bytes = gridpoints * 4.0 * 2.0
            return est_flops, est_bytes, gridpoints

        # Cross_Correlation
        if name == "Cross_Correlation":
            shp, _ = Find_2D_Array(argvals)
            if shp is None:
                gridpoints = int(kwargs.get("nx", 0) * kwargs.get("nz", 0) or 0)
            else:
                gridpoints = shp[0] * shp[1]
            est_flops = 2.0 * gridpoints
            est_bytes = gridpoints * 4.0 * 3.0
            return est_flops, est_bytes, gridpoints

        # Image Gaussian
        if name == "Image_Gaussian":
            shp, _ = Find_2D_Array(argvals)
            kernel = None
            for v in argvals:
                if hasattr(v, "__len__") and not hasattr(v, "ndim"):
                    kernel = v
                    break
                try:
                    import cupy as cpz
                    if isinstance(v, cpz.ndarray) and v.ndim == 1:
                        kernel = v
                        break
                except Exception:
                    pass
            if shp is None:
                nx = int(kwargs.get("nx", 0)); nz = int(kwargs.get("nz", 0))
                gridpoints = nx * nz
            else:
                gridpoints = shp[0] * shp[1]
            k = (len(kernel) if kernel is not None else int(max(3, math.ceil(3.0 * kwargs.get("sigma", 1.5)))))
            est_flops = 4.0 * k * gridpoints
            est_bytes = gridpoints * 4.0 * 4.0
            return est_flops, est_bytes, gridpoints

        # Absorbing Boundary Condition
        if name == "Absorbing_Boundary_Condition":
            nx = int(kwargs.get("nx", args[0] if len(args) > 0 else 0))
            nz = int(kwargs.get("nz", args[2] if len(args) > 2 else 0))
            dt = float(kwargs.get("dt", args[4] if len(args) > 4 else 0.0))
            nbc = max(1, min(40, min(max(1, nx), nz) // 10))
            gridpoints = nx * nz
            boundary_ops = nbc * (nx + nz)
            est_flops = 50.0 * float(boundary_ops)
            est_bytes = float(boundary_ops) * 4.0 * 4.0
            return est_flops, est_bytes, gridpoints

        # Preconditioning
        if name == "Preconditioning":
            shp, _ = Find_2D_Array(argvals)
            if shp is None:
                nx = int(kwargs.get("nx", 0)); nz = int(kwargs.get("nz", 0))
                gridpoints = nx * nz
            else:
                gridpoints = shp[0] * shp[1]
            est_flops = 10.0 * gridpoints
            est_bytes = gridpoints * 4.0 * 4.0
            return est_flops, est_bytes, gridpoints

        # Elastic Denoise
        if name == "Elastic_Denoise" or name == "Image_Denoise_Normalize_Cross_Correlation":
            shp, _ = Find_2D_Array(argvals)
            if shp is None:
                gridpoints = int(kwargs.get("nx", 0) * kwargs.get("nz", 0) or 0)
            else:
                gridpoints = shp[0] * shp[1]
            est_flops = 50.0 * gridpoints
            est_bytes = gridpoints * 4.0 * 8.0
            return est_flops, est_bytes, gridpoints

        # Ricker Wavelet
        if name == "Ricker_Wavelet":
            nt = int(kwargs.get("nt", args[2] if len(args) > 2 else 0))
            est_flops = 10.0 * max(1, nt)
            est_bytes = max(1, nt) * 4.0
            gridpoints = nt
            return est_flops, est_bytes, gridpoints

        # Fallback
        shp, _ = Find_2D_Array(argvals)
        if shp is not None:
            gridpoints = int(shp[0] * shp[1])
            est_flops = 5.0 * gridpoints
            est_bytes = gridpoints * 4.0 * 3.0
        return est_flops, est_bytes, gridpoints


    #----------------------------
    # Benchmarking helpers
    #----------------------------
    def _default_setup_for_target(self, name, nx, nz, tile=None):
        xp = cp if (HAS_CUPY_FOR_PROF and self.owner and getattr(self.owner, "has_cupy", False)) else np
        grid_shape = (nx, nz)
        if name == "Elastic_TimeDomain_Propagation":
            vx = xp.zeros(grid_shape, dtype=xp.float32)
            vz = xp.zeros_like(vx)
            sxx = xp.zeros_like(vx)
            szz = xp.zeros_like(vx)
            sxz = xp.zeros_like(vx)
            rho = xp.ones_like(vx) * 2000.0
            lam = xp.ones_like(vx) * 1e9
            mu = xp.ones_like(vx) * 3e8
            args = (vx, vz, sxx, szz, sxz, rho, lam, mu)
            kwargs = {"nx": nx, "dx": getattr(self.owner, "DX", 2), "nz": nz, "dz": getattr(self.owner, "DZ", 2),
                      "dt": 0.001, "fd_order_radius": 3, "use_gpu": True, "inplace": True}
            return args, kwargs, nx * nz

        if name == "Apply_1D_Shifted_Convolution":
            arr = xp.random.randn(nx, nz).astype(xp.float32)
            kernel = xp.array([0.25, 0.5, 0.25], dtype=xp.float32)
            args = (arr, kernel)
            kwargs = {"axis": 0, "pad_mode": "reflect"}
            return args, kwargs, nx * nz

        if name == "Absorbing_Boundary_Condition":
            dvv = xp.ones(grid_shape, dtype=xp.float32) * 1500.0
            p = xp.zeros(grid_shape, dtype=xp.float32)
            pm = p.copy()
            pp = p.copy()
            args = (nx, getattr(self.owner, "DX", 2), nz, getattr(self.owner, "DZ", 2), 0.001, dvv, None, p, pm, pp, (1,1,1,1))
            kwargs = {}
            return args, kwargs, nx * nz

        if name == "Cross_Correlation":
            a = xp.random.randn(nx, nz).astype(xp.float32)
            b = xp.random.randn(nx, nz).astype(xp.float32)
            args = (a, b)
            kwargs = {"image": None, "use_gpu": True, "accumulate": False}
            if tile is not None:
                kwargs["tile"] = tile
            return args, kwargs, nx * nz

        if name == "Image_Gaussian":
            a = xp.random.randn(nx, nz).astype(xp.float32)
            b = xp.zeros_like(a)
            args = (a, b)
            kwargs = {"sigma": getattr(self.owner, "SIGMA", 1.5), "truncate": getattr(self.owner, "TRUNCATE", 3.0),
                      "mode": "reflect", "use_gpu": True}
            return args, kwargs, nx * nz

        if name == "Preconditioning":
            arr = xp.random.randn(nx, nz).astype(xp.float32)
            args = (arr, nx, 0, nz)
            kwargs = {"use_gpu": True}
            return args, kwargs, nx * nz

        if name == "Elastic_Denoise" or name == "Image_Denoise_Normalize_Cross_Correlation":
            vx = xp.random.randn(nx, nz).astype(xp.float32)
            vz = xp.random.randn(nx, nz).astype(xp.float32)
            sxx = xp.random.randn(nx, nz).astype(xp.float32)
            szz = xp.random.randn(nx, nz).astype(xp.float32)
            sxz = xp.random.randn(nx, nz).astype(xp.float32)
            args = (vx, vz, sxx, szz, sxz)
            kwargs = {"nx": nx, "nz": nz, "use_gpu": True}
            return args, kwargs, nx * nz

        if name == "Ricker_Wavelet":
            nt = max(16, int(math.sqrt(nx * nz) // 4))
            args = (25.0, 0.001, nt)
            kwargs = {"delay_cycles": 2.0}
            return args, kwargs, nt

        a = xp.random.randn(nx, nz).astype(xp.float32)
        args = (a,)
        kwargs = {}
        return args, kwargs, nx * nz


    def benchmark_target(
        self,
        target_name,
        grid_list=None,
        repeats=10,
        tiles=None,
        warmups=2,
        save_prefix=None
    ):
        if not self.enable:
            raise RuntimeError("Profiler disabled")

        if grid_list is None:
            grid_list = [(256, 256), (384, 384), (512, 512), (640, 640)]

        owner = self.owner
        if owner is None:
            raise RuntimeError("Profiler requires owner instance reference")

        results = {}
        for (nx, nz) in grid_list:
            tile_configs = tiles or [None]
            for tile in tile_configs:
                try:
                    args, kwargs, gridpoints = self._default_setup_for_target(target_name, nx, nz, tile=tile)
                except Exception as e:
                    if self.verbose:
                        print("[Profiler] setup for", target_name, "failed:", e)
                    continue

                try:
                    est_flops, est_bytes, _ = self.estimate(target_name, args, kwargs)
                except Exception:
                    est_flops, est_bytes = 0.0, 0.0

                for _ in range(max(1, warmups)):
                    try:
                        getattr(owner, target_name)(*args, **kwargs)
                    except Exception:
                        pass

                time_s = None
                try:
                    owner_has_cupy = bool(getattr(owner, "has_cupy", False))
                    if HAS_CUPY_FOR_PROF and cp is not None and owner_has_cupy:
                        s = cp.cuda.Event(); e = cp.cuda.Event()
                        s.record()
                        for _ in range(max(1, repeats)):
                            getattr(owner, target_name)(*args, **kwargs)
                        e.record()
                        e.synchronize()
                        time_ms = cp.cuda.get_elapsed_time(s, e)
                        time_s = float(time_ms) / 1000.0
                    else:
                        t0 = time.time()
                        for _ in range(max(1, repeats)):
                            getattr(owner, target_name)(*args, **kwargs)
                        time_s = time.time() - t0
                except Exception:
                    t0 = time.time()
                    try:
                        getattr(owner, target_name)(*args, **kwargs)
                    except Exception:
                        pass
                    time_s = time.time() - t0

                throughput_mpix_s = (gridpoints * max(1, repeats) / time_s) / 1e6 if time_s and time_s > 0 else 0.0

                self.add_sample(target_name, est_flops * max(1, repeats), est_bytes * max(1, repeats), time_s, gridpoints * max(1, repeats))

                key = f"{nx}x{nz}" + (f"_tile{tile[0]}x{tile[1]}" if tile is not None else "")
                results[key] = {
                    "nx": nx, "nz": nz, "tile": tile,
                    "time_s": time_s, "throughput_mpix_s": throughput_mpix_s,
                    "est_flops": est_flops, "est_bytes": est_bytes, "gridpoints": gridpoints
                }

                if self.verbose:
                    tstr = f"{target_name} grid={nx}x{nz} tile={tile} -> {throughput_mpix_s:.1f} MPix/s (time={time_s:.4f}s)"
                    try:
                        self.log("[Profiler]", tstr)
                    except Exception:
                        print("[Profiler]", tstr)

        if save_prefix:
            try:
                np.savez_compressed(f"{save_prefix}_{target_name}.npz", **results)
            except Exception:
                pass

        return results


    def run_benchmarks(
        self,
        targets=None,
        grid_list=None,
        repeats=10,
        tiles=None,
        save_prefix=None
    ):
        if targets is None:
            targets = list(self.targets)
        out = {}
        for t in targets:
            try:
                out[t] = self.benchmark_target(t, grid_list=grid_list, repeats=repeats, tiles=tiles, save_prefix=save_prefix)
            except Exception as e:
                if self.verbose:
                    try:
                        self.log(f"[Profiler] benchmark {t} failed:", e)
                    except Exception:
                        print(f"[Profiler] benchmark {t} failed:", e)
                out[t] = {}
        return out


    def plot_throughput_comparison(
        self,
        results_dict,
        targets=None,
        logscale=False,
        title="Cross Correlation Kernel Performance",
        save_path=None
    ):
        if not results_dict:
            print("[plot] results_dict is empty!")
            return False

        if targets is None:
            targets = list(results_dict.keys())

        plt.figure(figsize=(10,6))
        plotted = False
        for t in targets:
            res = results_dict.get(t, {})
            if not res:
                if self.verbose:
                    print("[plot] no data for target", t)
                continue
            groups = {}
            for key, entry in res.items():
                if not entry:
                    continue
                tile = entry.get("tile")
                grp_key = f"{t}_tile{tile[0]}x{tile[1]}" if tile is not None else t
                groups.setdefault(grp_key, []).append((entry["nx"] * entry["nz"], entry["throughput_mpix_s"]))

            for gkey, pts in groups.items():
                if not pts:
                    continue
                pts_sorted = sorted(pts, key=lambda x: x[0])
                xs = np.array([p[0] for p in pts_sorted])
                ys = np.array([p[1] for p in pts_sorted])

                if self.verbose:
                    print("[plot] plotting", gkey, "xs:", xs, "ys:", ys)

                plt.plot(xs, ys, marker='o', label=gkey.replace("_", " "))
                plotted = True

        if not plotted:
            print("[plot] Nothing plotted.")
            return False

        plt.xlabel("Total Grid Points (nx × nz)")
        plt.ylabel("Throughput (MPixel/s)")
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        if logscale:
            plt.xscale("log"); plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400)
            print("[plot] Saved to", save_path)
        else:
            plt.savefig("debug_plot.png", dpi=400)
            print("[plot] Saved to debug_plot.png")
        plt.close()
        return True


    def benchmark_all_and_plot(
        self,
        targets=None,
        grid_list=None,
        tiles=None,
        repeats=10,
        warmups=2,
        save_prefix="benchmark",
        show_combined=True,
        per_target_plots=True
    ):

        if targets is None:
            targets = list(self.targets)

        if grid_list is None:
            grid_list = [(512, 512), (576, 576), (624, 624)]

        if tiles is None:
            tiles = [(16, 16), (32, 8)]

        results = self.run_benchmarks(targets=targets, grid_list=grid_list, repeats=repeats, tiles=tiles, save_prefix=save_prefix)

        if show_combined:
            combined_path = f"{save_prefix}_combined.png"
            try:
                self.plot_throughput_comparison(results, targets=list(results.keys()), logscale=False,
                                                title="2D Kernel Throughput Comparison", save_path=combined_path)
                if self.verbose:
                    print(f"[benchmark_all_and_plot] Saved combined plot -> {combined_path}")
            except Exception as e:
                print("[benchmark_all_and_plot] Failed to create combined plot:", e)

        if per_target_plots:
            for t in targets:
                try:
                    per = {t: results.get(t, {})}
                    ppath = f"{save_prefix}_{t}.png"
                    self.plot_throughput_comparison(per, targets=[t], logscale=False, title=f"{t} Performance", save_path=ppath)
                    if self.verbose:
                        print(f"[benchmark_all_and_plot] Saved per-target plot -> {ppath}")
                except Exception as e:
                    print(f"[benchmark_all_and_plot] Failed per-target plot for {t}:", e)

        return results


    def Aggregate(self):
        agg = {}
        for name, samples in self.samples.items():
            total_flops = sum(s["flops"] for s in samples)
            total_bytes = sum(s["bytes"] for s in samples)
            total_time = sum(s["time_s"] for s in samples)
            total_grid = sum(s["gridpoints"] for s in samples)
            avg_ai = (total_flops / total_bytes) if total_bytes > 0 else float("nan")
            achieved_gflops = (total_flops / total_time) / 1e9 if total_time > 0 else 0.0
            throughput_mpix_s = (total_grid / total_time) / 1e6 if total_time > 0 else 0.0
            agg[name] = {
                "total_flops": total_flops,
                "total_bytes": total_bytes,
                "total_time_s": total_time,
                "avg_ai": avg_ai,
                "achieved_gflops": achieved_gflops,
                "throughput_mpix_s": throughput_mpix_s,
                "total_gridpoints": total_grid,
                "calls": len(samples)
            }
        return agg

    def Report(self, show_plots=True, save_prefix="rtm_profile"):
        agg = self.aggregate()
        if not agg:
            print("[Elastic_RTM_2D_Profiler] No samples collected.")
            return agg
        print("\n[Elastic_RTM_2D_Profiler] Summary:")
        print("{:40s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "kernel", "calls", "time(s)", "GFLOPs", "GBytes", "GF/s"
        ))
        for name, v in sorted(agg.items(), key=lambda kv: kv[1]["total_time_s"], reverse=True):
            gf = v["total_flops"] / 1e9
            gb = v["total_bytes"] / 1e9
            gfs = v["achieved_gflops"]
            print("{:40s} {:8d} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
                name, v["calls"], v["total_time_s"], gf, gb, gfs
            ))

        if show_plots:
            try:
                pass
            except Exception:
                pass

        return agg


def Init_Profiler_in_ERTM2D(self, enable=True):
    self.profiler = Elastic_Reverse_Time_Migration_2D_Profiler(self, enable=enable)
    try:
        self.profiler.wrap_methods(self, method_names=self.profiler.targets)
    except Exception:
        pass
    if getattr(self, "verbose", False):
        try:
            self.log("[Elastic_RTM_2D_Profiler] initialized and wrapped methods:", ",".join(self.profiler.targets))
        except Exception:
            print("[Elastic_RTM_2D_Profiler] initialized and wrapped methods:", ",".join(self.profiler.targets))














