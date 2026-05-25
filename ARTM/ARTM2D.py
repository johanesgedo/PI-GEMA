"""
All Rights Reserved

Copyright (c) 2026 Johanes Gedo Sea, Ardiansyah, and Fifiyatma

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
import math
import numpy as np
import logging

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.ndimage import convolve1d as cp_convolve1d
    HAS_CUPY = True
except Exception:
    cp = np
    cp_convolve1d = None
    HAS_CUPY = False

from scipy.ndimage import gaussian_filter as sp_gaussian_filter
from scipy.ndimage import convolve1d as sp_convolve1d
from scipy.sparse.linalg import LinearOperator, gmres
from tqdm import tqdm
from ERTM2D_2_Profiler_2 import Init_Profiler_in_ERTM2D


DX = 10.0
DZ = 10.0
SIGMA = 1.5
LOCAL_SIGMA = 3.0
EPS = 1e-9
GAMMA = 0.94
STRENGTH = 1.0
LXPAD = 80
RXPAD = 80
STRIDE = 4
RANDWIDTH = 0
SOURCE_DEPTH = 5
RECEIVER_DEPTH = 0
RANDOM_BOUNDARY_INPUT = 0
SIGMA = 2.0
TRUNCATE = 3.0
MAXITER = 50
HALF_PRECON = 4
EPS_PRECON = 1e-6
SCALE_PRECON = 1.0
WMIN_PRECON = 0.2
WMAX_PRECON = 5.0
NSHOT = 10
FREQUENCY0 = 25.0



class Acoustic_Reverse_Time_Migration_2D:

    def _init_logger(self):
        logger = logging.getLogger(f"Acoustic_RTM[{id(self)}]")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            fh = logging.FileHandler("log_running.txt", mode="a")
            fh.setLevel(logging.DEBUG)
            fh_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(fh_formatter)
            logger.addHandler(fh)
            if self.verbose:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                ch_formatter = logging.Formatter("[Acoustic_RTM] %(message)s")
                ch.setFormatter(ch_formatter)
                logger.addHandler(ch)
        return logger


    def __init__(
        self,
        device: int = 0,
        verbose: bool = False,
        enable_stream: bool = True,
        fd_cache: dict | None = None,
        register_kernels: bool = True
    ):
        self.device = int(device)
        self.verbose = bool(verbose)

        if fd_cache is None:
            self._fd_cache = {}
        else:
            self._fd_cache = fd_cache
        if "FD_WEIGHTS_CACHE" not in globals():
            globals()["FD_WEIGHTS_CACHE"] = self._fd_cache
        else:
            if globals()["FD_WEIGHTS_CACHE"] is not self._fd_cache:
                globals()["FD_WEIGHTS_CACHE"] = self._fd_cache

        self.has_cupy = HAS_CUPY
        if self.has_cupy:
            try:
                self.gpu_device = int(self.device)
                if enable_stream:
                    try:
                        self.stream = cp.cuda.Stream(non_blocking=True)
                    except Exception:
                        self.stream = None
                else:
                    self.stream = None
            except Exception:
                self.gpu_device = 0
                self.stream = None
        else:
            self.gpu_device = None
            self.stream = None

        self._logger = self._init_logger()

        def _log(*args):
            msg = " ".join(str(a) for a in args)
            self._logger.info(msg)

        self.log = _log

        if register_kernels:
            def _time_domain_propagation_fallback(*args, **kwargs):
                return self.Acoustic_TimeDomain_Propagation(*args, **kwargs)

            self.k = {
                "Acoustic_TimeDomain_Propagation": self.Acoustic_TimeDomain_Propagation,
                "absorbgpuup": self.Absorbgpuup,
                "Apply_1D_Shifted_Convolution": self.Apply_1D_Shifted_Convolution,
                "Cross_Correlation": self.Cross_Correlation,
                "Laplace_Denoise": self.Laplace_Denoise,
                "Image_Gaussian": self.Image_Gaussian,
                "FD_Weights_Fornberg": type(self).FD_Weights_Fornberg,
                "Get_FD_Weights": type(self).Get_FD_Weights,
                "Ricker_Wavelet": self.Ricker_Wavelet,
                "Make_FD_First_Derivative_Weights": type(self).Make_FD_First_Derivative_Weights,
                # "TimeDomain_Propagation": self.TimeDomain_Propagation_via_Helmholtz,
                "Helmholtz_Acoustic_Propagation": self.Helmholtz_Acoustic_Propagation,
                # "receivedata": _default_receivedata,
            }

        # Add Profiler
        try:
            Init_Profiler_in_ERTM2D(self, enable=True)
        except Exception as e:
            if self.verbose:
                self.log("[Acoustic_RTM_2D_Profiler] failed to initialize:", e)

        if self.verbose:
            self.log("initialized; CuPy available =", self.has_cupy, "device =", self.gpu_device)


    def Ricker_Wavelet(
        self,
        freq: float,
        dt: float,
        nt: int,
        delay_cycles: float = 2.0,
        use_gpu: bool | None = None
    ):
        if nt <= 0:
            return cp.asarray([], dtype=cp.float32) if (use_gpu or (use_gpu is None and self.has_cupy)) else np.asarray(
                [], dtype=np.float32)

        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if use_gpu and HAS_CUPY else np

        t = xp.arange(nt, dtype=xp.float32) * float(dt)
        t0 = float(delay_cycles) / float(freq)
        pi2 = (xp.pi * float(freq)) ** 2
        dt_t = t - t0
        w = (1.0 - 2.0 * pi2 * (dt_t ** 2)) * xp.exp(-pi2 * (dt_t ** 2))
        if use_gpu and HAS_CUPY:
            mask = t < 0.0
            if mask.any():
                w[mask] = xp.float32(0.0)
        else:
            w[t < 0.0] = 0.0
        return w.astype(xp.float32, copy=False)


    def Random_Boundary(
        self,
        v,
        bc_max: int,
        nx: int | None = None,
        nz: int | None = None,
        use_gpu: bool | None = None,
        inplace: bool = True
    ):
        # Backend
        is_cp_array = HAS_CUPY and isinstance(v, cp.ndarray)
        use_gpu = (self.has_cupy and is_cp_array) if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        arr = v if (xp is np and isinstance(v, np.ndarray)) or (xp is cp and isinstance(v, cp.ndarray)) else xp.asarray(
            v)

        if arr.ndim != 2:
            raise ValueError("Random_Boundary: input v must be 2D (nx, nz)")

        # Sizes
        sx, sz = arr.shape
        if nx is not None and nx != sx:
            raise ValueError(f"Random_Boundary: given nx {nx} != array.shape[0] {sx}")
        if nz is not None and nz != sz:
            raise ValueError(f"Random_Boundary: given nz {nz} != array.shape[1] {sz}")

        if int(bc_max) <= 0:
            return arr if inplace else arr.copy()

        # Clamp bc_max to reasonable range
        bc = max(0, min(int(bc_max), min(sx, sz) - 1))

        # Indexes
        ix = xp.arange(sx, dtype=xp.int32)
        iz = xp.arange(sz, dtype=xp.int32)
        ixg, izg = xp.meshgrid(ix, iz, indexing="ij")  # shape (sx, sz)

        l_max = bc
        r_max = sx - bc - 1
        b_max = sz - bc - 1

        # Ensure valid corner indices
        l_max = max(0, min(l_max, sx - 1))
        r_max = max(0, min(r_max, sx - 1))
        b_max = max(0, min(b_max, sz - 1))

        mask_left = ixg < l_max
        mask_right = ixg > r_max
        mask_bottom = izg > b_max

        # Distances to boundaries (non-negative)
        d_left = xp.maximum(l_max - ixg, 0)
        d_right = xp.maximum(ixg - r_max, 0)
        d_bottom = xp.maximum(izg - b_max, 0)

        # Distance map initialized
        d = xp.zeros_like(arr, dtype=xp.float32)

        # Corner masks
        mask_lb = mask_bottom & mask_left
        mask_rb = mask_bottom & mask_right

        # Assign distance values
        if mask_lb.any():
            d[mask_lb] = xp.maximum(d_left[mask_lb], d_bottom[mask_lb])
        if mask_rb.any():
            d[mask_rb] = xp.maximum(d_right[mask_rb], d_bottom[mask_rb])

        # Bottom only
        mask_b = mask_bottom & ~(mask_left | mask_right)
        if mask_b.any():
            d[mask_b] = d_bottom[mask_b]

        # Left only
        mask_l = mask_left & ~mask_bottom
        if mask_l.any():
            d[mask_l] = d_left[mask_l]

        # Right only
        mask_r = mask_right & ~mask_bottom
        if mask_r.any():
            d[mask_r] = d_right[mask_r]

        v_ref = xp.zeros_like(arr, dtype=arr.dtype)

        v_ref_val_lb = arr[l_max, b_max]
        v_ref_val_rb = arr[r_max, b_max]

        if mask_b.any():
            v_ref[mask_b] = arr[ixg[mask_b], b_max]

        if mask_l.any():
            v_ref[mask_l] = arr[l_max, izg[mask_l]]

        if mask_r.any():
            v_ref[mask_r] = arr[r_max, izg[mask_r]]

        if mask_lb.any():
            v_ref[mask_lb] = v_ref_val_lb
        if mask_rb.any():
            v_ref[mask_rb] = v_ref_val_rb

        seed = v_ref / float(bc if bc != 0 else 1.0)
        v_new = v_ref - d * seed

        boundary_mask = mask_left | mask_right | mask_bottom

        if not inplace:
            out = arr.copy()
        else:
            out = arr

        out[boundary_mask] = v_new[boundary_mask]

        return out


    def Preconditioning(
        self,
        d_pp,
        nx: int,
        minnz: int,
        maxnz: int,
        d_cmpelev=None,
        ix2: int = 0,
        nz: int = None,
        HALF: int = HALF_PRECON,
        EPS: float = EPS_PRECON,
        SCALE: float = SCALE_PRECON,
        WMIN: float = WMIN_PRECON,
        WMAX: float = WMAX_PRECON,
        use_gpu: bool | None = None
    ):
        # Backend
        is_cp = HAS_CUPY and isinstance(d_pp, cp.ndarray)
        use_gpu = (self.has_cupy and is_cp) if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        arr = d_pp if (xp is np and isinstance(d_pp, np.ndarray)) or (
                    xp is cp and isinstance(d_pp, cp.ndarray)) else xp.asarray(d_pp)
        arr = arr.astype(xp.float32, copy=False)

        if nz is None:
            nz = arr.shape[1]
        if arr.ndim != 2 or arr.shape != (nx, nz):
            raise ValueError(f"Preconditioning: expected d_pp shape ({nx},{nz}), got {arr.shape}")

        mnz = int(max(0, minnz))
        mxz = int(min(maxnz, nz))
        if mnz >= mxz:
            return arr

        col_idx = xp.arange(nx, dtype=xp.int32) + xp.int32(ix2)

        if d_cmpelev is None or (hasattr(d_cmpelev, "size") and d_cmpelev.size == 0):
            cmp_vec = xp.full((nx,), mnz, dtype=xp.int32)
        else:
            cmp_src = xp.asarray(d_cmpelev, dtype=xp.int32)
            valid = (col_idx >= 0) & (col_idx < cmp_src.size)
            tmp = xp.full((nx,), mnz, dtype=xp.int32)
            if valid.any():
                tmp[valid] = cmp_src[col_idx[valid]]
            cmp_vec = xp.clip(tmp, mnz, nz).astype(xp.int32)

        sq = arr * arr
        cumsq = xp.concatenate((xp.zeros((nx, 1), dtype=sq.dtype), xp.cumsum(sq, axis=1)), axis=1)

        iz = xp.arange(mnz, mxz, dtype=xp.int32)
        L = iz.size
        if L == 0:
            return arr

        cmp_2d = cmp_vec[:, None]
        iz_2d = iz[None, :]

        z0 = xp.maximum(cmp_2d, iz_2d - int(HALF))
        z0 = xp.maximum(z0, mnz)
        z1 = xp.minimum(iz_2d + int(HALF), mxz - 1)

        z0_idx = z0.astype(xp.int64)
        z1p1_idx = (z1 + 1).astype(xp.int64)

        s2_hi = xp.take_along_axis(cumsq, z1p1_idx, axis=1)
        s2_lo = xp.take_along_axis(cumsq, z0_idx, axis=1)
        s2 = s2_hi - s2_lo

        count = (z1 - z0 + 1).astype(xp.float32)
        mask_pos = count > 0.0
        mean_sq = xp.zeros_like(s2, dtype=xp.float32)
        if mask_pos.any():
            mean_sq[mask_pos] = s2[mask_pos] / count[mask_pos]

        inv_rms = 1.0 / xp.sqrt(mean_sq + (EPS * EPS))
        w = SCALE * inv_rms
        w = xp.clip(w, WMIN, WMAX)

        mask_apply = (iz_2d >= cmp_2d)
        w_masked = xp.where(mask_apply, w, xp.ones_like(w, dtype=w.dtype)).astype(arr.dtype)
        arr[:, mnz:mxz] = arr[:, mnz:mxz] * w_masked
        return arr


    def Absorbgpuup(
        self,
        d_v,
        d_pn,
        d_pp,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        dt: float,
        iz: int = 1,
        eps: float = EPS,
        nbc_override: int | None = None,
        damping_map: object | None = None,
        apply_to: tuple = ("pp",),
        inplace: bool = True
    ):
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for Absorbgpuup (GPU implementation).")
        xp = cp

        if d_pp is None and d_pn is None:
            return d_pp
        if nx <= 0 or nz <= 0:
            return d_pp

        if nbc_override is not None:
            nbc = int(max(1, nbc_override))
        else:
            if iz is None or iz <= 1:
                nbc = max(4, min(20, min(nx, nz) // 12))
            else:
                nbc = int(max(1, min(max(1, min(nx, nz) // 2 - 1), iz)))

        v = xp.asarray(d_v, dtype=xp.float32)
        pp = xp.asarray(d_pp, dtype=xp.float32) if d_pp is not None else None
        pn = xp.asarray(d_pn, dtype=xp.float32) if d_pn is not None else None

        if damping_map is not None:
            damp = xp.asarray(damping_map, dtype=xp.float32)
            if damp.shape != (nx, nz):
                raise ValueError("damping_map must have shape (nx, nz)")
        else:
            ix = xp.arange(nx, dtype=xp.float32)
            izs = xp.arange(nz, dtype=xp.float32)

            left = xp.clip((nbc - ix) / float(nbc), 0.0, 1.0)
            right = xp.clip((ix - (nx - 1 - nbc)) / float(nbc), 0.0, 1.0)
            tx = xp.maximum(left, right)

            top = xp.clip((nbc - izs) / float(nbc), 0.0, 1.0)
            bottom = xp.clip((izs - (nz - 1 - nbc)) / float(nbc), 0.0, 1.0)
            tz = xp.maximum(top, bottom)

            tx2 = tx[:, None]
            tz2 = tz[None, :]
            edge_strength = xp.maximum(tx2, tz2)  # shape (nx,nz), 0..1

            edge_thresh = 1e-3
            mask_edge = edge_strength > edge_thresh

            v_local = xp.maximum(v, 1e-6, dtype=xp.float32)
            vmax = float(xp.max(v_local).astype(xp.float32))
            if vmax <= 0.0:
                vmax = 1.0
            v_factor = 0.7 + 0.3 * (v_local / (vmax + eps))
            v_factor = xp.clip(v_factor, 0.7, 1.0)

            tscale = min(1.0, max(0.05, float(dt) * 20.0))
            base_alpha = 0.08
            alpha = base_alpha * tscale
            alpha = float(min(alpha, 0.09))

            power = 2.0

            exponent = -alpha * (edge_strength ** power) * v_factor

            damp = xp.exp(exponent).astype(xp.float32)

            damp_min = 0.20
            damp = xp.clip(damp, damp_min, 1.0)

            if mask_edge is not None:
                ones = xp.ones_like(damp, dtype=xp.float32)
                damp = xp.where(mask_edge, damp, ones)

        try:
            if xp is cp:
                if "pp" in apply_to and pp is not None:
                    if inplace:
                        pp *= damp
                    else:
                        pp = pp * damp
                if "pn" in apply_to and pn is not None:
                    if inplace:
                        pn *= damp
                    else:
                        pn = pn * damp
            else:
                if "pp" in apply_to and pp is not None:
                    if inplace:
                        pp *= damp
                    else:
                        pp = pp * damp
                if "pn" in apply_to and pn is not None:
                    if inplace:
                        pn *= damp
                    else:
                        pn = pn * damp
        except Exception:
            if "pp" in apply_to and pp is not None:
                if inplace:
                    pp[...] = pp * damp
                else:
                    pp = pp * damp
            if "pn" in apply_to and pn is not None:
                if inplace:
                    pn[...] = pn * damp
                else:
                    pn = pn * damp

        try:
            if d_pp is not None and isinstance(d_pp, cp.ndarray):
                if pp is not d_pp:
                    d_pp[:] = pp
            if d_pn is not None and isinstance(d_pn, cp.ndarray):
                if pn is not d_pn:
                    d_pn[:] = pn
        except Exception:
            pass

        if getattr(self, "verbose", False):
            try:
                dmin = float(xp.min(damp))
                dmax = float(xp.max(damp))
                self.log(f"[Absorbgpuup] nbc={nbc}, alpha={alpha:.4f}, damp_min={dmin:.3f}, damp_max={dmax:.3f}")
            except Exception:
                try:
                    print("[Absorbgpuup] debug: could not read damp stats")
                except Exception:
                    pass

        return d_pp


    def Absorbing_Boundary_Condition(
        self,
        nx: int,
        dx: float,
        nz: int,
        dz: float,
        dt: float,
        dvv,
        od,
        p,
        pm,
        pp,
        abs_flags,
    ):
        # Backend
        xp = cp if getattr(self, "has_cupy", False) else np

        p = xp.asarray(p, dtype=xp.float32)
        pm = xp.asarray(pm, dtype=xp.float32)
        pp = xp.asarray(pp, dtype=xp.float32)
        dvv = xp.asarray(dvv, dtype=xp.float32)
        od_arr = None if od is None else xp.asarray(od, dtype=xp.float32)

        # Flags (top, left, bottom, right)
        flags = list(abs_flags)
        if len(flags) < 4:
            flags = (flags + [0, 0, 0, 0])[:4]
        top_flag, left_flag, bottom_flag, right_flag = [bool(int(f)) for f in flags[:4]]

        if nx < 2 or nz < 2:
            return pp

        # Precompute constants
        inv_2dt = 1.0 / (2.0 * float(dt))
        inv_dx = 1.0 / float(dx)
        inv_dz = 1.0 / float(dz)
        eps = 1e-12

        # Safe divide -> avoids large temporaries with masks
        def safe_ratio(num, denom, out=None):
            if out is None:
                return xp.where(denom > eps, num / denom, 0.0)
            else:
                xp.copyto(out, xp.where(denom > eps, num / denom, 0.0))
                return out

        def compute_dp_dx_at_z(iz):
            if nx == 1:
                return xp.zeros((1,), dtype=xp.float32)
            out = xp.empty((nx,), dtype=xp.float32)
            if nx == 2:
                val = (p[1, iz] - p[0, iz]) * inv_dx
                out[0] = val
                out[1] = val
                return out
            out[1:-1] = (p[2:, iz] - p[:-2, iz]) * (0.5 * inv_dx)
            out[0] = (p[1, iz] - p[0, iz]) * inv_dx
            out[-1] = (p[-1, iz] - p[-2, iz]) * inv_dx
            return out

        def compute_dp_dz_at_x(ix):
            if nz == 1:
                return xp.zeros((1,), dtype=xp.float32)
            out = xp.empty((nz,), dtype=xp.float32)
            if nz == 2:
                val = (p[ix, 1] - p[ix, 0]) * inv_dz
                out[0] = val
                out[1] = val
                return out
            out[...] = 0.0
            out[1:-1] = (p[ix, 2:] - p[ix, :-2]) * (0.5 * inv_dz)
            out[0] = (p[ix, 1] - p[ix, 0]) * inv_dz
            out[-1] = (p[ix, -1] - p[ix, -2]) * inv_dz
            return out

        #---------------------------
        # TOP BOUNDARY (PP[:,0])
        #---------------------------
        if top_flag:
            iz = 1 if nz > 1 else 0
            dv = dvv[:, iz]
            if od_arr is not None:
                with_od = od_arr[:, iz]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdx = compute_dp_dx_at_z(iz)
            dpdt = (pp[:, iz] - pm[:, iz]) * inv_2dt

            dpdxs = dpdx * dpdx
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdxs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dz) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[:, 0] = gamma * (pp[:, iz] - p[:, 0]) + p[:, iz]

        else:
            # pp[:,0] = 0.0
            pass

        #------------------------------
        # LEFT BOUNDARY (pp[0,:])
        #------------------------------
        if left_flag:
            ix = 1 if nx > 1 else 0
            dv = dvv[ix, :]
            if od_arr is not None:
                with_od = od_arr[ix, :]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdz = compute_dp_dz_at_x(ix)
            dpdt = (pp[ix, :] - pm[ix, :]) * inv_2dt

            dpdzs = dpdz * dpdz
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdzs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dx) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[0, :] = gamma * (pp[ix, :] - p[0, :]) + p[ix, :]

        else:
            # pp[0,:] = 0.0
            pass

        #----------------------------------
        # BOTTOM BOUNDARY (pp[:, nz-1])
        #----------------------------------
        if bottom_flag:
            iz = nz - 2 if nz > 1 else 0
            dv = dvv[:, iz]
            if od_arr is not None:
                with_od = od_arr[:, iz]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdx = compute_dp_dx_at_z(iz)
            dpdt = (pp[:, iz] - pm[:, iz]) * inv_2dt

            dpdxs = dpdx * dpdx
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdxs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dz) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[:, nz - 1] = gamma * (pp[:, iz] - p[:, nz - 1]) + p[:, iz]

        else:
            # pp[:, nz-1] = 0.0
            pass

        #------------------------------
        # RIGHT BOUNDARY
        #------------------------------
        if right_flag:
            ix = nx - 2 if nx > 1 else 0
            dv = dvv[ix, :]
            if od_arr is not None:
                with_od = od_arr[ix, :]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdz = compute_dp_dz_at_x(ix)
            dpdt = (pp[ix, :] - pm[ix, :]) * inv_2dt

            dpdzs = dpdz * dpdz
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdzs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dx) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[nx - 1, :] = gamma * (pp[ix, :] - p[nx - 1, :]) + p[ix, :]

        else:
            # pp[nx-1,:] = 0.0
            pass

        return pp


    def Image_Gaussian(
        self,
        d_pn,
        d_pnrec,
        sigma: float = SIGMA,
        truncate: float = TRUNCATE,
        mode: str = "reflect",
        use_gpu: bool | None = None,
        use_fast: bool = True,
    ):
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        if d_pn.shape != d_pnrec.shape:
            raise ValueError("d_pn and d_pnrec must have the same shape")

        if sigma is None or sigma <= 0.0:
            a = xp.asarray(d_pn, dtype=xp.float32)
            b = xp.asarray(d_pnrec, dtype=xp.float32)
            return a + b

        if not hasattr(self, "_gauss_kernel_cache"):
            self._gauss_kernel_cache = {}

        backend_tag = "cupy" if (xp is cp) else "numpy"
        cache_key = (backend_tag, float(sigma), float(truncate))

        k = self._gauss_kernel_cache.get(cache_key)
        if k is None:
            radius = max(1, int(truncate * sigma + 0.5))
            x = np.arange(-radius, radius + 1, dtype=np.float64)
            kk = np.exp(-0.5 * (x / float(sigma)) ** 2)
            kk = kk.astype(np.float32)
            kk /= kk.sum(dtype=np.float32)
            if xp is cp:
                try:
                    k = cp.asarray(kk, dtype=cp.float32)
                except Exception:
                    k = kk
            else:
                k = kk
            self._gauss_kernel_cache[cache_key] = k

        a = xp.asarray(d_pn, dtype=xp.float32)
        b = xp.asarray(d_pnrec, dtype=xp.float32)

        try:
            if xp is cp and use_fast and (cp_convolve1d is not None):
                res_a = cp_convolve1d(a, k, axis=0, mode=mode)
                res_a = cp_convolve1d(res_a, k, axis=1, mode=mode)
                res_b = cp_convolve1d(b, k, axis=0, mode=mode)
                res_b = cp_convolve1d(res_b, k, axis=1, mode=mode)
                return res_a + res_b

            if xp is np and use_fast and (sp_gaussian_filter is not None):
                res_a = sp_gaussian_filter(a, sigma=sigma, truncate=truncate, mode=mode)
                res_b = sp_gaussian_filter(b, sigma=sigma, truncate=truncate, mode=mode)
                return res_a.astype(np.float32) + res_b.astype(np.float32)

        except Exception:
            pass

        try:
            res_a = self.Apply_1D_Shifted_Convolution(xp, a, k, axis=0, pad_mode=mode)
            res_a = self.Apply_1D_Shifted_Convolution(xp, res_a, k, axis=1, pad_mode=mode)
            res_b = self.Apply_1D_Shifted_Convolution(xp, b, k, axis=0, pad_mode=mode)
            res_b = self.Apply_1D_Shifted_Convolution(xp, res_b, k, axis=1, pad_mode=mode)
            return (res_a + res_b).astype(xp.float32)
        except Exception as e:
            pad = int((k.size - 1) // 2)
            if xp is cp:
                pa = cp.pad(a, ((pad, pad), (pad, pad)), mode=mode)
                pb = cp.pad(b, ((pad, pad), (pad, pad)), mode=mode)
            else:
                pa = np.pad(a, ((pad, pad), (pad, pad)), mode=mode)
                pb = np.pad(b, ((pad, pad), (pad, pad)), mode=mode)

            tmp_a = xp.zeros_like(a)
            tmp_b = xp.zeros_like(b)
            for i in range(a.shape[0]):
                window = pa[i:i + k.size, pad:pad + a.shape[1]]
                tmp_a[i, :] = xp.tensordot(k, window, axes=(0, 0))
                window_b = pb[i:i + k.size, pad:pad + b.shape[1]]
                tmp_b[i, :] = xp.tensordot(k, window_b, axes=(0, 0))
            out_a = xp.zeros_like(a)
            out_b = xp.zeros_like(b)
            for j in range(a.shape[1]):
                window = tmp_a[pad:pad + a.shape[0], j:j + k.size]
                out_a[:, j] = xp.tensordot(k, window, axes=(0, 1))
                window_b = tmp_b[pad:pad + b.shape[0], j:j + k.size]
                out_b[:, j] = xp.tensordot(k, window_b, axes=(0, 1))
            return (out_a + out_b).astype(xp.float32)


    def Image_Denoise_Normalize_Cross_Correlation(
        self,
        image,
        sigma=SIGMA,
        local_sigma=LOCAL_SIGMA,
        gamma=GAMMA,
        eps=EPS,
        use_gpu=True
    ):
        xp = cp if (use_gpu and self.has_cupy) else np
        image = xp.asarray(image, dtype=xp.float32)

        smooth = self.Image_Gaussian(image, xp.zeros_like(image), sigma=sigma, use_gpu=use_gpu)

        prod = image * smooth
        a2 = image * image
        b2 = smooth * smooth

        num = self.Image_Gaussian(prod, xp.zeros_like(prod), sigma=local_sigma, use_gpu=use_gpu)
        den1 = self.Image_Gaussian(a2, xp.zeros_like(a2), sigma=local_sigma, use_gpu=use_gpu)
        den2 = self.Image_Gaussian(b2, xp.zeros_like(b2), sigma=local_sigma, use_gpu=use_gpu)

        denom = xp.sqrt(den1 * den2 + eps)
        ncc = xp.clip(num / denom, 0.0, 1.0)

        if gamma != 1.0:
            ncc = ncc ** gamma

        return ncc * image + (1.0 - ncc) * smooth


    #=============================
    # Acoustic Wave Propagation
    #=============================
    @staticmethod
    def FD_Weights_Fornberg(x, x0, m):
        n = len(x)
        c = np.zeros((n, m + 1), dtype=np.float64)
        c1 = 1.0
        c4 = x[0] - x0
        c[0, 0] = 1.0
        for i in range(1, n):
            mn = min(i, m)
            c2 = 1.0
            c5 = c4
            c4 = x[i] - x0
            for j in range(0, i):
                c3 = x[i] - x[j]
                c2 = c2 * c3
                if c3 == 0.0:
                    raise ValueError("Two nodes are identical in FD Stencil")
                for k in range(mn, 0, -1):
                    c[i, k] = (c4 * c[i - 1, k] - k * c[i - 1, k - 1]) / c3
                c[i, 0] = c4 * c[i - 1, 0] / c3
                for k in range(mn, 0, -1):
                    c[j, k] = (c5 * c[j, k] - k * c[j, k - 1]) / c3
                c[j, 0] = c5 * c[j, 0] / c3
            c1 = c2
        return c[:, m].astype(np.float32)

    @staticmethod
    def Make_FD_First_Derivative_Weights(self, R, dx):
        nodes = np.arange(-R, R + 1, dtype=np.float64) * dx
        # w = cls.FD_Weights_Fornberg(nodes, 0.0, 1)
        w = Acoustic_Reverse_Time_Migration_2D.FD_Weights_Fornberg(nodes, 0.0, 1)
        return w.astype(np.float32)

    @staticmethod
    def Get_FD_Weights(self, R, dx):
        key = (int(R), float(dx))
        w = FD_WEIGHTS_CACHE.get(key)
        if w is None:
            # w = cls.Make_FD_First_Derivative_Weights(R, dx)
            w = Acoustic_Reverse_Time_Migration_2D.Make_FD_First_Derivative_Weights(R, dx)
            FD_WEIGHTS_CACHE[key] = w
        return w

    def Ensure_FD_Weights(self, R, dx):
        R = int(max(1, R))
        if R == 1:
            w = np.array([-0.5, 0.0, 0.5], dtype=np.float32) / float(dx)
        elif R == 2:
            w = np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=np.float32) / float(dx)
        elif R == 3:
            w = np.array([-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60], dtype=np.float32) / float(dx)
        else:
            n = 2 * R + 1
            w = np.zeros(n, dtype=np.float32)
            mid = n // 2
            w[mid - 1] = -0.5 / float(dx)
            w[mid + 1] = 0.5 / float(dx)
        key = (int(R), float(dx))
        self._fd_cache[key] = w
        if self.verbose:
            self.log("FD weights sum check:", float(w.sum()))
        return w

    def Apply_1D_Shifted_Convolution(
        self,
        xp,
        arr,
        kernel,
        axis=0,
        pad_mode="nearest",
        out=None
    ):
        #-------------
        # CuPy
        #-------------
        if xp is cp:
            if cp_convolve1d is None:
                raise RuntimeError(
                    "cupyx.scipy.ndimage.convolve1d is not available. "
                    "Install full CuPy with SciPy support."
                )

            k = kernel if isinstance(kernel, cp.ndarray) else cp.asarray(kernel, dtype=cp.float32)

            if out is None:
                out = cp.empty_like(arr)

            out[...] = cp_convolve1d(arr, k, axis=axis, mode=pad_mode)
            return out

        #-----------
        # NumPy
        #-----------
        else:
            if sp_convolve1d is None:
                raise RuntimeError(
                    "scipy.ndimage.convolve1d is not available. "
                    "Install SciPy."
                )

            k = kernel if isinstance(kernel, np.ndarray) else np.asarray(kernel, dtype=np.float32)

            if out is None:
                out = np.empty_like(arr)

            out[...] = sp_convolve1d(arr, k, axis=axis, mode=pad_mode)
            return out

    def Acoustic_TimeDomain_Propagation(
        self,
        p,
        vx,
        vz,
        rho,
        K=None,
        vp=None,
        nx: int = None,
        dx: float = None,
        nz: int = None,
        dz: float = None,
        dt: float = None,
        fd_order_radius: int = 3,
        use_gpu: bool | None = True,
        pad_mode: str = "reflect",
        absorb_R: int = 0,
        src: dict | None = None,
        inplace: bool = True,
        use_external_abc: bool = True,
        cfl_warn: float = 0.5
    ):
        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        if nx is None or nz is None or dx is None or dz is None or dt is None:
            raise ValueError("nx,nz,dx,dz,dt must be provided")

        if not (hasattr(p, "shape") and len(p.shape) == 2):
            raise ValueError("p must be 2D (nx,nz)")
        if p.shape != (nx, nz):
            raise ValueError(f"p shape {p.shape} != ({nx},{nz})")

        if inplace:
            p_arr = xp.asarray(p, dtype=xp.float32)
            vx_arr = xp.asarray(vx, dtype=xp.float32)
            vz_arr = xp.asarray(vz, dtype=xp.float32)
        else:
            p_arr = xp.asarray(p, dtype=xp.float32).copy()
            vx_arr = xp.asarray(vx, dtype=xp.float32).copy()
            vz_arr = xp.asarray(vz, dtype=xp.float32).copy()

        # density and modulus
        if vp is not None and K is not None:
            # if both given, prefer explicit K but keep vp for CFL check
            K_arr = xp.asarray(K, dtype=xp.float32) if not np.isscalar(K) else xp.full((nx, nz), float(K),
                                                                                       dtype=xp.float32)
            vp_arr = xp.asarray(vp, dtype=xp.float32) if not np.isscalar(vp) else xp.full((nx, nz), float(vp),
                                                                                          dtype=xp.float32)
        elif vp is not None:
            vp_arr = xp.asarray(vp, dtype=xp.float32) if not np.isscalar(vp) else xp.full((nx, nz), float(vp),
                                                                                          dtype=xp.float32)
            rho_arr = xp.asarray(rho, dtype=xp.float32) if not np.isscalar(rho) else xp.full((nx, nz), float(rho),
                                                                                             dtype=xp.float32)
            K_arr = rho_arr * (vp_arr ** 2)
        elif K is not None:
            K_arr = xp.asarray(K, dtype=xp.float32) if not np.isscalar(K) else xp.full((nx, nz), float(K),
                                                                                       dtype=xp.float32)
            # compute vp for CFL check from K/rho if possible
            rho_arr = xp.asarray(rho, dtype=xp.float32) if not np.isscalar(rho) else xp.full((nx, nz), float(rho),
                                                                                             dtype=xp.float32)
            vp_arr = xp.sqrt(xp.maximum(K_arr / xp.maximum(rho_arr, 1e-12), 1e-12))
        else:
            raise ValueError("Either vp or K must be provided")

        rho_arr = xp.asarray(rho, dtype=xp.float32) if not np.isscalar(rho) else xp.full((nx, nz), float(rho),
                                                                                         dtype=xp.float32)
        if rho_arr.shape != (nx, nz):
            try:
                rho_arr = xp.broadcast_to(rho_arr, (nx, nz)).astype(xp.float32)
            except Exception:
                raise ValueError("rho must be scalar or shape (nx,nz) or (nx,) or (nz,)")

        vp_max = float(xp.max(vp_arr))
        if vp_max > 0.0:
            dt_stable = cfl_warn * min(dx, dz) / (vp_max * (2.0 ** 0.5))  # heuristic for 2D high-order
            if dt > dt_stable and self.verbose:
                self.log(
                    f"[Acoustic] WARNING: dt ({dt}) may exceed recommended stable dt ({dt_stable:.3e}). Reduce dt or increase grid spacing.")

        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        if self.verbose:
            self.log("Acoustic FD weights sums:", float(w_x.sum()), float(w_z.sum()))

        dp_dx = self.Apply_1D_Shifted_Convolution(xp, p_arr, w_x, axis=0, pad_mode=pad_mode)
        dp_dz = self.Apply_1D_Shifted_Convolution(xp, p_arr, w_z, axis=1, pad_mode=pad_mode)

        inv_rho = 1.0 / xp.maximum(rho_arr, xp.array(1e-12, dtype=xp.float32))
        factor_v = (-dt * inv_rho).astype(xp.float32)

        vx_arr += factor_v * dp_dx
        vz_arr += factor_v * dp_dz

        if xp is cp:
            if bool(cp.any(~cp.isfinite(vx_arr))) or bool(cp.any(~cp.isfinite(vz_arr))):
                vx_arr = cp.nan_to_num(vx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                vz_arr = cp.nan_to_num(vz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Acoustic] NaN/Inf detected in velocity - clamped")
        else:
            if (not np.isfinite(vx_arr).all()) or (not np.isfinite(vz_arr).all()):
                vx_arr = np.nan_to_num(vx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                vz_arr = np.nan_to_num(vz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Acoustic] NaN/Inf detected in velocity - clamped")

        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            amp = float(src.get("amp", 0.0))
            stype = src.get("type", "pressure")
            if 0 <= sx < nx and 0 <= sz < nz and amp != 0.0:
                if stype == "pressure":
                    p_arr[sx, sz] += amp
                elif stype == "velocity_x":
                    vx_arr[sx, sz] += amp
                elif stype == "velocity_z":
                    vz_arr[sx, sz] += amp

        dvx_dx = self.Apply_1D_Shifted_Convolution(xp, vx_arr, w_x, axis=0, pad_mode=pad_mode)
        dvz_dz = self.Apply_1D_Shifted_Convolution(xp, vz_arr, w_z, axis=1, pad_mode=pad_mode)
        div_v = dvx_dx + dvz_dz

        factor_p = (-dt * K_arr).astype(xp.float32)
        p_arr += factor_p * div_v

        if xp is cp:
            if bool(cp.any(~cp.isfinite(p_arr))):
                p_arr = cp.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Acoustic] NaN/Inf detected in pressure - clamped")
        else:
            if not np.isfinite(p_arr).all():
                p_arr = np.nan_to_num(p_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Acoustic] NaN/Inf detected in pressure - clamped")

        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            amp = float(src.get("amp", 0.0))
            stype = src.get("type", "pressure")
            if 0 <= sx < nx and 0 <= sz < nz and amp != 0.0 and stype == "pressure":
                p_arr[sx, sz] += amp

        if absorb_R is not None and int(absorb_R) > 0:
            R_abs = int(absorb_R)
            R_abs = min(R_abs, max(1, min(nx, nz) // 2 - 1))
            ix = xp.arange(nx, dtype=xp.float32)
            izs = xp.arange(nz, dtype=xp.float32)
            left = xp.clip((R_abs - ix) / float(R_abs), 0.0, 1.0)
            right = xp.clip((ix - (nx - 1 - R_abs)) / float(R_abs), 0.0, 1.0)
            tx = xp.maximum(left, right)
            top = xp.clip((R_abs - izs) / float(R_abs), 0.0, 1.0)
            bottom = xp.clip((izs - (nz - 1 - R_abs)) / float(R_abs), 0.0, 1.0)
            tz = xp.maximum(top, bottom)
            tx2 = 1.0 - (tx ** 2)
            tz2 = 1.0 - (tz ** 2)
            tap2d = (tx2[:, None] * tz2[None, :]).astype(xp.float32)

            vx_arr *= tap2d
            vz_arr *= tap2d
            p_arr *= tap2d

            if use_external_abc and "absorbgpuup" in self.k:
                try:
                    vel_map = vp_arr
                    abc_fn = self.k["absorbgpuup"]
                    abc_fn(vel_map, vx_arr, vx_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, vz_arr, vz_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, p_arr, p_arr, nx, nz, dx, dz, dt, iz=R_abs)
                except Exception:
                    pass

        return p_arr, vx_arr, vz_arr


    def Helmholtz_Acoustic_Propagation(
        self,
        p,
        rho,
        K=None,
        vp=None,
        nx: int = None,
        dx: float = None,
        nz: int = None,
        dz: float = None,
        freq: float = 0.0,
        fd_order_radius: int = 3,
        use_gpu: bool | None = True,
        pad_mode: str = 'reflect',
        absorb_R: int = 0,
        src: dict | None = None,
        inplace: bool = True,
        use_external_abc: bool = False,
        solver_tol: float = 1e-6,
        solver_maxiter: int = 1000,
        restart: int = 50,
        verbose: bool | None = None
    ):
        if verbose is None:
            verbose = getattr(self, "verbose", False)

        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and cp is not None) else np

        # Checks
        if nx is None or nz is None or dx is None or dz is None:
            raise ValueError("nx, nz, dx, dz must be provided")
        if p is not None and (not (hasattr(p, "shape") and len(p.shape) == 2)):
            raise ValueError("p must be 2D (nx,nz) or None for allocation")
        if p is not None and p.shape != (nx, nz):
            raise ValueError(f"p shape {p.shape} != ({nx},{nz})")

        # Allocate arrays
        if inplace:
            p_arr = xp.asarray(p, dtype=xp.complex64) if p is not None else xp.zeros((nx, nz), dtype=xp.complex64)
        else:
            p_arr = xp.asarray(p, dtype=xp.complex64).copy() if p is not None else xp.zeros((nx, nz),
                                                                                            dtype=xp.complex64)
        # Helper to broadcast material to (nx,nz)
        def to_field(a):
            if np.isscalar(a):
                return xp.full((nx, nz), float(a), dtype=xp.float32)
            arr = xp.asarray(a, dtype=xp.float32)
            if arr.shape == (nx,):
                return xp.broadcast_to(arr.reshape(nx, 1), (nx, nz)).astype(xp.float32)
            if arr.shape == (nz,):
                return xp.broadcast_to(arr.reshape(1, nz), (nx, nz)).astype(xp.float32)
            if arr.shape == (nx, nz):
                return arr.astype(xp.float32)
            raise ValueError("material must be scalar or shape (nx,nz) or (nx,) or (nz,)")

        # Build K and vp maps
        if K is not None:
            K_arr = to_field(K)
            rho_arr = to_field(rho)
            vp_arr = xp.sqrt(xp.maximum(K_arr / xp.maximum(rho_arr, 1e-12), 1e-12))
        elif vp is not None:
            rho_arr = to_field(rho)
            vp_arr = to_field(vp)
            K_arr = rho_arr * (vp_arr ** 2)
        else:
            raise ValueError("Either bulk modulus K or wave speed vp must be provided")

        # Frequency
        omega = 2.0 * np.pi * float(freq)

        # FD weights
        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        if verbose:
            try:
                self.log("Helmholtz Acoustic: FD weights sums:", float(w_x.sum()), float(w_z.sum()))
            except Exception:
                print("[Helmholtz Acoustic] FD weights sums:", float(w_x.sum()), float(w_z.sum()))

        def D_x(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_x, axis=0, pad_mode=pad_mode)

        def D_z(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_z, axis=1, pad_mode=pad_mode)

        # Sponge (taper)
        def make_sponge(R_abs, strength=50.0):
            if R_abs is None or int(R_abs) <= 0:
                return xp.zeros((nx, nz), dtype=xp.float32)
            R_abs = min(int(R_abs), max(1, min(nx, nz) // 2 - 1))
            ix = np.arange(nx, dtype=np.float32)
            izs = np.arange(nz, dtype=np.float32)
            left = np.clip((R_abs - ix) / float(R_abs), 0.0, 1.0)
            right = np.clip((ix - (nx - 1 - R_abs)) / float(R_abs), 0.0, 1.0)
            tx = np.maximum(left, right)
            top = np.clip((R_abs - izs) / float(R_abs), 0.0, 1.0)
            bottom = np.clip((izs - (nz - 1 - R_abs)) / float(R_abs), 0.0, 1.0)
            tz = np.maximum(top, bottom)
            tx2 = 1.0 - (tx ** 2)
            tz2 = 1.0 - (tz ** 2)
            tap2d = (tx2[:, None] * tz2[None, :]).astype(np.float32)
            eta_np = strength * (1.0 - tap2d)
            return xp.asarray(eta_np, dtype=xp.float32)

        damping_eta = make_sponge(absorb_R) if int(absorb_R) > 0 else xp.zeros((nx, nz), dtype=xp.float32)

        def apply_operator_p(p_field):
            p_local = xp.asarray(p_field, dtype=xp.complex64)
            dp_dx = D_x(p_local)
            dp_dz = D_z(p_local)
            inv_rho = 1.0 / xp.maximum(rho_arr, xp.array(1e-12, dtype=xp.float32))
            # flux components (real dtype promoted to complex when multiplied)
            fx = (inv_rho * dp_dx).astype(p_local.dtype, copy=False)
            fz = (inv_rho * dp_dz).astype(p_local.dtype, copy=False)
            div_flux = D_x(fx) + D_z(fz)
            # Helmholtz mass term
            mass = ((omega ** 2) * (1.0 / xp.maximum(K_arr, xp.array(1e-12, dtype=xp.float32)))).astype(p_local.dtype)
            Ap = div_flux + mass * p_local
            # sponge damping (complex)
            if int(absorb_R) > 0:
                Ap = Ap + (-1j * omega) * (damping_eta.astype(Ap.dtype)) * p_local
            # external ABC hook (optional)
            if use_external_abc and hasattr(self, "k") and "absorbgpuup" in getattr(self, "k", {}):
                try:
                    abc_fn = self.k["absorbgpuup"]
                    abc_fn(omega, Ap, nx, nz, dx, dz)
                except Exception:
                    pass
            return Ap

        # Build RHS b
        f_rhs = xp.zeros((nx, nz), dtype=xp.complex64)
        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            amp = complex(src.get("amp", 0.0))
            if 0 <= sx < nx and 0 <= sz < nz and amp != 0.0:
                f_rhs[sx, sz] = amp

        n = nx * nz

        # matvec for solver
        def matvec_xp(v):
            v2 = v.reshape((nx, nz))
            Av = apply_operator_p(v2)
            return Av.ravel()

        # prepare b in backend
        b = xp.asarray(f_rhs.ravel(), dtype=xp.complex64)

        if verbose:
            try:
                self.log(
                    f"[Helmholtz Acoustic] Solving freq={freq} Hz, n={n}, tol={solver_tol}, maxiter={solver_maxiter}, use_gpu={use_gpu}")
            except Exception:
                print(
                    f"[Helmholtz Acoustic] Solving freq={freq} Hz, n={n}, tol={solver_tol}, maxiter={solver_maxiter}, use_gpu={use_gpu}")

        # initial guess
        x0 = None
        if p is not None:
            try:
                maybe = xp.asarray(p_arr.ravel(), dtype=xp.complex64)
                if xp.any(maybe != 0.0):
                    x0 = maybe
            except Exception:
                x0 = None

        info = 0
        x_sol = None

        # Try GPU cupyx.gmres
        if use_gpu and cp is not None:
            try:
                from cupyx.scipy.sparse.linalg import gmres as cupy_gmres
                class CupyLinearOp:
                    def __init__(self, n, matvec):
                        self.shape = (n, n)
                        self.dtype = cp.complex64
                        self.matvec = matvec

                    def __call__(self, x):
                        return self.matvec(x)

                Lop = CupyLinearOp(n, matvec_xp)
                x_sol, info = cupy_gmres(Lop, b, x0=x0, tol=solver_tol, restart=restart, maxiter=solver_maxiter)
                x_sol = x_sol.astype(xp.complex64)
            except Exception:
                if verbose:
                    self.log(
                        "[Helmholtz Acoustic] cupyx.gmres unavailable or failed - using internal GPU GMRES fallback")

                # internal GMRES (matrix-free) for xp backend
                def _solve_upper_hessenberg(Hmat, gvec, xp_local):
                    k = Hmat.shape[0]
                    y = xp_local.zeros((k,), dtype=Hmat.dtype)
                    for i in range(k - 1, -1, -1):
                        s = gvec[i]
                        if i < k - 1:
                            s = s - (Hmat[i, i + 1:k] @ y[i + 1:k])
                        diag = Hmat[i, i]
                        if abs(diag) == 0:
                            y[i] = 0
                        else:
                            y[i] = s / diag
                    return y

                def gmres_gpu(matvec, b_vec, x0=None, restart=restart, tol=solver_tol, maxiter=solver_maxiter):
                    linalg = xp.linalg
                    vdot = xp.vdot if hasattr(xp, "vdot") else lambda a, b: (a.conjugate() * b).sum()
                    if x0 is None:
                        x = xp.zeros_like(b_vec)
                    else:
                        x = x0.astype(b_vec.dtype, copy=True)
                    b_norm = float(linalg.norm(b_vec))
                    if b_norm == 0.0:
                        return x, 0
                    iter_total = 0
                    m = int(max(1, restart))
                    while iter_total < maxiter:
                        Ax = matvec(x)
                        r = b_vec - Ax
                        beta = float(linalg.norm(r))
                        if beta / (b_norm + 1e-30) <= tol:
                            return x, 0
                        V = []
                        H = xp.zeros((m + 1, m), dtype=b_vec.dtype)
                        v0 = r / (beta + 1e-30)
                        V.append(v0)
                        g = xp.zeros((m + 1,), dtype=b_vec.dtype)
                        g[0] = beta
                        cs = xp.zeros((m,), dtype=b_vec.dtype)
                        sn = xp.zeros((m,), dtype=b_vec.dtype)
                        for j in range(m):
                            iter_total += 1
                            vj = V[j]
                            w = matvec(vj)
                            # Modified Gram-Schmidt
                            for i in range(j + 1):
                                vi = V[i]
                                hij = vdot(vi, w)
                                H[i, j] = hij
                                w = w - hij * vi
                            hjp1 = float(linalg.norm(w))
                            H[j + 1, j] = hjp1
                            if hjp1 != 0.0:
                                V.append(w / hjp1)
                            else:
                                y = _solve_upper_hessenberg(H[:j + 1, :j + 1], g[:j + 1], xp)
                                Vmat = xp.stack(V[:j + 1], axis=1)
                                x = x + (Vmat @ y)
                                return x, 0
                            # Apply Givens rotations
                            for i in range(j):
                                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                                H[i, j] = temp
                            hj = H[j, j]
                            hj1 = H[j + 1, j]
                            denom = xp.sqrt((hj.conjugate() * hj).real + (hj1.conjugate() * hj1).real)
                            if denom == 0:
                                c = 1.0
                                s = 0.0
                            else:
                                c = hj / denom
                                s = hj1 / denom
                            cs[j] = c
                            sn[j] = s
                            H[j, j] = c * H[j, j] + s * H[j + 1, j]
                            H[j + 1, j] = 0.0
                            tempg = c * g[j] + s * g[j + 1]
                            g[j + 1] = - (s.conjugate()) * g[j] + (c.conjugate()) * g[j + 1]
                            g[j] = tempg
                            res_norm = float(abs(g[j + 1]))
                            if res_norm / (b_norm + 1e-30) <= tol:
                                y = _solve_upper_hessenberg(H[:j + 1, :j + 1], g[:j + 1], xp)
                                Vmat = xp.stack(V[:j + 1], axis=1)
                                x = x + (Vmat @ y)
                                return x, 0
                            if iter_total >= maxiter:
                                break
                        y = _solve_upper_hessenberg(H[:m, :m], g[:m], xp)
                        Vmat = xp.stack(V[:m], axis=1)
                        x = x + (Vmat @ y)
                        Ax = matvec(x)
                        r = b_vec - Ax
                        beta = float(linalg.norm(r))
                        if beta / (b_norm + 1e-30) <= tol:
                            return x, 0
                    return x, iter_total

                b_xp = xp.asarray(b, dtype=xp.complex64)
                x0_xp = xp.asarray(x0, dtype=xp.complex64) if x0 is not None else None
                x_sol, info = gmres_gpu(matvec_xp, b_xp, x0=x0_xp, restart=restart, tol=solver_tol, maxiter=solver_maxiter)

        else:
            try:
                from scipy.sparse.linalg import LinearOperator as SciPyLOp, gmres as scipy_gmres
                def mv_for_scipy(v_numpy):
                    v_np = np.asarray(v_numpy, dtype=np.complex64)
                    v2 = v_np.reshape((nx, nz))
                    Av = apply_operator_p(v2)
                    return np.asarray(Av.ravel(), dtype=np.complex64)

                Aop = SciPyLOp((n, n), matvec=mv_for_scipy, dtype=np.complex64)
                b_cpu = np.asarray(b.ravel(), dtype=np.complex64)
                x0_cpu = None
                if x0 is not None:
                    x0_cpu = np.asarray(x0, dtype=np.complex64)
                x_sol, info = scipy_gmres(Aop, b_cpu, x0=x0_cpu, tol=solver_tol, restart=restart,
                                          maxiter=solver_maxiter)
                x_sol = np.asarray(x_sol, dtype=np.complex64)
                if xp is cp:
                    x_sol = xp.asarray(x_sol)
            except Exception as e:
                raise RuntimeError("CPU GMRES failed: " + str(e))

        if x_sol is None:
            raise RuntimeError("Solver produced no solution (x_sol is None)")
        try:
            if xp is np:
                p_out = np.asarray(x_sol.reshape((nx, nz))).astype(np.complex64)
            else:
                p_out = x_sol.reshape((nx, nz)).astype(xp.complex64)
        except Exception:
            p_out = xp.asarray(np.asarray(x_sol).reshape((nx, nz)), dtype=xp.complex64)

        return p_out, int(info)


    """
    def Laplace_Denoise(
        self,
        *fields,
        nx: int = None,
        nz: int = None,
        dx: float = None,
        dz: float = None,
        use_gpu: bool | None = None,
        alpha: float = 1.0,
        fd_order_radius: int = 3,
        tol: float = 1e-6,
        maxiter: int = MAXITER,
        restart: int = 50,
        pad_mode: str = "reflect",
        energy_normalize: bool = True,
        energy_clip: tuple = (0.5, 2.0),
        inplace: bool = False,
        verbose: bool | None = None
    ):
        if verbose is None:
            verbose = getattr(self, "verbose", False)

        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        # Checks
        if len(fields) == 0:
            raise ValueError("At least one field must be provided")

        # Infer grid sizes
        sample_field = fields[0]
        if nx is None or nz is None:
            if hasattr(sample_field, "shape") and len(sample_field.shape) == 2:
                nx_, nz_ = sample_field.shape
                nx = nx if nx is not None else nx_
                nz = nz if nz is not None else nz_
            else:
                raise ValueError("Cannot infer nx,nz from input; pass nx and nz explicitly")

        if dx is None or dz is None:
            dx = float(dx) if dx is not None else float(getattr(self, "DX", 1.0))
            dz = float(dz) if dz is not None else float(getattr(self, "DZ", 1.0))

        # FD weights
        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        def D_x(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_x, axis=0, pad_mode=pad_mode)

        def D_z(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_z, axis=1, pad_mode=pad_mode)

        # Laplacian using FD
        def laplacian(u):
            return D_x(D_x(u)) + D_z(D_z(u))

        # Build matrix-free operator
        def A_op(u):
            return u - alpha * laplacian(u)

        # Matrix-free Conjugate Gradient
        def cg_solve(b, x0=None, tol_local=tol, maxit=maxiter):
            n = b.size
            if x0 is None:
                x = xp.zeros_like(b)
            else:
                x = x0.astype(b.dtype, copy=True)

            r = b - A_op(x.reshape((nx, nz))).ravel()
            p = r.copy()
            rsold = float(xp.dot(r.conjugate(), r).real)

            if rsold == 0.0:
                return x, 0

            for k in range(maxit):
                Ap = A_op(p.reshape((nx, nz))).ravel()
                pAp = float(xp.dot(p.conjugate(), Ap).real)
                if pAp == 0.0:
                    if verbose:
                        self.log("[Laplace_Denoise] breakdown in CG (pAp==0)")
                    break
                alpha_k = rsold / (pAp + 1e-30)
                x = x + alpha_k * p
                r = r - alpha_k * Ap
                rsnew = float(xp.dot(r.conjugate(), r).real)
                if rsnew ** 0.5 <= tol_local:
                    if verbose:
                        self.log(f"[Laplace_Denoise] CG converged k={k + 1}, res={rsnew ** 0.5:.3e}")
                    return x, 0
                p = r + (rsnew / (rsold + 1e-30)) * p
                rsold = rsnew
            if verbose:
                self.log(f"[Laplace_Denoise] CG reached maxiter={maxit}, final resid={rsold ** 0.5:.3e}")
            return x, 1

        use_external_solver = False
        if xp is np:
            try:
                from scipy.sparse.linalg import LinearOperator as SciPyLOp, cg as scipy_cg
                def mv_scipy(v_numpy):
                    v_np = np.asarray(v_numpy, dtype=np.float32)
                    Av = A_op(xp.asarray(v_np).reshape((nx, nz))).ravel()
                    return np.asarray(Av, dtype=np.float32)

                Aop_scipy = SciPyLOp((nx * nz, nx * nz), matvec=mv_scipy, dtype=np.float32)
                use_external_solver = True
                solver_cpu = ("scipy", scipy_cg, Aop_scipy)
            except Exception:
                use_external_solver = False
        else:
            try:
                import cupyx.scipy.sparse.linalg as cpx_linalg
                if hasattr(cpx_linalg, "cg"):
                    use_external_solver = True
                    solver_gpu = (
                    "cupyx", cpx_linalg.cg, None)
            except Exception:
                use_external_solver = False

        def solve_field(f_in):
            f = xp.asarray(f_in, dtype=xp.float32)
            b = f.ravel()
            x0 = None
            if xp is np and use_external_solver and solver_cpu[0] == "scipy":
                scipy_cg = solver_cpu[1]
                Aop = solver_cpu[2]
                try:
                    x_sol, info = scipy_cg(Aop, b, x0=None, tol=tol, maxiter=maxiter)
                    return xp.asarray(x_sol.reshape((nx, nz)), dtype=xp.float32), int(info)
                except Exception:
                    if verbose:
                        self.log("[Laplace_Denoise] scipy.cg failed - falling back to internal CG")
            x_sol, info = cg_solve(b.astype(xp.float32), x0=None, tol_local=tol, maxit=maxiter)
            x_out = x_sol.reshape((nx, nz)).astype(xp.float32)
            return x_out, int(info)

        outputs = []
        infos = []
        for fld in fields:
            if fld is None:
                fld_arr = xp.zeros((nx, nz), dtype=xp.float32)
            else:
                fld_arr = xp.asarray(fld, dtype=xp.float32)
            denoised, info = solve_field(fld_arr)
            outputs.append(denoised)
            infos.append(info)

        if energy_normalize:
            emin, emax = float(energy_clip[0]), float(energy_clip[1])
            for i, (orig, den) in enumerate(zip(fields, outputs)):
                orig_arr = xp.asarray(orig, dtype=xp.float32)
                try:
                    E_orig = self.Image_Gaussian(orig_arr * orig_arr, xp.zeros_like(orig_arr), sigma=dx, truncate=3.0, mode=pad_mode, use_gpu=(xp is cp))
                    E_den = self.Image_Gaussian(den * den, xp.zeros_like(den), sigma=dx, truncate=3.0, mode=pad_mode, use_gpu=(xp is cp))
                except Exception:
                    kernel = xp.ones((3, 3), dtype=xp.float32) / 9.0
                    E_orig = self.Apply_1D_Shifted_Convolution(xp, self.Apply_1D_Shifted_Convolution(orig_arr * orig_arr, kernel[1, :], axis=0, pad_mode=pad_mode), kernel[:, 1], axis=1, pad_mode=pad_mode)
                    E_den = self.Apply_1D_Shifted_Convolution(xp, self.Apply_1D_Shifted_Convolution(den * den, kernel[1, :], axis=0, pad_mode=pad_mode), kernel[:, 1], axis=1, pad_mode=pad_mode)
                ratio = xp.sqrt((E_orig + 1e-12) / (E_den + 1e-12))
                ratio = xp.clip(ratio, emin, emax)
                outputs[i] = (den * ratio).astype(xp.float32, copy=False)

        if len(outputs) == 1:
            out0 = outputs[0]
            if inplace:
                try:
                    fields[0][:] = out0
                    return out0
                except Exception:
                    return out0
            return out0
        else:
            if inplace:
                out_tuple = []
                for orig, den in zip(fields, outputs):
                    try:
                        orig[:] = den
                        out_tuple.append(orig)
                    except Exception:
                        out_tuple.append(den)
                return tuple(out_tuple)
            return tuple(outputs)
    """


    def Laplace_Denoise(
        self,
        *fields,
        nx: int = None,
        nz: int = None,
        dx: float = None,
        dz: float = None,
        use_gpu: bool | None = None,
        alpha: float = 1.0,
        fd_order_radius: int = 3,
        tol: float = 1e-6,
        maxiter: int = MAXITER,
        restart: int = 50,
        pad_mode: str = "reflect",
        energy_normalize: bool = True,
        energy_clip: tuple = (0.5, 2.0),
        inplace: bool = False,
        verbose: bool | None = None
    ):
        if verbose is None:
            verbose = getattr(self, "verbose", False)

        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        # Checks
        if len(fields) == 0:
            raise ValueError("At least one field must be provided")

        # Infer grid sizes
        sample_field = fields[0]
        if nx is None or nz is None:
            if hasattr(sample_field, "shape") and len(sample_field.shape) == 2:
                nx_, nz_ = sample_field.shape
                nx = nx if nx is not None else nx_
                nz = nz if nz is not None else nz_
            else:
                raise ValueError("Cannot infer nx,nz from input; pass nx and nz explicitly")

        if dx is None or dz is None:
            dx = float(dx) if dx is not None else float(getattr(self, "DX", 1.0))
            dz = float(dz) if dz is not None else float(getattr(self, "DZ", 1.0))

        # FD Weights
        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        def D_x(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_x, axis=0, pad_mode=pad_mode)

        def D_z(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_z, axis=1, pad_mode=pad_mode)

        # Laplacian using FD
        def Laplacian(u):
            return D_x(D_x(u)) + D_z(D_z(u))

        # Build matrix-free operator
        def A_op(u):
            return u - alpha * Laplacian(u)

        # Matrix-free Conjugate Gradient
        def Conjugate_Gradient(b, x0=None, tol_local=tol, maxit=maxiter):
            n = b.size
            if x0 is None:
                x = xp.zeros_like(b)
            else:
                x = x0.astype(b.dtype, copy=True)

            r = b - A_op(x.reshape((nx, nz))).ravel()
            p = r.copy()
            rsold = float(xp.dot(r.conjugate(), r).real)

            if rsold == 0.0:
                return x, 0

            for k in range(maxit):
                Ap = A_op(p.reshape((nx, nz))).ravel()
                pAp = float(xp.dot(p.conjugate(), Ap).real)
                if pAp == 0.0:
                    if verbose:
                        self.log("[Laplace_Denoise] breakdown in Conjugate Gradient (pAp==0)")
                    break
                alpha_k = rsold / (pAp + 1e-30)
                x = x + alpha_k * p
                r = r - alpha_k * Ap
                rsnew = float(xp.dot(r.conjugate(), r).real)
                if rsnew ** 0.5 <= tol_local:
                    if verbose:
                        self.log(f"[Laplace_Denoise] Conjugate Gradient converged k={k+1}, res={rsnew ** 0.5:.3e}")
                    return x, 0
                p = r + (rsnew / (rsold + 1e-30)) * p
                rsold = rsnew
            if verbose:
                self.log(f"[Laplace_Denoise] Conjugate Gradient reached maxiter={maxit}, final resid={rsold ** 0.5:.3e}")
            return x, 1

        use_external_solver = False
        solver_cpu = None
        solver_gpu = None

        if xp is np and HAS_SCIPY and SciPyLOp is not None and scipy_cg is not None:
            def mv_scipy(v_numpy):
                v_np = np.asarray(v_numpy, dtype=np.float32)
                Av = A_op(xp.asarray(v_np).reshape((nx, nz))).ravel()
                return np.asarray(Av, dtype=np.float32)
            Aop_scipy = SciPyLop((nx * nz, nx * nz), matvec=mv_scipy, dtype=np.float32)
            use_external_solver = True
            solver_cpu = ("scipy", scipy_cg, Aop_scipy)
        elif xp is cp and HAS_CUPYX and cpx_linalg is not None and hasattr(cpx_linalg, "cg"):
            use_external_solver = True
            solver_gpu = ("cupyx", cpx_linalg.cg, None)

        def Solve_Field(f_in):
            f = xp.asarray(f_in, dtype=xp.float32)
            b = f.ravel()
            x0 = None
            if xp is np and use_external_solver and solver_cpu is not None and solver_cpu[0] == "scipy":
                scipy_cg_local = solver_cpu[1]
                Aop_local = solver_cpu[2]
                try:
                    x_sol, info = scipy_cg_local(Aop_local, b, x0=None, tol=tol, maxiter=maxiter)
                    return xp.asarray(x_sol.reshape((nx, nz)), dtype=xp.float32), int(info)
                except Exception:
                    if verbose:
                        self.log("[Laplace_Denoise] scipy.cg failed - falling back to internal Conjugate Gradient")
            x_sol, info = Conjugate_Gradient(b.astype(xp.float32), x0=None, tol_local=tol, maxit=maxiter)
            x_out = x_sol.reshape((nx, nz)).astype(xp.float32)
            return x_out, int(info)

        outputs = []
        infos = []
        for fld in fields:
            if fld is None:
                fld_arr = xp.zeros((nx, nz), dtype=xp.float32)
            else:
                fld_arr = xp.asarray(fld, dtype=xp.float32)
            denoised, info = Solve_Field(fld_arr)
            outputs.append(denoised)
            infos.append(info)

        if energy_normalize:
            emin, emax = float(energy_clip[0]), float(energy_clip[1])
            for i, (orig, den) in enumerate(zip(fields, outputs)):
                orig_arr = xp.asarray(orig, dtype=xp.float32)
                try:
                    E_orig = self.Image_Gaussian(orig_arr * orig_arr, xp.zeros_like(orig_arr), sigma=dx, truncate=3.0, mode=pad_mode, use_gpu=(xp is cp))
                    E_den = self.Image_Gaussian(den * den, xp.zeros_like(den), sigma=dx, truncate=3.0, mode=pad_mode, use_gpu=(xp is cp))
                except Exception:
                    kernel = xp.ones((3, 3), dtype=xp.float32) / 9.0
                    E_orig = self.Apply_1D_Shifted_Convolution(xp, self.Apply_1D_Shifted_Convolution(orig_arr * orig_arr, kernel[1,:], axis=0, pad_mode=pad_mode), kernel[:,1], axis=1, pad_mode=pad_mode)
                    E_den = self.Apply_1D_Shifted_Convolution(xp, self.Apply_1D_Shifted_Convolution(den * den, kernel[1,:], axis=0, pad_mode=pad_mode), kernel[:,1], axis=1, pad_mode=pad_mode)
                ratio = xp.sqrt((E_orig  1e-12) / (E_den + 1e-12))
                ratio = xp.clip(ratio, emin, emax)
                outputs[i] = (den * ratio).astype(xp.float32, copy=False)

        if len(outputs) == 1:
            out0 = outputs[0]
            if inplace:
                try:
                    fields[0][:] = out0
                    return out0
                except Exception:
                    return out0
            return out0
        else:
            if inplace:
                out_tuple = []
                for orig, den in zip(fields, outputs):
                    try:
                        orig[:] = den
                        out_tuple.append(orig)
                    except Exception:
                        out_tuple.append(den)
                return tuple


    def Cross_Correlation(
        self,
        u_fwd,
        u_bwd,
        image=None,
        nx: int | None = None,
        nz: int | None = None,
        layout: str = "auto",
        use_gpu: bool | None = None,
        tile: tuple | None = None,
        accumulate: bool = True,
    ):
        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        u_fwd_xp = xp.asarray(u_fwd, dtype=xp.float32)
        u_bwd_xp = xp.asarray(u_bwd, dtype=xp.float32)

        if u_fwd_xp.ndim != 2 or u_bwd_xp.ndim != 2:
            raise ValueError("Cross_Correlation: u_fwd and u_bwd must be 2D arrays")

        if u_fwd_xp.shape != u_bwd_xp.shape:
            if u_fwd_xp.shape == (u_bwd_xp.shape[1], u_bwd_xp.shape[0]):
                u_bwd_xp = u_bwd_xp.T
            else:
                raise ValueError(f"Cross_Correlation: incompatible shapes {u_fwd_xp.shape} vs {u_bwd_xp.shape}")

        shape = u_fwd_xp.shape
        if layout not in ("auto", "z_major", "x_major"):
            raise ValueError("layout must be 'auto','z_major' or 'x_major'")

        if layout == "x_major":
            u_fwd_proc = u_fwd_xp.T
            u_bwd_proc = u_bwd_xp.T
            inferred = "x_major"
        else:
            u_fwd_proc = u_fwd_xp
            u_bwd_proc = u_bwd_xp
            inferred = "z_major"

        nz_tot, nx_tot = u_fwd_proc.shape

        if nz is not None and nz != nz_tot:
            raise ValueError(f"Cross_Correlation: nz hint {nz} mismatches inferred nz {nz_tot}")
        if nx is not None and nx != nx_tot:
            raise ValueError(f"Cross_Correlation: nx hint {nx} mismatches inferred nx {nx_tot}")

        if image is None:
            image_xp = xp.zeros_like(u_fwd_proc, dtype=xp.float32)
        else:
            image_xp = xp.asarray(image, dtype=xp.float32)
            if image_xp.shape != u_fwd_proc.shape:
                if image_xp.T.shape == u_fwd_proc.shape:
                    image_xp = image_xp.T
                else:
                    raise ValueError("Cross_Correlation: provided image has incompatible shape")

        if tile is None:
            if accumulate:
                image_xp += u_fwd_proc * u_bwd_proc
            else:
                image_xp[...] = u_fwd_proc * u_bwd_proc
        else:
            tile_z, tile_x = int(tile[0]), int(tile[1])
            if tile_z <= 0 or tile_x <= 0:
                raise ValueError("Cross_Correlation: tile dims must be positive integers")
            if not accumulate:
                image_xp.fill(0.0)
            for z0 in range(0, nz_tot, tile_z):
                z1 = min(z0 + tile_z, nz_tot)
                for x0 in range(0, nx_tot, tile_x):
                    x1 = min(x0 + tile_x, nx_tot)
                    uf_block = u_fwd_proc[z0:z1, x0:x1]
                    ub_block = u_bwd_proc[z0:z1, x0:x1]
                    blk = uf_block * ub_block
                    if accumulate:
                        image_xp[z0:z1, x0:x1] += blk
                    else:
                        image_xp[z0:z1, x0:x1] = blk
                    if xp is cp and self.stream is not None:
                        self.stream.synchronize()

        return image_xp


    """
    def run(
        self,
        vp,
        pdata=None,
        dx=DX,
        dz=DZ,
        lxpad=LXPAD,
        rxpad=RXPAD,
        dt=None,
        snap_stride=STRIDE,
        imaging=True,
        random_boundary_input=RANDOM_BOUNDARY_INPUT,
        randwidth=RANDWIDTH,
        src_depth=SOURCE_DEPTH,
        rec_depth=RECEIVER_DEPTH,
        vs=None,
        rho=None,
        denoise_forward=False,
        denoise_params: dict | None = None,
        rec_type: str = "pressure",
        do_forward=True,
        do_backward=True,
        external_pdata=None,
        return_pdata=False,
        snap_dtype="float16",
        checkpoint_dir=None,
        use_mixed_precision=True,
        max_gpu_snap_bytes=None,
        survey_type: str = "land"
    ):
        # Backend
        if not self.has_cupy:
            raise RuntimeError("CuPy required for GPU execution in this run()")
        if survey_type not in ("land", "marine"):
            raise ValueError("survey_type must be 'land' or 'marine'")

        xp = cp
        device = int(self.device)

        if external_pdata is not None:
            pdata = external_pdata

        if pdata is None and not do_forward:
            raise ValueError("pdata is None but do_forward==False. Provide external_pdata or set do_forward=True.")

        if imaging and do_backward and not do_forward:
            if self.verbose:
                self.log(
                    "[RTM] imaging + backward requested but do_forward=False - enabling do_forward=True automatically.")
            do_forward = True

        if pdata is not None:
            nshot, totaltr, nt = pdata.shape
        else:
            nshot = NSHOT
            totaltr = int(min(600, vp.shape[0]))
            vmax = float(vp.max())
            h = dx * dz / math.sqrt(dx * dx + dz * dz)
            if dt is None:
                dt = 0.45 * h / vmax
            tmax = (vp.shape[1] * dz + (vp.shape[0] + lxpad + rxpad) * dx) / vmax
            nt = int(tmax / dt) + 1

        nxo, nz = vp.shape

        # Apply Random Boundary
        if random_boundary_input > 0:
            if self.verbose:
                self.log(f"[RTM] Applying Random Boundary to model with bc={random_boundary_input}")
            vp = self.Random_Boundary(vp, bc_max=random_boundary_input, inplace=False)
            if vs is not None:
                vs = self.Random_Boundary(vs, bc_max=random_boundary_input, inplace=False)
            if rho is not None:
                rho = self.Random_Boundary(rho, bc_max=random_boundary_input, inplace=False)

        # CFL and dt
        vmax = float(vp.max())
        h = dx * dz / math.sqrt(dx * dx + dz * dz)
        if dt is None:
            dt = 0.45 * h / vmax
        dt_cfl = 0.45 * h / vmax
        if dt > dt_cfl:
            dt = dt_cfl

        # Recompute modelling extent
        nx_global = nxo + lxpad + rxpad
        tmax = (nz * dz + nx_global * dx) / vmax
        ntnewmax = int(tmax / dt) + 1
        ntnewrec = ntnewmax

        # Wavelet
        wf = self.Ricker_Wavelet(freq=25.0, dt=dt, nt=nt)
        wf_np = cp.asnumpy(wf) if isinstance(wf, cp.ndarray) else np.asarray(wf)
        if np.max(np.abs(wf_np)) > 0:
            wf_np = wf_np / np.max(np.abs(wf_np))
        wf_np = wf_np * 1.0
        d_w = cp.asarray(wf_np, dtype=cp.float32)

        if self.verbose:
            self.log(
                f"[RTM] nshot={nshot}, ntr={totaltr}, nt(in)={nt}, ntmod={ntnewmax}, dt={dt:.3e}, survey={survey_type}")

        nx_max = nxo + lxpad + rxpad
        n_snapshots = max(1, (ntnewmax + snap_stride - 1) // snap_stride)

        # Snapshot dtype (GPU)
        snap_dtype_cp = cp.float16 if (snap_dtype == "float16" and use_mixed_precision) else cp.float32
        snap_itemsize = 2 if snap_dtype_cp == cp.float16 else 4
        approx_snap_bytes = n_snapshots * nx_max * nz * snap_itemsize

        # Decide checkpoint
        do_checkpoint = False
        if max_gpu_snap_bytes is not None and approx_snap_bytes > max_gpu_snap_bytes:
            do_checkpoint = True
        if checkpoint_dir is not None:
            do_checkpoint = True

        # Prepare outputs
        if (vs is not None) and (rho is not None):
            Acoustic_mode = True
            image_PP = cp.zeros((nx_max, nz), dtype=cp.float32)
            image_PS = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum_PP = cp.zeros_like(image_PP)
        else:
            Acoustic_mode = False
            image_global = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum = cp.zeros_like(image_global)

        if denoise_params is None:
            denoise_params = {"sigma": 1.5, "local_sigma": 3.0, "gamma": 1.0}

        # Precompute FD weights
        _ = self.Ensure_FD_Weights(3, dx)

        # Pinned buffer for receiver copy
        pinned_host_buf = None
        pdata_host_out = None
        if do_forward and pdata is None and return_pdata:
            pdata_host_out = np.zeros((nshot, totaltr, ntnewmax), dtype=np.float32)

        def Record_Trace_Array(target, field2d, lxpad, ntr, rec_depth):
            if isinstance(rec_depth, (int, np.integer, cp.integer)):
                depth_idx = int(rec_depth)
                target[:, :] = target[:, :]
                return field2d[lxpad:lxpad + ntr, depth_idx]
            else:
                rd = cp.asarray(rec_depth, dtype=cp.int32)
                if rd.size == 1:
                    rd = cp.full((ntr,), int(rd.ravel()[0]), dtype=cp.int32)
                elif rd.size != ntr:
                    rd = cp.resize(rd, (ntr,))
                ix = cp.arange(ntr, dtype=cp.int32) + lxpad
                out = cp.zeros((ntr,), dtype=field2d.dtype)
                valid = (ix >= 0) & (ix < field2d.shape[0]) & (rd >= 0) & (rd < field2d.shape[1])
                if valid.any():
                    out[valid] = field2d[ix[valid], rd[valid]]
                return out

        #----------------
        # SHOT LOOP
        #----------------
        for ishot in tqdm(range(nshot), desc="[RTM] Shot", unit="shot"):
            if self.verbose:
                self.log(f"[RTM] Shot {ishot + 1}/{nshot}")

            # forward stress snapshot containers (we store stresses, not velocities)
            d_sxx_fwd_snap_shot = None
            d_szz_fwd_snap_shot = None
            snap_memmap = None

            # Geometry and traces
            if pdata is not None:
                traces = pdata[ishot]
                ntr = traces.shape[0]
            else:
                ntr = int(min(600, nxo))

            # Place source in model center
            sx = nxo // 2
            ix22 = max(0, sx - ntr // 2)
            nx = ntr + lxpad + rxpad
            src_local = sx - ix22 + lxpad

            # Build local vp block
            src_ixs = np.clip(np.arange(ix22 - lxpad, ix22 - lxpad + nx), 0, nxo - 1)
            vp_block = vp[src_ixs, :].astype(np.float32)

            # Acoustic blocks
            if Acoustic_mode:
                vs_block = (vs[src_ixs, :].astype(np.float32) if hasattr(vs, "shape") and vs.shape == vp.shape else (
                    vs if np.isscalar(vs) else vs[src_ixs, :].astype(np.float32)))
                rho_block = (
                    rho[src_ixs, :].astype(np.float32) if hasattr(rho, "shape") and rho.shape == vp.shape else (
                        rho if np.isscalar(rho) else rho[src_ixs, :].astype(np.float32)))
                mu_block = rho_block * (vs_block ** 2)
                lam_block = rho_block * (vp_block ** 2 - 2.0 * (vs_block ** 2))

            # Move to device and allocate fields
            with cp.cuda.Device(device):
                d_vp = cp.asarray(vp_block, dtype=cp.float32)
                if Acoustic_mode:
                    d_vs = cp.asarray(vs_block, dtype=cp.float32)
                    d_rho = cp.asarray(rho_block, dtype=cp.float32)
                    d_mu = cp.asarray(mu_block, dtype=cp.float32)
                    d_lam = cp.asarray(lam_block, dtype=cp.float32)

                    vx = cp.zeros((nx, nz), dtype=cp.float32)
                    vz = cp.zeros_like(vx)
                    sxx = cp.zeros_like(vx)
                    szz = cp.zeros_like(vx)
                    sxz = cp.zeros_like(vx)

                    # Pressure-equivalent buffers for ABC (Acoustic)
                    p_pres = cp.zeros((nx, nz), dtype=cp.float32)
                    pm_pres = cp.zeros((nx, nz), dtype=cp.float32)

                    # Backward fields
                    vx_rec = cp.zeros_like(vx)
                    vz_rec = cp.zeros_like(vx)
                    sxx_rec = cp.zeros_like(vx)
                    szz_rec = cp.zeros_like(vx)
                    sxz_rec = cp.zeros_like(vx)

                else:
                    d_v = cp.asarray(vp_block, dtype=cp.float32)
                    d_pn = cp.zeros_like(d_v)
                    d_pp = cp.zeros_like(d_v)
                    d_pnrec = cp.zeros_like(d_v)
                    d_pprec = cp.zeros_like(d_v)

                # Per-shot traces buffer on device
                pdata_shot_d = cp.zeros((ntr, ntnewmax), dtype=cp.float32)

                # Snapshot container (store stresses sxx,szz)
                if do_forward:
                    if do_checkpoint:
                        if checkpoint_dir is None:
                            checkpoint_dir = "/tmp/rtm_chk"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        dtype_np = np.float16 if snap_dtype_cp == cp.float16 else np.float32
                        snap_sxx_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps_sxx.dat")
                        snap_szz_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps_szz.dat")
                        snap_memmap_sxx = np.memmap(snap_sxx_path, dtype=dtype_np, mode="w+",
                                                    shape=(n_snapshots, nx, nz))
                        snap_memmap_szz = np.memmap(snap_szz_path, dtype=dtype_np, mode="w+",
                                                    shape=(n_snapshots, nx, nz))
                        snap_memmap = (snap_memmap_sxx, snap_memmap_szz)
                    else:
                        d_sxx_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)
                        d_szz_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)

                # Prepare pinned host buffer once
                bytes_per_shot = ntr * ntnewmax * np.dtype(np.float32).itemsize
                if pinned_host_buf is None:
                    try:
                        pinned_host_buf = cp.cuda.alloc_pinned_memory(min(bytes_per_shot, 2 ** 30))
                    except Exception:
                        pinned_host_buf = None
                if pinned_host_buf is not None:
                    host_flat = np.frombuffer(pinned_host_buf, dtype=np.float32, count=ntr * ntnewmax)
                    host_view = host_flat.reshape((ntr, ntnewmax))
                else:
                    host_view = None

                #------------------------
                # Forward Propagation
                #------------------------
                if do_forward:
                    for it in range(ntnewmax):
                        if Acoustic_mode:
                            vx, vz, sxx, szz, sxz = self.Acoustic_TimeDomain_Propagation(
                                vx, vz, sxx, szz, sxz,
                                d_rho, d_lam, d_mu,
                                nx, dx, nz, dz, dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=True
                            )

                            if self.verbose and (it % 50 == 0):
                                energy = float(
                                    cp.sum(vx * vx + vz * vz + sxx * sxx + szz * szz + sxz * sxz).astype(cp.float64))
                                self.log(f"[Forward] it={it} energy={energy:.6e}")

                            # Inject source into stresses
                            if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                                val = float(d_w[it])
                                sxx[src_local, src_depth] += val
                                szz[src_local, src_depth] += val

                            # Compute pressure-equivalent (for recording)
                            p_pres[...] = 0.5 * (sxx + szz)

                            # Apply Absorbing_Boundary_Condition
                            try:
                                pp_tmp = p_pres.copy()
                                p_pres_mod = self.Absorbing_Boundary_Condition(
                                    nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                    dvv=d_vp, od=None, p=p_pres, pm=pm_pres,
                                    pp=pp_tmp, abs_flags=(1, 1, 1, 1)
                                )
                                mask_boundary = (p_pres_mod != p_pres)
                                if mask_boundary.any():
                                    sxx[mask_boundary] = p_pres_mod[mask_boundary]
                                    szz[mask_boundary] = p_pres_mod[mask_boundary]
                                pm_pres[...] = p_pres.copy()
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_vp, sxx, sxx, nx, nz, dx, dz, dt, iz=1)
                                        self.k["absorbgpuup"](d_vp, szz, szz, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            # Record pressure-equivalent
                            pres = 0.5 * (sxx + szz)
                            # pdata_shot_d[:, it] = pres[lxpad:lxpad + ntr, rec_depth]
                            pdata_shot_d[:, it] = Record_Trace_Array(pdata_shot_d[:, it:it + 1], pres, lxpad, ntr,
                                                                     rec_depth)

                            if it % 100 == 0 and self.verbose:
                                tr_max = float(cp.max(cp.abs(pdata_shot_d)).astype(cp.float32))
                                self.log(f"[Forward] recorded traces max abs so far = {tr_max:.5e}")

                        else:
                            # Acoustic path
                            # self.k["TimeDomain_Propagation"](d_v, d_pn, d_pp, nx=nx, dx=dx, nz=nz, dz=dz, dt=dt)
                            # if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                            #     d_pp[src_local, src_depth] += d_w[it] * float(dt)

                            # Swap buffers and record
                            # d_pn, d_pp = d_pp, d_pn

                            # ----------------------------
                            # Helmholtz-based Forward
                            # ----------------------------
                            freq0 = FREQUENCY0
                            try:
                                ux_c = cp.zeros((nx, nz), dtype=cp.complex64)
                                uz_c = d_pn.astype(cp.complex64)

                                if 'd_rho' in locals():
                                    rho_local = d_rho
                                else:
                                    rho_local = cp.ones((nx, nz), dtype=cp.float32)

                                mu_local = cp.full((nx, nz), 1e-6, dtype=cp.float32)
                                lam_local = rho_local * (d_v ** 2) - 2.0 * mu_local

                                # Apply Helmholtz Acoustic Propagation
                                ux_out, uz_out, info_h = self.Helmholtz_Acoustic_Propagation(
                                    ux_c, uz_c, rho_local, lam_local, mu_local,
                                    nx, dx, nz, dz, freq=freq0,
                                    fd_order_radius=3,
                                    use_gpu=True,
                                    pad_mode='reflect',
                                    absorb_R=0,
                                    src=None,
                                    inplace=True,
                                    use_external_abc=True,
                                    solver_tol=1e-6,
                                    solver_maxiter=MAXITER,
                                    verbose=self.verbose
                                )

                                w_x_np = self.Ensure_FD_Weights(3, dx)
                                w_z_np = self.Ensure_FD_Weights(3, dz)
                                w_x = cp.asarray(w_x_np, dtype=cp.float32)
                                w_z = cp.asarray(w_z_np, dtype=cp.float32)

                                dux_dx = self.Apply_1D_Shifted_Convolution(cp, ux_out.real.astype(cp.float32), w_x,
                                                                           axis=0, pad_mode='reflect')
                                duz_dz = self.Apply_1D_Shifted_Convolution(cp, uz_out.real.astype(cp.float32), w_z,
                                                                           axis=1, pad_mode='reflect')

                                K_local = rho_local * (d_v ** 2)
                                p_complex = -(K_local.astype(cp.complex64)) * (
                                            dux_dx.astype(cp.complex64) + duz_dz.astype(cp.complex64))

                                d_pp[...] = cp.real(p_complex).astype(cp.float32)

                                if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                                    d_pp[src_local, src_depth] += d_w[it] * float(dt)

                                d_pn, d_pp = d_pp, d_pn

                                try:
                                    abs_flags = (1, 1, 1, 1)
                                    pp_tmp = d_pn.copy()
                                    d_pn = self.Absorbing_Boundary_Condition(
                                        nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                        dvv=d_v, od=None, p=d_pn, pm=d_pp, pp=pp_tmp,
                                        abs_flags=abs_flags
                                    )
                                except Exception:
                                    try:
                                        if "absorbgpuup" in self.k:
                                            self.k["absorbgpuup"](d_v, d_pn, d_pn, nx, nz, dx, dz, dt, iz=1)
                                    except Exception:
                                        pass

                                # pdata_shot_d[:, it] = d_pn[lxpad:lxpad + ntr, rec_depth]

                                pdata_shot_d[:, it] = Record_Trace_Array(d_pn, lxpad, ntr, rec_depth)

                            # except Exception as ex_helm_forward:
                            #     if self.verbose:
                            #         self.log("[Forward][Helmhotz] fallback to time-domain due to:", ex_helm_forward)
                            #     self.k["TimeDomain_Propagation"](d_v, d_pn, d_pp, nx=nx, dx=dx, nz=nz, dz=dz, dt=dt)
                            #     if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                            #         d_pp[src_local, src_depth] += d_w[it] * float(dt)
                            #     d_pn, d_pp = d_pp, d_pn
                            #     try:
                            #         abs_flags = (1, 1, 1, 1)
                            #         pp_tmp = d_pn.copy()
                            #         d_pn = self.Absorbing_Boundary_Condition(
                            #             nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                            #             dvv=d_v, od=None, p=d_pn, pm=d_pp, pp=pp_tmp,
                            #             abs_flags=abs_flags
                            #         )
                            #     except Exception:
                            #         try:
                            #             if "absorbgpuup" in self.k:
                            #                 self.k["absorbgpuup"](d_v, d_pn, d_pn, nx, nz, dx, dz, dt, iz=1)
                            #         except Exception:
                            #             pass
                            #     pdata_shot_d[:, it] = d_pn[lxpad:lxpad + ntr, rec_depth]

                            except Exception as ex_helm_forward:
                                if self.verbose:
                                    self.log("[Forward][Helmholtz] retry ")

                                ux_c = cp.zeros((nx, nz), dtype=cp.complex64)
                                uz_c = d_pn.astype(cp.complex64)

                                if 'd_rho' in locals():
                                    rho_local = d_rho
                                else:
                                    rho_local = cp.ones((nx, nz), dtype=cp.float32)

                                mu_local = cp.full((nx, nz), 1e-6, dtype=cp.float32)
                                lam_local = rho_local * (d_v ** 2) - 2.0 * mu_local

                                ux_out, uz_out, info_h = self.Helmholtz_Acoustic_Propagation(
                                    ux_x, uz_c,
                                    rho_local, lam_local, mu_local,
                                    nx, dx, nz, dz,
                                    freq=FREQUENCY0,
                                    fd_order_radius=3,
                                    use_gpu=True,
                                    pad_mode='reflect',
                                    absorb_R=0,
                                    src=None,
                                    inplace=True,
                                    use_external_abc=True,
                                    solver_tol=5e-5,
                                    solver_maxiter=MAXITER,
                                    verbose=self.verbose
                                )

                                w_x = cp.asarray(self.Ensure_FD_Weights(3, dx), dtype=cp.float32)
                                w_z = cp.asarray(self.Ensure_FD_Weights(3, dz), dtype=cp.float32)

                                dux_dx = self.Apply_1D_Shifted_Convolution(cp, ux_out.real, w_x, axis=0,
                                                                           pad_mode='reflect')
                                duz_dz = self.Apply_1D_Shifted_Convolution(cp, uz_out.real, w_z, axis=1,
                                                                           pad_mode='reflect')

                                K_local = rho_local * (d_v ** 2)
                                p_complex = -(K_local.astype(cp.complex64)) * (dux_dx + duz_dz)

                                d_pp[...] = cp.real(p_complex).astype(cp.float32)

                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_v, d_pp, d_pp, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                                d_pn, d_pp = d_pp, d_pn

                                # Record trace
                                # pdata_shot_d[:, it] = d_pn[lxpad:lxpad + ntr, rec_depth]
                                pdata_shot_d[:, it] = Record_Trace_Array(d_pn, lxpad, ntr, rec_depth)

                            # Apply Absorbing_Boundary_Condition
                            try:
                                abs_flags = (1, 1, 1, 1)
                                pp_tmp = d_pn.copy()
                                d_pn = self.Absorbing_Boundary_Condition(
                                    nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                    dvv=d_v, od=None, p=d_pn, pm=d_pp, pp=pp_tmp,
                                    abs_flags=abs_flags
                                )
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_v, d_pn, d_pn, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            pdata_shot_d[:, it] = d_pn[lxpad:lxpad + ntr, rec_depth]

                            if it % 100 == 0 and self.verbose:
                                tr_max = float(cp.max(cp.abs(pdata_shot_d)).astype(cp.float32))
                                self.log(f"[Forward] recorded traces max abs so far = {tr_max:.5e}")

                        # Snapshotting (store stresses sxx,szz)
                        if imaging and (it % snap_stride == 0):
                            sid = it // snap_stride

                            if do_checkpoint:
                                if self.stream is not None:
                                    self.stream.synchronize()
                                else:
                                    cp.cuda.runtime.deviceSynchronize()
                                if snap_dtype_cp == cp.float16:
                                    arr_sxx = cp.asnumpy(sxx.astype(cp.float16))
                                    arr_szz = cp.asnumpy(szz.astype(cp.float16))
                                else:
                                    arr_sxx = cp.asnumpy(sxx.astype(cp.float32))
                                    arr_szz = cp.asnumpy(szz.astype(cp.float32))
                                snap_memmap[0][sid, :, :] = arr_sxx
                                snap_memmap[1][sid, :, :] = arr_szz
                            else:
                                if snap_dtype_cp == cp.float16:
                                    d_sxx_fwd_snap_shot[sid, :, :] = sxx.astype(cp.float16)
                                    d_szz_fwd_snap_shot[sid, :, :] = szz.astype(cp.float16)
                                else:
                                    d_sxx_fwd_snap_shot[sid, :, :] = sxx
                                    d_szz_fwd_snap_shot[sid, :, :] = szz

                    # Return pdata per-shot to host buffer
                    if return_pdata and pdata_host_out is not None and host_view is not None:
                        try:
                            with (self.stream or cp.cuda.Stream.null):
                                pdata_shot_d.get(out=host_view, stream=(self.stream or None))
                            if self.stream is not None:
                                self.stream.synchronize()
                            else:
                                cp.cuda.runtime.deviceSynchronize()
                            pdata_host_out[ishot] = host_view.copy()
                        except Exception:
                            pdata_host_out[ishot] = cp.asnumpy(pdata_shot_d)

                else:
                    if self.verbose:
                        self.log("[RTM] forward skipped for this shot")

                # -------------------------------------
                # Backward Propagation and Imaging
                # -------------------------------------
                if do_backward:
                    if do_forward and pdata is None:
                        traces_gpu = pdata_shot_d
                    elif pdata is not None:
                        traces_gpu = cp.asarray(pdata[ishot], dtype=cp.float32)
                    else:
                        if pdata_host_out is not None:
                            traces_gpu = cp.asarray(pdata_host_out[ishot], dtype=cp.float32)
                        else:
                            raise RuntimeError("No traces available for backward propagation")

                    if traces_gpu is None:
                        raise RuntimeError("traces_gpu is None after selection")
                    if traces_gpu.ndim == 1:
                        traces_gpu = traces_gpu.reshape((ntr, -1))
                    elif traces_gpu.ndim == 2:
                        pass
                    else:
                        raise RuntimeError(f"Unexpected traces_gpu.ndim = {traces_gpu.ndim}")

                    if traces_gpu.shape[0] != ntr and traces_gpu.shape[1] == ntr:
                        traces_gpu = traces_gpu.T  # transpose untuk jadi (ntr, nt)

                    if self.verbose:
                        try:
                            self.log("[Backward] trace max/min:", float(traces_gpu.max()), float(traces_gpu.min()))
                        except Exception:
                            self.log("[Backward] trace stats unavailable (maybe shape issue)")

                    if Acoustic_mode:
                        vx_rec.fill(0.0)
                        vz_rec.fill(0.0)
                        sxx_rec.fill(0.0)
                        szz_rec.fill(0.0)
                        sxz_rec.fill(0.0)

                        pres_tmp = cp.empty_like(sxx_rec)
                        pres_before = cp.empty_like(sxx_rec)

                        # Handle array rec_depth variants
                        if isinstance(rec_depth, (int, np.integer)):
                            rec_depth_is_scalar = True
                            rec_depth_val = int(rec_depth)
                        else:
                            rec_depth_arr = cp.asarray(rec_depth, dtype=cp.int32) if not isinstance(rec_depth,
                                                                                                    cp.ndarray) else rec_depth.astype(
                                cp.int32)
                            if rec_depth_arr.size != ntr:
                                if rec_depth_arr.size == 1:
                                    rec_depth_arr = cp.full((ntr,), int(rec_depth_arr.ravel()[0]), dtype=cp.int32)
                                else:
                                    rec_depth_arr = cp.resize(rec_depth_arr, (ntr,))
                            rec_depth_is_scalar = False

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            ix_arr = cp.arange(ntr, dtype=cp.int32)
                            ix_global = ix_arr + lxpad

                            # Inject recorded trace into stresses (adjoint)
                            valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                            if valid.any():
                                idx = ix_global[valid]
                                amp = trace_t[valid]
                                if isinstance(rec_depth, (int, np.integer)):
                                    depth_idx = rec_depth
                                else:
                                    depth_idx = rec_depth[valid].astype(cp.int32)
                                sxx_rec[idx, depth_idx] += amp
                                szz_rec[idx, depth_idx] += amp

                            valid = (ix_global >= 0) & (ix_global < nx)

                            if rec_depth_is_scalar:
                                depth_ok = (rec_depth_val >= 0) and (rec_depth_val < nz)
                                if depth_ok and valid.any():
                                    idx = ix_global[valid]
                                    amp = trace_t[valid]
                                    sxx_rec[idx, rec_depth_val] += amp
                                    szz_rec[idx, rec_depth_val] += amp
                            else:
                                depth_mask = (rec_depth_arr >= 0) & (rec_depth_arr < nz)
                                valid = valid & depth_mask
                                if valid.any():
                                    idx = ix_global[valid]
                                    amp = trace_t[valid]
                                    depth_idx = rec_depth_arr[valid].astype(cp.int32)
                                    sxx_rec[idx, depth_idx] += amp
                                    szz_rec[idx, depth_idx] += amp

                            # Step backward one time
                            vx_rec, vz_rec, sxx_rec, szz_rec, sxz_rec = self.Acoustic_TimeDomain_Propagation(
                                vx_rec, vz_rec, sxx_rec, szz_rec, sxz_rec,
                                d_rho, d_lam, d_mu,
                                nx, dx, nz, dz, dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=True
                            )

                            # Apply ABC to backward fields
                            try:
                                pres_before[:] = 0.5 * (sxx_rec + szz_rec)
                                pres_tmp[:] = pres_before

                                p_rec_mod = self.Absorbing_Boundary_Condition(
                                    nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                    dvv=d_vp, od=None, p=pres_tmp, pm=pm_pres, pp=pres_tmp.copy(),
                                    abs_flags=(1, 1, 1, 1)
                                )

                                if p_rec_mod is None:
                                    p_rec_mod = pres_tmp

                                mask_b = (p_rec_mod != pres_before)
                                if mask_b.any():
                                    sxx_rec[mask_b] = p_rec_mod[mask_b]
                                    szz_rec[mask_b] = p_rec_mod[mask_b]

                                pm_pres[...] = pres_before.copy()
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_vp, sxx_rec, sxx_rec, nx, nz, dx, dz, dt, iz=1)
                                        self.k["absorbgpuup"](d_vp, szz_rec, szz_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            if "absorbgpuup" in self.k:
                                try:
                                    self.k["absorbgpuup"](d_vp, vx_rec, vx_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, vz_rec, vz_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, sxx_rec, sxx_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, szz_rec, szz_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, sxz_rec, sxz_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            # ----------------------------------------
                            # Imaging Condition (Pressure-Pressure)
                            # ----------------------------------------
                            if imaging and (it % snap_stride == 0):
                                sid = it // snap_stride

                                max_sid = (d_sxx_fwd_snap_shot.shape[0] - 1) if (d_sxx_fwd_snap_shot is not None) else (
                                            snap_memmap[0].shape[0] - 1)
                                if sid < 0 or sid > max_sid:
                                    raise RuntimeError(f"Snapshot id out of range: sid={sid}, max={max_sid}")

                                if do_checkpoint:
                                    sxx_host = snap_memmap[0][sid, :, :].astype(np.float32)
                                    szz_host = snap_memmap[1][sid, :, :].astype(np.float32)
                                    sxx_fwd_local = cp.asarray(sxx_host)
                                    szz_fwd_local = cp.asarray(szz_host)
                                else:
                                    if d_sxx_fwd_snap_shot is None or d_szz_fwd_snap_shot is None:
                                        raise RuntimeError(
                                            "Forward stress snapshots missing: ensure do_forward=True, imaging=True and not do_checkpoint"
                                            " OR enable checkpoint snapshots"
                                        )
                                    sxx_fwd_local = d_sxx_fwd_snap_shot[sid, :, :].astype(cp.float32)
                                    szz_fwd_local = d_szz_fwd_snap_shot[sid, :, :].astype(cp.float32)

                                p_fwd = 0.5 * (sxx_fwd_local + szz_fwd_local)
                                p_bwd = 0.5 * (sxx_rec + szz_rec)

                                d_result_PP = p_fwd * p_bwd

                                image_PP[ix22: ix22 + nx, :] += d_result_PP[ix22: ix22 + nx, :]

                                illum_PP[ix22: ix22 + nx, :] += p_fwd[ix22: ix22 + nx, :] * p_fwd[ix22: ix22 + nx, :]

                    else:
                        # Acoustic backward path
                        d_pnrec.fill(0.0)
                        d_pprec.fill(0.0)
                        d_result = cp.zeros((nx, nz), dtype=cp.float32)

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            try:
                                if "receivedata" in self.k:
                                    self.k["receivedata"](d_pnrec, trace_t, nz, ntr, lxpad, 0)
                                else:
                                    ix_arr = cp.arange(ntr, dtype=cp.int32)
                                    ix_global = ix_arr + lxpad
                                    valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                                    if valid.any():
                                        idx = ix_global[valid]
                                        depth_idx = rec_depth[valid].astype(cp.int32)
                                        amp = trace_t[valid]
                                        d_pnrec[idx, depth_idx] += amp
                            except Exception:
                                ix_arr = cp.arange(ntr, dtype=cp.int32)
                                ix_global = ix_arr + lxpad
                                valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                                if valid.any():
                                    idx = ix_global[valid]
                                    depth_idx = rec_depth[valid].astype(cp.int32)
                                    amp = trace_t[valid]
                                    d_pnrec[idx, depth_idx] += amp

                            # self.k["TimeDomain_Propagation"](d_v, d_pnrec, d_pprec, nx=nx, dx=dx, nz=nz, dz=dz, dt=dt)
                            # if "absorbgpuup" in self.k:
                            #     try:
                            #         self.k["absorbgpuup"](d_v, d_pnrec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                            #     except Exception:
                            #         pass

                            # if imaging and (it % snap_stride == 0):
                            #     sid = it // snap_stride
                            #     if do_checkpoint:
                            #         u_fwd_local = cp.asarray(snap_memmap[sid, :, :].astype(np.float32))
                            #     else:
                            #         if d_sxx_fwd_snap_shot is None:
                            #             raise RuntimeError(
                            #                 "Forward snapshots are not available for imaging. Ensure do_forward=True and imaging=True.")
                            #         u_fwd_local = d_sxx_fwd_snap_shot[sid, lxpad:lxpad + nx, :].astype(cp.float32)

                            #     corr = self.Cross_Correlation(u_fwd_local, d_pnrec, image=None, use_gpu=True)
                            #     d_result += corr
                            #     illum[ix22:ix22 + nx, :] += u_fwd_local * u_fwd_local

                            # d_pnrec, d_pprec = d_pprec, d_pnrec

                            freq0 = FREQUENCY0
                            try:
                                ux_c = cp.zeros((nx, nz), dtype=cp.complex64)
                                uz_c = d_pnrec.astype(cp.complex64)

                                if 'd_rho' in locals():
                                    rho_local = d_rho
                                else:
                                    rho_local = cp.ones((nx, nz), dtype=cp.float32)

                                mu_local = cp.full((nx, nz), 1e-6, dtype=cp.float32)
                                lam_local = rho_local * (d_v ** 2) - 2.0 * mu_local

                                ux_out, uz_out, info_h = self.Helmholtz_Acoustic_Propagation(
                                    ux_c, uz_c, rho_local, lam_local, mu_local,
                                    nx, dx, nz, dz, freq=freq0,
                                    fd_order_radius=3,
                                    use_gpu=True,
                                    pad_mode='reflect',
                                    absorb_R=0,
                                    src=None,
                                    inplace=True,
                                    use_external_abc=True,
                                    solver_tol=1e-6,
                                    solver_maxiter=MAXITER,
                                    verbose=self.verbose
                                )

                                w_x_np = self.Ensure_FD_Weights(3, dx)
                                w_z_np = self.Ensure_FD_Weights(3, dz)
                                w_x = cp.asarray(w_x_np, dtype=cp.float32)
                                w_z = cp.asarray(w_z_np, dtype=cp.float32)

                                dux_dx = self.Apply_1D_Shifted_Convolution(cp, ux_out.real.astype(cp.float32), w_x, axis=0, pad_mode='reflect')
                                duz_dz = self.Apply_1D_Shifted_Convolution(cp, uz_out.real.astype(cp.float32), w_z, axis=1, pad_mode='reflect')

                                K_local = rho_local * (d_v ** 2)
                                p_complex = - (K_local.astype(cp.complex64)) * (
                                            dux_dx.astype(cp.complex64) + duz_dz.astype(cp.complex64))

                                # Update Adjoint pressure
                                d_pprec[...] = cp.real(p_complex).astype(cp.float32)

                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_v, d_pprec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                                d_pnrec, d_pprec = d_pprec, d_pnrec

                            except Exception as ex_helm_backward:
                                if self.verbose:
                                    self.log("[[Backward][Helmholtz] fallback to time-domain due to:", ex_helm_backward)
                                self.k["TimeDomain_Propagation"](d_v, d_pnrec, d_pprec, nx=nx, dx=dx, nz=nz, dz=dz,
                                                                 dt=dt)
                                if "absorbgpuup" in self.k:
                                    try:
                                        self.k["absorbgpuup"](d_v, d_pnrec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                                    except Exception:
                                        pass
                                d_pnrec, d_pprec = d_pprec, d_pnrec

                        image_global[ix22:ix22 + nx, :] += d_result

                    # cleanup per-shot GPU objects
                    try:
                        del d_vp
                        if Acoustic_mode:
                            del d_vs, d_rho, d_mu, d_lam
                        del pdata_shot_d
                        if (d_sxx_fwd_snap_shot is not None) and (not do_checkpoint):
                            del d_sxx_fwd_snap_shot, d_szz_fwd_snap_shot
                    except Exception:
                        pass

                    cp.cuda.Stream.null.synchronize()

        # End shot loop

        # Normalize and final denoise
        if Acoustic_mode:
            image_PP = self.Preconditioning(image_PP, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)
            image_PS = self.Preconditioning(image_PS, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)

            out_cp = (image_PP + 0.5 * image_PS) if Acoustic_mode else image_global
            if isinstance(out_cp, cp.ndarray):
                out_cp = cp.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
                maxv = float(cp.max(cp.abs(out_cp)))
                out = cp.asnumpy(out_cp / (maxv if maxv > 1e-12 else 1.0))
            else:
                out_np = np.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
                maxv = float(np.max(np.abs(out_np)))
                out = out_np / (maxv if maxv > 1e-12 else 1.0)

            if np.max(np.abs(out)) < 1e-10:
                if getattr(self, "verbose", False):
                    self.log("[RTM DEBUG] final image is nearly zero after imaging. Check forward snapshots, traces and illumination.")

        else:
            image_global = image_global / (illum + EPS)

            image_global = self.Preconditioning(image_global, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)

            d_out = cp.zeros_like(image_global)

            out_cp = image_global
            out_cp = cp.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
            maxv = float(cp.max(cp.abs(out_cp)))
            out = cp.asnumpy(out_cp / (maxv if maxv > 1e-12 else 1.0))

        if return_pdata and pdata_host_out is not None:
            return out, pdata_host_out
        return out
    """


    def run(
        self,
        vp,
        pdata=None,
        dx=DX,
        dz=DZ,
        lxpad=LXPAD,
        rxpad=RXPAD,
        dt=None,
        snap_stride=STRIDE,
        imaging=True,
        random_boundary_input=RANDOM_BOUNDARY_INPUT,
        randwidth=RANDWIDTH,
        src_depth=SOURCE_DEPTH,
        rec_depth=RECEIVER_DEPTH,
        vs=None,
        rho=None,
        denoise_forward=False,
        denoise_params: dict | None = None,
        rec_type: str = "pressure",
        do_forward=True,
        do_backward=True,
        external_pdata=None,
        return_pdata=False,
        snap_dtype="float16",
        checkpoint_dir=None,
        use_mixed_precision=True,
        max_gpu_snap_bytes=None,
        survey_type: str = "land"
    ):
        # Backend
        if not self.has_cupy:
            raise RuntimeError("CuPy required for GPU execution in this run()")
        if survey_type not in ("land", "marine"):
            raise ValueError("survey_type must be 'land' or 'marine'")

        xp = cp
        device = int(self.device)

        if external_pdata is not None:
            pdata = external_pdata

        if pdata is None and not do_forward:
            raise ValueError("pdata is None but do_forward==False. Provide external_pdata or set do_forward=True.")

        if imaging and do_backward and not do_forward:
            if self.verbose:
                self.log("[RTM] imaging + backward requested but do_forward=False - enabling do_forward=True automatically.")
            do_forward = True

        if pdata is not None:
            nshot, totaltr, nt = pdata.shape
        else:
            nshot = NSHOT
            totaltr = int(min(600, vp.shape[0]))
            vmax = float(vp.max())
            h = dx * dz / math.sqrt(dx * dx + dz * dz)
            if dt is None:
                dt = 0.45 * h / vmax
            tmax = (vp.shape[1] * dz + (vp.shape[0] + lxpad + rxpad) * dx) / vmax
            nt = int(tmax / dt) + 1

        nxo, nz = vp.shape

        # Apply Random Boundary
        if random_boundary_input > 0:
            if self.verbose:
                self.log(f"[RTM] Applying Random Boundary to model with bc={random_boundary_input}")
            vp = self.Random_Boundary(vp, bc_max=random_boundary_input, inplace=False)
            if vs is not None:
                vs = self.Random_Boundary(vs, bc_max=random_boundary_input, inplace=False)
            if rho is not None:
                rho = self.Random_Boundary(rho, bc_max=random_boundary_input, inplace=False)

        # CFL and dt
        vmax = float(vp.max())
        h = dx * dz / math.sqrt(dx * dx + dz * dz)
        if dt is None:
            dt = 0.45 * h / vmax
        dt_cfl = 0.45 * h / vmax
        if dt > dt_cfl:
            dt = dt_cfl

        # Recompute modelling extent
        nx_global = nxo + lxpad + rxpad
        tmax = (nz * dz + nx_global * dx) / vmax
        ntnewmax = int(tmax / dt) + 1
        ntnewrec = ntnewmax

        # Wavelet
        wf = self.Ricker_Wavelet(freq=25.0, dt=dt, nt=nt)
        wf_np = cp.asnumpy(wf) if isinstance(wf, cp.ndarray) else np.asarray(wf)
        if np.max(np.abs(wf_np)) > 0:
            wf_np = wf_np / np.max(np.abs(wf_np))
        wf_np = wf_np * 1.0
        d_w = cp.asarray(wf_np, dtype=cp.float32)

        if self.verbose:
            self.log(f"[RTM] nshot={nshot}, ntr={totaltr}, nt(in)={nt}, ntmod={ntnewmax}, dt={dt:.3e}, survey={survey_type}")

        nx_max = nxo + lxpad + rxpad
        n_snapshots = max(1, (ntnewmax + snap_stride - 1) // snap_stride)

        # Snapshot dtype (GPU)
        snap_dtype_cp = cp.float16 if (snap_dtype == "float16" and use_mixed_precision) else cp.float32
        snap_itemsize = 2 if snap_dtype_cp == cp.float16 else 4
        approx_snap_bytes = n_snapshots * nx_max * nz * snap_itemsize

        # Checkpoint
        do_checkpoint = False
        if max_gpu_snap_bytes is not None and approx_snap_bytes > max_gpu_snap_bytes:
            do_checkpoint = True
        if checkpoint_dir is not None:
            do_checkpoint = True

        # Prepare outputs
        if (vs is not None) and (rho is not None):
            Acoustic_mode = True
            image_PP = cp.zeros((nx_max, nz), dtype=cp.float32)
            image_PS = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum_PP = cp.zeros_like(image_PP)
        else:
            Acoustic_mode = False
            image_global = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum = cp.zeros_like(image_global)

        if denoise_params is None:
            denoise_params = {"sigma": 1.5, "local_sigma": 3.0, "gamma": 1.0}

        # Precompute FD weights
        _ = self.Ensure_FD_Weights(3, dx)

        # Pinned buffer for receiver copy
        pinned_host_buf = None
        pdata_host_out = None
        if do_forward and pdata is None and return_pdata:
            pdata_host_out = np.zeros((nshot, totaltr, ntnewmax), dtype=np.float32)

        def Record_Trace_Array(target, field2d, lxpad, ntr, rec_depth):
            if isinstance(rec_depth, (int, np.integer, cp.integer)):
                depth_idx = int(rec_depth)
                target[:, :] = target[:, :]
                return field2d[lxpad:lxpad + ntr, depth_idx]
            else:
                rd = cp.asarray(rec_depth, dtype=cp.int32)
                if rd.size == 1:
                    rd = cp.full((ntr,), int(rd.ravel()[0]), dtype=cp.int32)
                elif rd.size != ntr:
                    rd = cp.resize(rd, (ntr,))
                ix = cp.arange(ntr, dtype=cp.int32) + lxpad
                out = cp.zeros((ntr,), dtype=field2d.dtype)
                valid = (ix >= 0) & (ix < field2d.shape[0]) & (rd >= 0) & (rd < field2d.shape[1])
                if valid.any():
                    out[valid] = field2d[ix[valid], rd[valid]]
                return out

        #----------------
        # SHOT LOOP
        #----------------
        for ishot in tqdm(range(nshot), desc="[RTM] Shot", unit="shot"):
            if self.verbose:
                self.log(f"[RTM] Shot {ishot + 1}/{nshot}")

            d_p_fwd_snap_shot = None
            snap_memmap = None

            # Geometry and traces
            if pdata is not None:
                traces = pdata[ishot]
                ntr = traces.shape[0]
            else:
                ntr = int(min(600, nxo))

            sx = nxo // 2
            ix22 = max(0, sx - ntr // 2)
            nx = ntr + lxpad + rxpad
            src_local = sx - ix22 + lxpad

            src_ixs = np.clip(np.arange(ix22 - lxpad, ix22 - lxpad + nx), 0, nxo - 1)
            vp_block = vp[src_ixs, :].astype(np.float32)

            if Acoustic_mode:
                vs_block = (vs[src_ixs, :].astype(np.float32) if hasattr(vs, "shape") and vs.shape == vp.shape else (
                    vs if np.isscalar(vs) else vs[src_ixs, :].astype(np.float32)))
                rho_block = (
                    rho[src_ixs, :].astype(np.float32) if hasattr(rho, "shape") and rho.shape == vp.shape else (
                        rho if np.isscalar(rho) else rho[src_ixs, :].astype(np.float32)))
                mu_block = rho_block * (vs_block ** 2)
                lam_block = rho_block * (vp_block ** 2 - 2.0 * (vs_block ** 2))

            with cp.cuda.Device(device):
                d_vp = cp.asarray(vp_block, dtype=cp.float32)
                if Acoustic_mode:
                    d_vs = cp.asarray(vs_block, dtype=cp.float32)
                    d_rho = cp.asarray(rho_block, dtype=cp.float32)

                    p = cp.zeros((nx, nz), dtype=cp.float32)
                    vx = cp.zeros((nx, nz), dtype=cp.float32)
                    vz = cp.zeros_like(vx)

                    p_prev = cp.zeros_like(p)

                    p_rec = cp.zeros_like(p)
                    vx_rec = cp.zeros_like(vx)
                    vz_rec = cp.zeros_like(vx)

                else:
                    d_v = cp.asarray(vp_block, dtype=cp.float32)
                    d_pn = cp.zeros_like(d_v)
                    d_pp = cp.zeros_like(d_v)
                    d_pnrec = cp.zeros_like(d_v)
                    d_pprec = cp.zeros_like(d_v)

                pdata_shot_d = cp.zeros((ntr, ntnewmax), dtype=cp.float32)

                if do_forward:
                    if do_checkpoint:
                        if checkpoint_dir is None:
                            checkpoint_dir = "/tmp/rtm_chk"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        dtype_np = np.float16 if snap_dtype_cp == cp.float16 else np.float32
                        snap_p_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps_p.dat")
                        snap_memmap_p = np.memmap(snap_p_path, dtype=dtype_np, mode="w+",
                                                  shape=(n_snapshots, nx, nz))
                        snap_memmap = (snap_memmap_p,)
                    else:
                        d_p_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)

                bytes_per_shot = ntr * ntnewmax * np.dtype(np.float32).itemsize
                if pinned_host_buf is None:
                    try:
                        pinned_host_buf = cp.cuda.alloc_pinned_memory(min(bytes_per_shot, 2 ** 30))
                    except Exception:
                        pinned_host_buf = None
                if pinned_host_buf is not None:
                    host_flat = np.frombuffer(pinned_host_buf, dtype=np.float32, count=ntr * ntnewmax)
                    host_view = host_flat.reshape((ntr, ntnewmax))
                else:
                    host_view = None

                #------------------------
                # Forward Propagation
                #------------------------
                if do_forward:
                    for it in range(ntnewmax):
                        if Acoustic_mode:
                            p, vx, vz = self.Acoustic_TimeDomain_Propagation(
                                p=p, vx=vx, vz=vz,
                                rho=d_rho, vp=d_vp,
                                nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=True
                            )

                            if self.verbose and (it % 50 == 0):
                                energy = float(
                                    cp.sum(p * p + vx * vx + vz * vz).astype(cp.float64))
                                self.log(f"[Forward] it={it} energy={energy:.6e}")

                            if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                                val = float(d_w[it])
                                p[src_local, src_depth] += val

                            try:
                                p_tmp = p.copy()
                                p_mod = self.Absorbing_Boundary_Condition(
                                    nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                    dvv=d_vp, od=None, p=p, pm=p_prev,
                                    pp=p_tmp, abs_flags=(1, 1, 1, 1)
                                )
                                mask_boundary = (p_mod != p)
                                if mask_boundary.any():
                                    p[mask_boundary] = p_mod[mask_boundary]
                                p_prev[...] = p.copy()
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_vp, p, p, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            pdata_shot_d[:, it] = Record_Trace_Array(pdata_shot_d[:, it:it + 1], p, lxpad, ntr, rec_depth)

                            if it % 100 == 0 and self.verbose:
                                tr_max = float(cp.max(cp.abs(pdata_shot_d)).astype(cp.float32))
                                self.log(f"[Forward] recorded traces max abs so far = {tr_max:.5e}")

                        else:
                            d_pn, d_pp = d_pp, d_pn

                        if imaging and (it % snap_stride == 0):
                            sid = it // snap_stride

                            if do_checkpoint:
                                if self.stream is not None:
                                    self.stream.synchronize()
                                else:
                                    cp.cuda.runtime.deviceSynchronize()
                                if snap_dtype_cp == cp.float16:
                                    arr_p = cp.asnumpy(p.astype(cp.float16))
                                else:
                                    arr_p = cp.asnumpy(p.astype(cp.float32))
                                snap_memmap[0][sid, :, :] = arr_p
                            else:
                                if d_p_fwd_snap_shot is None:
                                    raise RuntimeError("Forward pressure snapshots missing: ensure do_forward=True, imaging=True and not do_checkpoint")
                                if snap_dtype_cp == cp.float16:
                                    d_p_fwd_snap_shot[sid, :, :] = p.astype(cp.float16)
                                else:
                                    d_p_fwd_snap_shot[sid, :, :] = p

                    if return_pdata and pdata_host_out is not None and host_view is not None:
                        try:
                            with (self.stream or cp.cuda.Stream.null):
                                pdata_shot_d.get(out=host_view, stream=(self.stream or None))
                            if self.stream is not None:
                                self.stream.synchronize()
                            else:
                                cp.cuda.runtime.deviceSynchronize()
                            pdata_host_out[ishot] = host_view.copy()
                        except Exception:
                            pdata_host_out[ishot] = cp.asnumpy(pdata_shot_d)

                else:
                    if self.verbose:
                        self.log("[RTM] forward skipped for this shot")

                #-------------------------------------
                # Backward Propagation and Imaging
                #-------------------------------------
                if do_backward:
                    if do_forward and pdata is None:
                        traces_gpu = pdata_shot_d
                    elif pdata is not None:
                        traces_gpu = cp.asarray(pdata[ishot], dtype=cp.float32)
                    else:
                        if pdata_host_out is not None:
                            traces_gpu = cp.asarray(pdata_host_out[ishot], dtype=cp.float32)
                        else:
                            raise RuntimeError("No traces available for backward propagation")

                    if traces_gpu is None:
                        raise RuntimeError("traces_gpu is None after selection")
                    if traces_gpu.ndim == 1:
                        traces_gpu = traces_gpu.reshape((ntr, -1))
                    elif traces_gpu.ndim == 2:
                        pass
                    else:
                        raise RuntimeError(f"Unexpected traces_gpu.ndim = {traces_gpu.ndim}")

                    if traces_gpu.shape[0] != ntr and traces_gpu.shape[1] == ntr:
                        traces_gpu = traces_gpu.T  # transpose untuk jadi (ntr, nt)

                    if self.verbose:
                        try:
                            self.log("[Backward] trace max/min:", float(traces_gpu.max()), float(traces_gpu.min()))
                        except Exception:
                            self.log("[Backward] trace stats unavailable (maybe shape issue)")

                    if Acoustic_mode:
                        p_rec.fill(0.0)
                        vx_rec.fill(0.0)
                        vz_rec.fill(0.0)

                        if isinstance(rec_depth, (int, np.integer)):
                            rec_depth_is_scalar = True
                            rec_depth_val = int(rec_depth)
                        else:
                            rec_depth_arr = cp.asarray(rec_depth, dtype=cp.int32) if not isinstance(rec_depth,
                                                                                                    cp.ndarray) else rec_depth.astype(
                                cp.int32)
                            if rec_depth_arr.size != ntr:
                                if rec_depth_arr.size == 1:
                                    rec_depth_arr = cp.full((ntr,), int(rec_depth_arr.ravel()[0]), dtype=cp.int32)
                                else:
                                    rec_depth_arr = cp.resize(rec_depth_arr, (ntr,))
                            rec_depth_is_scalar = False

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            ix_arr = cp.arange(ntr, dtype=cp.int32)
                            ix_global = ix_arr + lxpad

                            valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                            if valid.any():
                                idx = ix_global[valid]
                                amp = trace_t[valid]
                                if isinstance(rec_depth, (int, np.integer)):
                                    depth_idx = rec_depth
                                    p_rec[idx, depth_idx] += amp
                                else:
                                    depth_idx = rec_depth_arr[valid].astype(cp.int32)
                                    p_rec[idx, depth_idx] += amp

                            p_rec, vx_rec, vz_rec = self.Acoustic_TimeDomain_Propagation(
                                p=p_rec, vx=vx_rec, vz=vz_rec,
                                rho=d_rho, vp=d_vp,
                                nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=True
                            )

                            # Apply ABC to backward pressure
                            try:
                                p_before = p_rec.copy()
                                p_mod = self.Absorbing_Boundary_Condition(
                                    nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                    dvv=d_vp, od=None, p=p_rec, pm=p_prev, pp=p_rec.copy(),
                                    abs_flags=(1, 1, 1, 1)
                                )
                                if p_mod is None:
                                    p_mod = p_before
                                mask_b = (p_mod != p_before)
                                if mask_b.any():
                                    p_rec[mask_b] = p_mod[mask_b]
                                p_prev[...] = p_before.copy()
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_vp, p_rec, p_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            if "absorbgpuup" in self.k:
                                try:
                                    self.k["absorbgpuup"](d_vp, vx_rec, vx_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, vz_rec, vz_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, p_rec, p_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            #----------------------------------------
                            # Imaging Condition (Pressure-Pressure)
                            #----------------------------------------
                            if imaging and (it % snap_stride == 0):
                                sid = it // snap_stride

                                max_sid = (d_p_fwd_snap_shot.shape[0] - 1) if (d_p_fwd_snap_shot is not None) else (
                                        snap_memmap[0].shape[0] - 1)
                                if sid < 0 or sid > max_sid:
                                    raise RuntimeError(f"Snapshot id out of range: sid={sid}, max={max_sid}")

                                if do_checkpoint:
                                    p_host = snap_memmap[0][sid, :, :].astype(np.float32)
                                    p_fwd_local = cp.asarray(p_host)
                                else:
                                    if d_p_fwd_snap_shot is None:
                                        raise RuntimeError("Forward pressure snapshots missing: ensure do_forward=True, imaging=True and not do_checkpoint")
                                    p_fwd_local = d_p_fwd_snap_shot[sid, :, :].astype(cp.float32)

                                p_bwd = p_rec

                                d_result_PP = p_fwd_local * p_bwd

                                image_PP[ix22: ix22 + nx, :] += d_result_PP[ix22: ix22 + nx, :]

                                illum_PP[ix22: ix22 + nx, :] += p_fwd_local[ix22: ix22 + nx, :] * p_fwd_local[ix22: ix22 + nx, :]

                    else:
                        d_pnrec.fill(0.0)
                        d_pprec.fill(0.0)
                        d_result = cp.zeros((nx, nz), dtype=cp.float32)

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            try:
                                if "receivedata" in self.k:
                                    self.k["receivedata"](d_pnrec, trace_t, nz, ntr, lxpad, 0)
                                else:
                                    ix_arr = cp.arange(ntr, dtype=cp.int32)
                                    ix_global = ix_arr + lxpad
                                    valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                                    if valid.any():
                                        idx = ix_global[valid]
                                        depth_idx = rec_depth[valid].astype(cp.int32)
                                        amp = trace_t[valid]
                                        d_pnrec[idx, depth_idx] += amp
                            except Exception:
                                ix_arr = cp.arange(ntr, dtype=cp.int32)
                                ix_global = ix_arr + lxpad
                                valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                                if valid.any():
                                    idx = ix_global[valid]
                                    depth_idx = rec_depth[valid].astype(cp.int32)
                                    amp = trace_t[valid]
                                    d_pnrec[idx, depth_idx] += amp

                            # legacy/helmh
                            freq0 = FREQUENCY0
                            try:
                                ux_c = cp.zeros((nx, nz), dtype=cp.complex64)
                                uz_c = d_pnrec.astype(cp.complex64)

                                if 'd_rho' in locals():
                                    rho_local = d_rho
                                else:
                                    rho_local = cp.ones((nx, nz), dtype=cp.float32)

                                mu_local = cp.full((nx, nz), 1e-6, dtype=cp.float32)
                                lam_local = rho_local * (d_v ** 2) - 2.0 * mu_local

                                ux_out, uz_out, info_h = self.Helmholtz_Acoustic_Propagation(
                                    ux_c, uz_c, rho_local, lam_local, mu_local,
                                    nx, dx, nz, dz, freq=freq0,
                                    fd_order_radius=3,
                                    use_gpu=True,
                                    pad_mode='reflect',
                                    absorb_R=0,
                                    src=None,
                                    inplace=True,
                                    use_external_abc=True,
                                    solver_tol=1e-6,
                                    solver_maxiter=MAXITER,
                                    verbose=self.verbose
                                )

                                w_x_np = self.Ensure_FD_Weights(3, dx)
                                w_z_np = self.Ensure_FD_Weights(3, dz)
                                w_x = cp.asarray(w_x_np, dtype=cp.float32)
                                w_z = cp.asarray(w_z_np, dtype=cp.float32)

                                dux_dx = self.Apply_1D_Shifted_Convolution(cp, ux_out.real.astype(cp.float32), w_x, axis=0, pad_mode='reflect')
                                duz_dz = self.Apply_1D_Shifted_Convolution(cp, uz_out.real.astype(cp.float32), w_z, axis=1, pad_mode='reflect')

                                K_local = rho_local * (d_v ** 2)
                                p_complex = - (K_local.astype(cp.complex64)) * (dux_dx.astype(cp.complex64) + duz_dz.astype(cp.complex64))

                                d_pprec[...] = cp.real(p_complex).astype(cp.float32)

                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_v, d_pprec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                                d_pnrec, d_pprec = d_pprec, d_pnrec

                            except Exception as ex_helm_backward:
                                if self.verbose:
                                    self.log("[[Backward][Helmholtz] fallback to time-domain due to:", ex_helm_backward)
                                self.k["TimeDomain_Propagation"](d_v, d_pnrec, d_pprec, nx=nx, dx=dx, nz=nz, dz=dz,
                                                                 dt=dt)
                                if "absorbgpuup" in self.k:
                                    try:
                                        self.k["absorbgpuup"](d_v, d_pnrec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                                    except Exception:
                                        pass
                                d_pnrec, d_pprec = d_pprec, d_pnrec

                        image_global[ix22:ix22 + nx, :] += d_result

                    try:
                        del d_vp
                        if Acoustic_mode:
                            del d_vs, d_rho
                        del pdata_shot_d
                        if (d_p_fwd_snap_shot is not None) and (not do_checkpoint):
                            del d_p_fwd_snap_shot
                    except Exception:
                        pass

                    cp.cuda.Stream.null.synchronize()

        # End shot loop

        # Normalize and final denoise
        if Acoustic_mode:
            image_PP = self.Preconditioning(image_PP, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)
            image_PS = self.Preconditioning(image_PS, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)

            out_cp = (image_PP + 0.5 * image_PS) if Acoustic_mode else image_global
            if isinstance(out_cp, cp.ndarray):
                out_cp = cp.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
                maxv = float(cp.max(cp.abs(out_cp)))
                out = cp.asnumpy(out_cp / (maxv if maxv > 1e-12 else 1.0))
            else:
                out_np = np.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
                maxv = float(np.max(np.abs(out_np)))
                out = out_np / (maxv if maxv > 1e-12 else 1.0)

            if np.max(np.abs(out)) < 1e-10:
                if getattr(self, "verbose", False):
                    self.log("[RTM DEBUG] final image is nearly zero after imaging. Check forward snapshots, traces and illumination.")

        else:
            image_global = image_global / (illum + EPS)

            image_global = self.Preconditioning(image_global, nx=nx_max, minnz=src_depth + 2, maxnz=nz, nz=nz, use_gpu=True)

            d_out = cp.zeros_like(image_global)

            out_cp = image_global
            out_cp = cp.nan_to_num(out_cp, nan=0.0, posinf=0.0, neginf=0.0)
            maxv = float(cp.max(cp.abs(out_cp)))
            out = cp.asnumpy(out_cp / (maxv if maxv > 1e-12 else 1.0))

        if return_pdata and pdata_host_out is not None:
            return out, pdata_host_out
        return out












