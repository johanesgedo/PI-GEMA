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
import math
import numpy as np

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy.ndimage import convolve1d as cp_convolve1d
    HAS_CUPY = True
except Exception:
    cp = np
    cp_convolve1d = None
    HAS_CUPY = False

from scipy.ndimage import convolve1d as sp_convolve1d
from scipy.sparse.linalg import LinearOperator, gmres
from tqdm import tqdm
from ERTM2D_2_Profiler_2 import Init_Profiler_in_ERTM2D


DX = 2
DZ = 2
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


class Elastic_Reverse_Time_Migration_2D:

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

        def _log(*args, **kwargs):
            if self.verbose:
                print("[Elastic_RTM]", *args, **kwargs)

        self.log = _log

        if register_kernels:
            def _default_receivedata(dst, trace, nz, ntr, lxpad, pad_pos=0):
                try:
                    dst[lxpad:lxpad + ntr, pad_pos] += trace
                except Exception:
                    for i in range(min(ntr, dst.shape[0] - lxpad)):
                        dst[lxpad + i, pad_pos] += trace[i]
                return

            def _time_domain_propagation_fallback(*args, **kwargs):
                return self.Elastic_TimeDomain_Propagation(*args, **kwargs)

            self.k = {
                "Elastic_TimeDomain_Propagation": self.Elastic_TimeDomain_Propagation,
                "absorbgpuup": self.Absorbgpuup,
                "Apply_1D_Shifted_Convolution": self.Apply_1D_Shifted_Convolution,
                "Cross_Correlation": self.Cross_Correlation,
                "Elastic_Denoise": self.Elastic_Denoise,
                "Image_Gaussian": self.Image_Gaussian,
                "FD_Weights_Fornberg": type(self).FD_Weights_Fornberg,
                "Get_FD_Weights": type(self).Get_FD_Weights,
                "Ricker_Wavelet": self.Ricker_Wavelet,
                "Make_FD_First_Derivative_Weights": type(self).Make_FD_First_Derivative_Weights,
                "TimeDomain_Propagation": _time_domain_propagation_fallback,
                "receivedata": _default_receivedata,
            }
            
        # Add Profiler
        try:
            Init_Profiler_in_ERTM2D(self, enable=True)
        except Exception as e:
            if self.verbose:
                self.log("[Elastic_RTM_2D_Profiler] failed to initialize:", e)

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
            return cp.asarray([], dtype=cp.float32) if (use_gpu or (use_gpu is None and self.has_cupy)) else np.asarray([], dtype=np.float32)

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

        arr = v if (xp is np and isinstance(v, np.ndarray)) or (xp is cp and isinstance(v, cp.ndarray)) else xp.asarray(v)

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
        HALF: int = 4,
        EPS: float = 1e-6,
        SCALE: float = 1.0,
        WMIN: float = 0.2,
        WMAX: float = 5.0,
        use_gpu: bool | None = None
    ):
        # Backend
        is_cp = HAS_CUPY and isinstance(d_pp, cp.ndarray)
        use_gpu = (self.has_cupy and is_cp) if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        arr = d_pp if (xp is np and isinstance(d_pp, np.ndarray)) or (xp is cp and isinstance(d_pp, cp.ndarray)) else xp.asarray(d_pp)
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


    """
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
                nbc = max(4, min(40, min(nx, nz) // 10))
            else:
                nbc = int(max(1, min(max(1, min(nx, nz) // 2 - 1), iz)))

        v = xp.asarray(d_v, dtype=xp.float32)
        if d_pp is not None:
            pp = xp.asarray(d_pp, dtype=xp.float32)
        else:
            pp = None
        if d_pn is not None:
            pn = xp.asarray(d_pn, dtype=xp.float32)
        else:
            pn = None

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
            edge_strength = xp.maximum(tx2, tz2)

            v_local = xp.maximum(v, 1e-6, dtype=xp.float32)
            vmax = float(xp.max(v_local).astype(xp.float32))
            if vmax <= 0.0:
                vmax = 1.0

            v_factor = 0.5 + 0.5 * (v_local / (vmax + eps))

            tscale = min(1.0, max(0.05, float(dt) * 50.0))
            base_alpha = 0.20
            power = 2.0

            alpha = base_alpha * tscale
            exponent = -alpha * (edge_strength ** power) * v_factor

            damp = xp.exp(exponent, dtype=xp.float32)
            damp = xp.clip(damp, 0.0, 1.0)

        if xp is cp:
            mask = damp < 0.9999999
            if "pp" in apply_to and pp is not None:
                if inplace:
                    if mask.any():
                        pp[mask] = pp[mask] * damp[mask].astype(pp.dtype)
                else:
                    pp = pp * damp
            if "pn" in apply_to and pn is not None:
                if inplace:
                    if mask.any():
                        pn[mask] = pn[mask] * damp[mask].astype(pn.dtype)
                else:
                    pn = pn * damp
        else:
            mask = damp < 0.9999999
            if "pp" in apply_to and pp is not None:
                if inplace:
                    pp[mask] = pp[mask] * damp[mask].astype(pp.dtype)
                else:
                    pp = pp * damp
            if "pn" in apply_to and pn is not None:
                if inplace:
                    pn[mask] = pn[mask] * damp[mask].astype(pn.dtype)
                else:
                    pn = pn * damp

        try:
            if isinstance(d_pp, cp.ndarray):
                if pp is not d_pp:
                    d_pp[:] = pp
            if isinstance(d_pn, cp.ndarray):
                if pn is not d_pn:
                    d_pn[:] = pn
        except Exception:
            pass

        return d_pp
    """


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
            out[1:-1] = (p[2:,iz] - p[:-2,iz]) * (0.5 * inv_dx)
            out[0] = (p[1,iz] - p[0,iz]) * inv_dx
            out[-1] = (p[-1,iz] - p[-2,iz]) * inv_dx
            return out

        def compute_dp_dz_at_x(ix):
            if nz == 1:
                return xp.zeros((1,), dtype=xp.float32)
            out = xp.empty((nz,), dtype=xp.float32)
            if nz == 2:
                val = (p[ix,1] - p[ix,0]) * inv_dz
                out[0] = val
                out[1] = val
                return out
            out[:,] = 0.0
            out[1:-1] = (p[ix,2:] - p[ix,:-2]) * (0.5 * inv_dz)
            out[0] = (p[ix,1] - p[ix,0]) * inv_dz
            out[-1] = (p[ix,-1] - p[ix,-2]) * inv_dz
            return out

        #---------------------------
        # TOP BOUNDARY (PP[:,0])
        #---------------------------
        if top_flag:
            iz = 1 if nz > 1 else 0
            dv = dvv[:,iz]
            if od_arr is not None:
                with_od = od_arr[:,iz]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs,0.0))

            dpdx = compute_dp_dx_at_z(iz)
            dpdt = (pp[:,iz] - pm[:,iz]) * inv_2dt

            dpdxs = dpdx * dpdx
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdxs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dz) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[:,0] = gamma * (pp[:,iz] - p[:,0]) + p[:,iz]

        else:
            pp[:,0] = 0.0

        #------------------------------
        # LEFT BOUNDARY (pp[0,:])
        #------------------------------
        if left_flag:
            ix = 1 if nx > 1 else 0
            dv = dvv[ix,:]
            if od_arr is not None:
                with_od = od_arr[ix,:]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdz = compute_dp_dz_at_x(ix)
            dpdt = (pp[ix,:] - pm[ix,:]) * inv_2dt

            dpdzs = dpdz * dpdz
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdzs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dx) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[0,:] = gamma * (pp[ix,:] - p[0,:]) + p[ix,:]

        else:
            pp[0,:] = 0.0

        #----------------------------------
        # BOTTOM BOUNDARY (pp[:, nz-1])
        #----------------------------------
        if bottom_flag:
            iz = nz - 2 if nz > 1 else 0
            dv = dvv[:,iz]
            if od_arr is not None:
                with_od = od_arr[:,iz]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdx = compute_dp_dx_at_z(iz)
            dpdt = (pp[:,iz] - pm[:,iz]) * inv_2dt

            dpdxs = dpdx * dpdx
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdxs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dz) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[:,nz-1] = gamma * (pp[:,iz] - p[:,nz-1]) + p[:,iz]

        else:
            pp[:, nz-1] = 0.0

        #------------------------------
        # RIGHT BOUNDARY
        #------------------------------
        if right_flag:
            ix = nx - 2 if nx > 1 else 0
            dv = dvv[ix, :]
            if od_arr is not None:
                with_od = od_arr[ix,:]
                ovs = 1.0 / (with_od * dv + eps)
            else:
                ovs = 1.0 / (dv + eps)
            ov = xp.sqrt(xp.maximum(ovs, 0.0))

            dpdz = compute_dp_dz_at_x(ix)
            dpdt = (pp[ix,:] - pm[ix,:]) * inv_2dt

            dpdzs = dpdz * dpdz
            dpdts = dpdt * dpdt

            denom = ovs * dpdts + eps
            ratio = safe_ratio(dpdzs, denom)
            ratio = xp.clip(ratio, 0.0, 1.0)
            cosa = xp.sqrt(xp.maximum(0.0, 1.0 - ratio))

            beta = ov * float(dx) * cosa / (float(dt) + eps)
            gamma = (1.0 - beta) / (1.0 + beta)
            gamma = xp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

            pp[nx-1,:] = gamma * (pp[ix,:] - p[nx-1,:]) + p[ix,:]

        else:
            pp[nx-1,:] = 0.0

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
    # Elastic Wave Propagation
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
    def Make_FD_First_Derivative_Weights(cls, R, dx):
        nodes = np.arange(-R, R + 1, dtype=np.float64) * dx
        w = cls.FD_Weights_Fornberg(nodes, 0.0, 1)
        return w.astype(np.float32)

    @staticmethod
    def Get_FD_Weights(cls, R, dx):
        key = (int(R), float(dx))
        w = FD_WEIGHTS_CACHE.get(key)
        if w is None:
            w = cls.Make_FD_First_Derivative_Weights(R, dx)
            FD_WEIGHTS_CACHE[key] = w
        return w

    def Ensure_FD_Weights(self, R, dx):
        R = int(max(1,R))
        if R == 1:
            w = np.array([-0.5, 0.0, 0.5], dtype=np.float32) / float(dx)
        elif R == 2:
            w = np.array([1/12, -2/3, 0.0, 2/3, -1/12], dtype=np.float32) / float(dx)
        elif R == 3:
            w = np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60], dtype=np.float32) / float(dx)
        else:
            n = 2*R + 1
            w = np.zeros(n, dtype=np.float32)
            mid = n // 2
            w[mid-1] = -0.5 / float(dx)
            w[mid+1] = 0.5 / float(dx)
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


    def Elastic_TimeDomain_Propagation(
        self,
        vx, vz,
        sxx, szz, sxz,
        rho, lam, mu,
        nx: int,
        dx: float,
        nz: int,
        dz: float,
        dt: float,
        fd_order_radius: int = 3,
        use_gpu: bool | None = True,
        # pad_mode: str = 'edge',
        # pad_mode: str = 'nearest',
        pad_mode: str = 'reflect',
        absorb_R: int = 0,
        src: dict | None = None,
        inplace: bool = True,
        use_external_abc: bool = False
    ):
        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        if not (hasattr(vx, "shape") and len(vx.shape) == 2):
            raise ValueError("vx must be 2D (nx,nz)")
        if vx.shape != (nx, nz):
            raise ValueError(f"vx shape {vx.shape} != ({nx},{nz})")

        if inplace:
            vx_arr = xp.asarray(vx, dtype=xp.float32)
            vz_arr = xp.asarray(vz, dtype=xp.float32)
            sxx_arr = xp.asarray(sxx, dtype=xp.float32)
            szz_arr = xp.asarray(szz, dtype=xp.float32)
            sxz_arr = xp.asarray(sxz, dtype=xp.float32)
        else:
            vx_arr = xp.asarray(vx, dtype=xp.float32).copy()
            vz_arr = xp.asarray(vz, dtype=xp.float32).copy()
            sxx_arr = xp.asarray(sxx, dtype=xp.float32).copy()
            szz_arr = xp.asarray(szz, dtype=xp.float32).copy()
            sxz_arr = xp.asarray(sxz, dtype=xp.float32).copy()

        rho_arr = xp.asarray(rho, dtype=xp.float32) if not np.isscalar(rho) else xp.full((nx, nz), float(rho), dtype=xp.float32)
        lam_arr = xp.asarray(lam, dtype=xp.float32) if not np.isscalar(lam) else xp.full((nx, nz), float(lam), dtype=xp.float32)
        mu_arr  = xp.asarray(mu, dtype=xp.float32)  if not np.isscalar(mu)  else xp.full((nx, nz), float(mu), dtype=xp.float32)

        if rho_arr.shape != (nx, nz):
            try:
                rho_arr = xp.broadcast_to(rho_arr.reshape(-1, 1) if rho_arr.ndim==1 and rho_arr.size==nx else rho_arr.reshape(1, -1), (nx, nz)).astype(xp.float32)
            except Exception:
                raise ValueError("rho must be scalar or shape (nx,nz) or (nx,) or (nz,)")

        if lam_arr.shape != (nx, nz):
            lam_arr = xp.broadcast_to(lam_arr, (nx, nz)).astype(xp.float32)
        if mu_arr.shape != (nx, nz):
            mu_arr = xp.broadcast_to(mu_arr, (nx, nz)).astype(xp.float32)

        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        if self.verbose:
            self.log("FD weights sums:", float(w_x.sum()), float(w_z.sum()))

        dsxx_dx = self.Apply_1D_Shifted_Convolution(xp, sxx_arr, w_x, axis=0, pad_mode=pad_mode)
        dsxz_dz = self.Apply_1D_Shifted_Convolution(xp, sxz_arr, w_z, axis=1, pad_mode=pad_mode)

        inv_rho = 1.0 / xp.maximum(rho_arr, xp.array(1e-12, dtype=xp.float32))
        factor_v = (dt * inv_rho).astype(xp.float32)

        vx_arr += factor_v * (dsxx_dx + dsxz_dz)
        dsxz_dx = self.Apply_1D_Shifted_Convolution(xp, sxz_arr, w_x, axis=0, pad_mode=pad_mode)
        dszz_dz = self.Apply_1D_Shifted_Convolution(xp, szz_arr, w_z, axis=1, pad_mode=pad_mode)
        vz_arr += factor_v * (dsxz_dx + dszz_dz)

        if xp is cp:
            if bool(cp.any(~cp.isfinite(vx_arr))) or bool(cp.any(~cp.isfinite(vz_arr))):
                vx_arr = cp.nan_to_num(vx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                vz_arr = cp.nan_to_num(vz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Elastic] NaN/Inf detected in velocity - clamped")
        else:
            if (not np.isfinite(vx_arr).all()) or (not np.isfinite(vz_arr).all()):
                vx_arr = np.nan_to_num(vx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                vz_arr = np.nan_to_num(vz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Elastic] NaN/Inf detected in velocity - clamped")

        if xp is cp and self.verbose:
            if cp.any(~cp.isfinite(vx_arr)):
                idx = cp.where(~cp.isfinite(vx_arr))
                self.log("NaN in vx at indices", idx[0][:5].get(), idx[1][:5].get())

        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            amp = float(src.get("amp", 0.0))
            if 0 <= sx < nx and 0 <= sz < nz and amp != 0.0:
                vx_arr[sx, sz] += 0.5 * amp * dt
                vz_arr[sx, sz] += 0.5 * amp * dt

        dvx_dx = self.Apply_1D_Shifted_Convolution(xp, vx_arr, w_x, axis=0, pad_mode=pad_mode)
        dvz_dz = self.Apply_1D_Shifted_Convolution(xp, vz_arr, w_z, axis=1, pad_mode=pad_mode)
        dvx_dz = self.Apply_1D_Shifted_Convolution(xp, vx_arr, w_z, axis=1, pad_mode=pad_mode)
        dvz_dx = self.Apply_1D_Shifted_Convolution(xp, vz_arr, w_x, axis=0, pad_mode=pad_mode)

        sxx_arr += dt * ((lam_arr + 2.0 * mu_arr) * dvx_dx + lam_arr * dvz_dz)
        szz_arr += dt * ((lam_arr + 2.0 * mu_arr) * dvz_dz + lam_arr * dvx_dx)
        sxz_arr += dt * (mu_arr * (dvx_dz + dvz_dx))


        if xp is cp:
            if bool(cp.any(~cp.isfinite(sxx_arr))) or bool(cp.any(~cp.isfinite(szz_arr))) or bool(cp.any(~cp.isfinite(sxz_arr))):
                sxx_arr = cp.nan_to_num(sxx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                szz_arr = cp.nan_to_num(szz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                sxz_arr = cp.nan_to_num(sxz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Elastic] NaN/Inf detected in stress - clamped")
        else:
            if (not np.isfinite(sxx_arr).all()) or (not np.isfinite(szz_arr).all()) or (not np.isfinite(sxz_arr).all()):
                sxx_arr = np.nan_to_num(sxx_arr, nan=0.0, posinf=0.0, neginf=0.0)
                szz_arr = np.nan_to_num(szz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                sxz_arr = np.nan_to_num(sxz_arr, nan=0.0, posinf=0.0, neginf=0.0)
                if self.verbose:
                    self.log("[Elastic] NaN/Inf detected in stress - clamped")


        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            amp = float(src.get("amp", 0.0))
            stype = src.get("type", "explosive")
            if 0 <= sx < nx and 0 <= sz < nz and amp != 0.0:
                if stype == "explosive" or stype == "pressure":
                    sxx_arr[sx, sz] += amp
                    szz_arr[sx, sz] += amp
                elif stype == "shear":
                    sxz_arr[sx, sz] += amp

        if absorb_R is not None and int(absorb_R) > 0:
            R_abs = int(absorb_R)
            R_abs = min(R_abs, max(1, min(nx, nz)//2 - 1))
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
            sxx_arr *= tap2d
            szz_arr *= tap2d
            sxz_arr *= tap2d

            if use_external_abc and "absorbgpuup" in self.k:
                try:
                    vel_map = xp.sqrt(xp.maximum((lam_arr + 2.0*mu_arr) / xp.maximum(rho_arr, 1e-12), 1e-6))
                    abc_fn = self.k["absorbgpuup"]
                    abc_fn(vel_map, vx_arr, vx_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, vz_arr, vz_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, sxx_arr, sxx_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, szz_arr, szz_arr, nx, nz, dx, dz, dt, iz=R_abs)
                    abc_fn(vel_map, sxz_arr, sxz_arr, nx, nz, dx, dz, dt, iz=R_abs)
                except Exception:
                    pass

        return vx_arr, vz_arr, sxx_arr, szz_arr, sxz_arr


    """
    def Helmholtz_Elastic_Propagation(
        self,
        ux, uz,
        rho, lam, mu,
        nx: int,
        dx: float,
        nz: int,
        dz: float,
        freq: float,
        fd_order_radius: int = 3,
        use_gpu: bool | None = True,
        pad_mode: str = 'reflect',
        absorb_R: int = 0,
        src: dict | None = None,
        inplace: bool = True,
        use_external_abc: bool = False,
        solver_tol: float = 1e-6,
        solver_maxiter: int = MAXITER,
        verbose: bool | None = None
    ):
        if verbose is None:
            verbose = getattr(self, "verbose", False)

        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and cp is not None) else np

        # Validation
        if ux is not None and not (hasattr(ux, "shape") and len(ux.shape) == 2):
            raise ValueError("ux must be 2D (nx,nz) or None for allocation")
        if uz is not None and not (hasattr(uz, "shape") and len(uz.shape) == 2):
            raise ValueError("uz must be 2D (nx,nz) or None for allocation")
        if ux is not None and ux.shape != (nx,nz):
            raise ValueError(f"ux shape {ux.shape} != ({nx},{nz}")
        if uz is not None and uz.shape != (nx, nz):
            raise ValueError(f"uz shape {uz.shape} != ({nx},{nz})")

        # Allocate
        if inplace:
            ux_arr = xp.asarray(ux, dtype=xp.complex64) if ux is not None else xp.zeros((nx, nz), dtype=xp.complex64)
            uz_arr = xp.asarray(uz, dtype=xp.complex64) if uz is not None else xp.zeros((nx, nz), dtype=xp.complex64)
        else:
            ux_arr = xp.asarray(ux, dtype=xp.complex64).copy() if ux is not None else xp.zeros((nx, nz), dtype=xp.complex64)
            uz_arr = xp.asarray(uz, dtype=xp.complex64).copy() if uz is not None else xp.zeros((nx, nz), dtype=xp.complex64)

        # Material fields
        def to_field(a):
            if np.isscalar(a):
                return xp.full((nx, nz), float(a), dtype=xp.float32)
            arr = xp.asarray(a, dtype=xp.float32)
            if arr.shape == (nx,):
                return xp.broadcast_to(arr.reshape(nx,1), (nx,nz)).astype(xp.float32)
            if arr.shape == (nz,):
                return xp.broadcast_to(arr.reshape(1,nz), (nx,nz)).astype(xp.float32)
            if arr.shape != (nx,nz):
                raise ValueError("material must be scalar or shape (nx,nz) or (nx,) or (nz,)")

        rho_arr = to_field(rho)
        lam_arr = to_field(lam)
        mu_arr = to_field(mu)

        # Finite Difference weights
        R = max(1, int(fd_order_radius))
        w_x_np = self.Ensure_FD_Weights(R, dx)
        w_z_np = self.Ensure_FD_Weights(R, dz)
        w_x = xp.asarray(w_x_np, dtype=xp.float32)
        w_z = xp.asarray(w_z_np, dtype=xp.float32)

        if verbose:
            try:
                self.log("Helmholtz: FD weights sums:", float(w_x.sum()), float(w_z.sum()))
            except Exception:
                print("[Helmholtz] FD weights sums:", float(w_x.sum()), float(w_z.sum()))

        def D_x(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_x, axis=0, pad_mode=pad_mode)

        def D_z(arr):
            return self.Apply_1D_Shifted_Convolution(xp, arr, w_z, axis=1, pad_mode=pad_mode)

        def make_sponge(R_abs, strength=50.0):
            if R_abs is None or int(R_abs) <= 0:
                return xp.zeros((nx, nz), dtype=xp.float32)
            R_abs = min(int(R_abs), max(1, min(nx,nz)//2 - 1))
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
            tap2d = (tx2[:,None] * tz2[None,:]).astype(np.float32)
            eta_np = strength * (1.0 - tap2d)
            return xp.asarray(eta_np, dtype=xp.float32)

        damping_eta = make_sponge(absorb_R) if int(absorb_R) > 0 else xp.zeros((nx,nz), dtype=xp.float32)

        # Frequency to angular frequency
        omega = 2.0 * np.pi * float(freq)

        def apply_operator_u(u2):
            ux_local = xp.asarray(u2[0], dtype=xp.complex64)
            uz_local = xp.asarray(u2[1], dtype=xp.complex64)

            # Derivatives
            ux_x = D_x(ux_local)
            ux_z = D_z(ux_local)
            uz_x = D_x(uz_local)
            uz_z = D_z(uz_local)

            lam = lam_arr
            mu = mu_arr
            rho = rho_arr

            # Stresses
            sxx = (lam + 2.0 * mu) * ux_x + lam * uz_z
            szz = (lam + 2.0 * mu) * uz_z + lam * ux_x
            sxz = mu * (ux_z + uz_x)

            # Divergence of stress
            div_x = D_x(sxx) + D_z(sxz)
            div_z = D_x(sxz) + D_z(szz)

            # Helmholtz opertor
            Aux = - (omega ** 2) * rho * ux_local - div_x
            Auz = - (omega ** 2) * rho * uz_local - div_z

            # Apply sponge damping as approximate PML
            if absorb_R is not None and int(absorb_R) > 0:
                Aux = Aux + (-1j * omega) * (damping_eta.astype(Aux.dtype)) * ux_local
                Auz = Auz + (-1j * omega) * (damping_eta.astype(Auz.dtype)) * uz_local

            # Optional external ABC hook
            if use_external_abc and hasattr(self, "k") and "absorbgpuup" in getattr(self, "k", {}):
                try:
                    abc_fn = self.k["absorbgpuup"]
                    abc_fn(omega, Aux, Auz, nx, nz, dx, dz)
                except Exception:
                    pass

            return xp.stack([Aux, Auz], axis=0)

        # Build RHS from src (force density)
        f_rhs = xp.zeros((2, nx, nz), dtype=xp.complex64)
        if src is not None:
            sx = int(src.get("ix", -1))
            sz = int(src.get("iz", -1))
            fx = complex(src.get("fx", 0.0))
            fz = complex(src.get("fz", 0.0))
            stype = src.get("type", "force")
            if 0 <= sx < nx and 0 <= sz < nz and (fx != 0.0 or fz != 0.0):
                f_rhs[0, sx, sz] = fx
                f_rhs[1, sx, sz] = fz
            elif stype in ("explosive", "pressure") and 0 <= sx < nx and 0 <= sz < nz:
                amp = complex(src.get("amp", 0.0))
                f_rhs[0, sx, sz] = amp
                f_rhs[1, sx, sz] = amp

        # GMRES (Generelized Minimal Residual) solver setup
        n = 2 * nx * nz
        def mv_for_scipy(v_numpy):
            if xp is np:
                v_xp = xp.asarray(v_numpy, dtype=xp.complex64)
                u2 = v_xp.reshape((2, nx, nz))
                Av = apply_operator_u(u2)
                return np.asarray(Av.ravel(), dtype=np.complex64)
            else:
                try:
                    v_cp = xp.asarray(v_numpy)
                except Exception:
                    v_cp = xp.asarray(np.asarray(v_numpy))
                u2 = v_cp.reshape((2, nx, nz))
                Av_cp = apply_operator_u(u2)
                Av_np = np.asarray(xp.asnumpy(Av_cp))
                return Av_np.ravel().astype(np.complex64)

        Aop = LinearOperator((n,n), matvec=mv_for_scipy, dtype=np.complex64)

        # Flatten rhs to numpy for GMRES
        if xp is np:
            b = np.asarray(f_rhs.ravel(), dtype=np.complex64)
        else:
            b = np.asarray(xp.asnumpy(f_rhs.ravel()), dtype=np.complex64)

        if verbose:
            try:
                self.log(f"[Helmholtz] Solving freq={freq} Hz, n={n}, tol={solver_tol}, maxiter={solver_maxiter}")
            except Exception:
                print(f"[Helmholtz] Solving freq={freq} Hz, n={n}, tol={solver_tol}, maxiter={solver_maxiter}")

        # Initial guess
        x0 = None
        try:
            if xp is np:
                maybe = np.asarray(ux_arr.ravel(), dtype=np.complex64)
                if np.any(maybe != 0.0):
                    x0 = np.concatenate([np.asarray(ux_arr.ravel()), np.asarray(uz_arr.ravel())]).astype(np.complex64)
            else:
                ux_np = np.asarray(xp.asnumpy(xp.asarray(ux_arr).ravel()))
                if np.any(ux_np != 0.0):
                    x0 = np.concatenate([np.asarray(ux_np), np.asarray(np.asarray(xp.asnumpy(uz_arr).ravel()))]).astype(np.complex64)
        except Exception:
            x0 = None

        # Run GMRES
        x_sol, info = gmres(Aop, b, x0=x0, tol=solver_tol, restart=50, maxiter=solver_maxiter)

        if info != 0 and verbose:
            try:
                self.log("[Helmholtz] GMRES did not converge, info=", info)
            except Exception:
                print("[Helmholtz] GMRES did not converge, info=", info)

        # Reshape solution to (2, nx, nz)
        u_np = x_sol.reshape((2, nx, nz)).astype(np.complex64)

        # Return in xp backend requested
        if xp is np:
            ux_out = u_np[0]
            uz_out = u_np[1]
        else:
            ux_out = xp.asarray(u_np[0])
            uz_out = xp.asarray(u_np[1])

        return ux_out, uz_out, info
    """


    def Elastic_Denoise(
        self,
        vx, vz, sxx, szz, sxz,
        nx: int = None,
        nz: int = None,
        dx: float = DX,
        dz: float = DZ,
        use_gpu: bool | None = None,
        sigma: float = SIGMA,
        local_sigma: float = LOCAL_SIGMA,
        eps: float = EPS,
        gamma: float = GAMMA,
        transpose_input: bool = False,
        mode: str = "reflect",
        inplace: bool = False,
        strength: float = STRENGTH,
        energy_clip: tuple = (0.5, 2.0)
    ):
        # Backend
        use_gpu = self.has_cupy if use_gpu is None else bool(use_gpu)
        xp = cp if (use_gpu and HAS_CUPY) else np

        transposed = False
        if transpose_input:
            vx = vx.T
            vz = vz.T
            sxx = sxx.T
            szz = szz.T
            sxz = sxz.T
            transposed = True

        vx = xp.asarray(vx, dtype=xp.float32)
        vz = xp.asarray(vz, dtype=xp.float32)
        sxx = xp.asarray(sxx, dtype=xp.float32)
        szz = xp.asarray(szz, dtype=xp.float32)
        sxz = xp.asarray(sxz, dtype=xp.float32)

        if nx is None or nz is None:
            nx_, nz_ = vx.shape
            nx = nx if nx is not None else nx_
            nz = nz if nz is not None else nz_

        if sigma is None or sigma <= 0.0:
            out_cast = (vx.astype(xp.float32), vz.astype(xp.float32),
                        sxx.astype(xp.float32), szz.astype(xp.float32), sxz.astype(xp.float32))
            if transposed:
                return tuple(a.T for a in out_cast)
            return out_cast

        def Smooth(arr, s):
            zeros = xp.zeros_like(arr)
            return self.Image_Gaussian(arr, zeros, sigma=s, truncate=3.0, mode=mode, use_gpu=(xp is cp))

        def Local_Normalized_Cross_Correlation(a, b, local_s):
            prod = a * b
            a2 = a * a
            b2 = b * b
            num = Smooth(prod, local_s)
            den1 = Smooth(a2, local_s)
            den2 = Smooth(b2, local_s)
            denom = xp.sqrt(den1 * den2 + eps)
            ncc = num / denom
            ncc = xp.clip(ncc, 0.0, 1.0)
            if gamma != 1.0 and gamma > 0.0:
                ncc = ncc ** float(gamma)
            return ncc

        vx_s = Smooth(vx, sigma)
        vz_s = Smooth(vz, sigma)
        sxx_s = Smooth(sxx, sigma)
        szz_s = Smooth(szz, sigma)
        sxz_s = Smooth(sxz, sigma)

        vx_ncc = Local_Normalized_Cross_Correlation(vx, vx_s, local_sigma)
        vz_ncc = Local_Normalized_Cross_Correlation(vz, vz_s, local_sigma)
        sxx_ncc = Local_Normalized_Cross_Correlation(sxx, sxx_s, local_sigma)
        szz_ncc = Local_Normalized_Cross_Correlation(szz, szz_s, local_sigma)
        sxz_ncc = Local_Normalized_Cross_Correlation(sxz, sxz_s, local_sigma)

        coherence = (vx_ncc + vz_ncc + sxx_ncc + szz_ncc + sxz_ncc) / 5.0
        if strength != 1.0:
            coherence = xp.clip(coherence * float(strength), 0.0, 1.0)

        def Blend(orig, smooth, coh):
            return coh * orig + (1.0 - coh) * smooth

        vx_d = Blend(vx, vx_s, coherence)
        vz_d = Blend(vz, vz_s, coherence)
        sxx_d = Blend(sxx, sxx_s, coherence)
        szz_d = Blend(szz, szz_s, coherence)
        sxz_d = Blend(sxz, sxz_s, coherence)

        emin, emax = float(energy_clip[0]), float(energy_clip[1])

        def Energy_Normalize(orig, denoised, norm_s):
            E_orig = Smooth(orig * orig, norm_s)
            E_den = Smooth(denoised * denoised, norm_s)
            ratio = xp.sqrt((E_orig + eps) / (E_den + eps))
            ratio = xp.clip(ratio, emin, emax)
            return denoised * ratio

        vx_d = Energy_Normalize(vx, vx_d, local_sigma)
        vz_d = Energy_Normalize(vz, vz_d, local_sigma)
        sxx_d = Energy_Normalize(sxx, sxx_d, local_sigma)
        szz_d = Energy_Normalize(szz, szz_d, local_sigma)
        sxz_d = Energy_Normalize(sxz, sxz_d, local_sigma)

        vx_out = vx_d.astype(xp.float32, copy=False)
        vz_out = vz_d.astype(xp.float32, copy=False)
        sxx_out = sxx_d.astype(xp.float32, copy=False)
        szz_out = szz_d.astype(xp.float32, copy=False)
        sxz_out = sxz_d.astype(xp.float32, copy=False)

        if inplace:
            try:
                if isinstance(vx, type(vx_out)):
                    vx[:] = vx_out
                if isinstance(vz, type(vz_out)):
                    vz[:] = vz_out
                if isinstance(sxx, type(sxx_out)):
                    sxx[:] = sxx_out
                if isinstance(szz, type(szz_out)):
                    szz[:] = szz_out
                if isinstance(sxz, type(sxz_out)):
                    sxz[:] = sxz_out
                out_tuple = (vx, vz, sxx, szz, sxz)
            except Exception:
                out_tuple = (vx_out, vz_out, sxx_out, szz_out, sxz_out)
        else:
            out_tuple = (vx_out, vz_out, sxx_out, szz_out, sxz_out)

        if transposed:
            out_tuple = tuple(a.T for a in out_tuple)

        return out_tuple


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
        # Checks and Backend
        if not self.has_cupy:
            raise RuntimeError("CuPy required for GPU execution in this optimized run()")
        if survey_type not in ("land", "marine"):
            raise ValueError("survey_type must be 'land' or 'marine'")

        xp = cp
        device = int(self.device)

        if external_pdata is not None:
            pdata = external_pdata

        if pdata is None and not do_forward:
            raise ValueError("pdata is None but do_forward==False. Provide external_pdata or set do_forward=True.")

        # if imaging and do_backward and not do_forward:
        #     raise ValueError(
        #         "Imaging with backward propagation requires forward modelling (do_forward=True). "
        #         "Either set do_forward=True or provide external_pdata with forward traces."
        #     )

        if imaging and do_backward and not do_forward:
            if self.verbose:
                self.log("[RTM] imaging + backward requested but do_forward=False - enabling do_forward=True automatically.")
            do_forward = True

        if pdata is not None:
            nshot, totaltr, nt = pdata.shape
        else:
            nshot = 10
            totaltr = int(min(600, vp.shape[0]))
            vmax = float(vp.max())
            h = dx * dz / math.sqrt(dx*dx + dz*dz)
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
        h = dx * dz / math.sqrt(dx*dx + dz*dz)
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

        # Decide checkpoint
        do_checkpoint = False
        if max_gpu_snap_bytes is not None and approx_snap_bytes > max_gpu_snap_bytes:
            do_checkpoint = True
        if checkpoint_dir is not None:
            do_checkpoint = True

        # Prepare outputs
        if (vs is not None) and (rho is not None):
            elastic_mode = True
            image_PP = cp.zeros((nx_max, nz), dtype=cp.float32)
            image_PS = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum_PP = cp.zeros_like(image_PP)
        else:
            elastic_mode = False
            image_global = cp.zeros((nx_max, nz), dtype=cp.float32)
            illum = cp.zeros_like(image_global)

        if denoise_params is None:
            denoise_params = {"sigma": 1.5, "local_sigma": 3.0, "gamma": 1.0}

        # Precompute FD weights
        _ = self.Ensure_FD_Weights(3, dx)

        # Optional pinned buffer for receiver copy
        pinned_host_buf = None
        pdata_host_out = None
        if do_forward and pdata is None and return_pdata:
            pdata_host_out = np.zeros((nshot, totaltr, ntnewmax), dtype=np.float32)

        #----------------
        # SHOT LOOP
        #----------------
        for ishot in tqdm(range(nshot), desc="[RTM] Shot", unit="shot"):
            if self.verbose:
                self.log(f"[RTM] Shot {ishot+1}/{nshot}")

            d_vx_fwd_snap_shot = None
            d_vz_fwd_snap_shot = None
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

            # Elastic blocks
            if elastic_mode:
                vs_block = (vs[src_ixs, :].astype(np.float32) if hasattr(vs, "shape") and vs.shape == vp.shape else (vs if np.isscalar(vs) else vs[src_ixs, :].astype(np.float32)))
                rho_block = (rho[src_ixs, :].astype(np.float32) if hasattr(rho, "shape") and rho.shape == vp.shape else (rho if np.isscalar(rho) else rho[src_ixs, :].astype(np.float32)))
                mu_block = rho_block * (vs_block ** 2)
                lam_block = rho_block * (vp_block ** 2 - 2.0 * (vs_block ** 2))

            # Move to device and allocate fields
            with cp.cuda.Device(device):
                d_vp = cp.asarray(vp_block, dtype=cp.float32)
                if elastic_mode:
                    d_vs = cp.asarray(vs_block, dtype=cp.float32)
                    d_rho = cp.asarray(rho_block, dtype=cp.float32)
                    d_mu = cp.asarray(mu_block, dtype=cp.float32)
                    d_lam = cp.asarray(lam_block, dtype=cp.float32)

                    vx = cp.zeros((nx, nz), dtype=cp.float32)
                    vz = cp.zeros_like(vx)
                    sxx = cp.zeros_like(vx)
                    szz = cp.zeros_like(vx)
                    sxz = cp.zeros_like(vx)

                    # Pressure-equivalent buffers for ABC (elastic)
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

                # Snapshot container

                # if do_forward:
                #     if do_checkpoint:
                #         if checkpoint_dir is None:
                #             checkpoint_dir = "/tmp/rtm_chk"
                #         os.makedirs(checkpoint_dir, exist_ok=True)
                #         snap_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps.dat")
                #         snap_memmap = np.memmap(snap_path, dtype=np.float16 if snap_dtype_cp==cp.float16 else np.float32,
                #                                 mode="w+", shape=(n_snapshots, nx, nz))
                #     else:
                #         d_vx_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)
                #         d_vz_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)

                if do_forward:
                    if do_checkpoint:
                        if checkpoint_dir is None:
                            checkpoint_dir = "/tmp/rtm_chk"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        dtype_np = np.float16 if snap_dtype_cp == cp.float16 else np.float32
                        snap_vx_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps_vx.dat")
                        snap_vz_path = os.path.join(checkpoint_dir, f"shot_{ishot:04d}_snaps_vz.dat")
                        snap_memmap_vx = np.memmap(snap_vx_path, dtype=dtype_np, mode="w+", shape=(n_snapshots, nx, nz))
                        snap_memmap_vz = np.memmap(snap_vz_path, dtype=dtype_np, mode="w+", shape=(n_snapshots, nx, nz))
                        snap_memmap = (snap_memmap_vx, snap_memmap_vz)
                    else:
                        d_vx_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)
                        d_vz_fwd_snap_shot = cp.zeros((n_snapshots, nx, nz), dtype=snap_dtype_cp)

                # Prepare pinned host buffer once
                bytes_per_shot = ntr * ntnewmax * np.dtype(np.float32).itemsize
                if pinned_host_buf is None:
                    try:
                        pinned_host_buf = cp.cuda.alloc_pinned_memory(min(bytes_per_shot, 2 ** 30))
                    except Exception:
                        pinned_host_buf = None
                if pinned_host_buf is not None:
                    host_flat = np.frombuffer(pinned_host_buf, dtype=np.float32, count=ntr*ntnewmax)
                    host_view = host_flat.reshape((ntr, ntnewmax))
                else:
                    host_view = None

                #------------------------
                # Forward Propagation
                #------------------------
                if do_forward:
                    for it in range(ntnewmax):
                        if elastic_mode:
                            vx, vz, sxx, szz, sxz = self.Elastic_TimeDomain_Propagation(
                                vx, vz, sxx, szz, sxz,
                                d_rho, d_lam, d_mu,
                                nx, dx, nz, dz, dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                # pad_mode='edge',
                                # pad_mode='nearest',
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=False
                            )

                            # if denoise_forward:
                            #     vx, vz, sxx, szz, sxz = self.Elastic_Denoise(
                            #         vx, vz, sxx, szz, sxz, nx=nx, nz=nz,
                            #         dx=dx, dz=dz, use_gpu=True,
                            #         **(denoise_params or {})
                            #     )

                            if self.verbose and (it % 50 == 0):
                                energy = float(cp.sum(vx*vx + vz*vz + sxx*sxx + szz*szz + sxz*sxz).astype(cp.float64))
                                self.log(f"[Forward] it={it} energy={energy:.6e}")

                            # Inject source into stresses
                            if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                                val = float(d_w[it])
                                sxx[src_local, src_depth] += val
                                szz[src_local, src_depth] += val

                            # Compute pressure-equivalent
                            p_pres[...] = 0.5 * (sxx + szz)

                            # Apply Absorbing_Boundary_Condition
                            try:
                                # abs_flags = (1, 1, 1, 1)
                                # pp_tmp = p_pres.copy()
                                # p_pres_mod = self.Absorbing_Boundary_Condition(
                                #     nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                #     dvv=d_vp, od=None, p=p_pres, pm=pm_pres,
                                #     pp=pp_tmp, abs_flags=(1, 1, 1, 1)
                                # )
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
                            pdata_shot_d[:, it] = pres[lxpad:lxpad + ntr, rec_depth]

                        else:
                            # Acoustic path using provided TimeDomain_Propagation
                            self.k["TimeDomain_Propagation"](d_v, d_pn, d_pp, nx=nx, dx=dx, nz=nz, dz=dz, dt=dt)
                            if it < d_w.size and 0 <= src_local < nx and 0 <= src_depth < nz:
                                d_pp[src_local, src_depth] += d_w[it] * float(dt)
                            # Swap buffers and record
                            d_pn, d_pp = d_pp, d_pn

                            # Apply Absorbing_Boundary_Condition
                            try:
                                # abs_flags = (1, 1, 1, 1)
                                # pp_tmp = d_pn.copy()
                                # d_pn = self.Absorbing_Boundary_Condition(
                                #     nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                #     dvv=d_v, od=None, p=d_pn, pm=d_pp, pp=pp_tmp,
                                #     abs_flags=abs_flags
                                # )
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_v, d_pn, d_pn, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            pdata_shot_d[:, it] = d_pn[lxpad:lxpad + ntr, rec_depth]

                        # Snapshotting
                        if imaging and (it % snap_stride == 0):
                            sid = it // snap_stride

                            # if snap_dtype_cp == cp.float16:
                            #     d_vx_fwd_snap_shot[sid, lxpad:lxpad + nx, :].view(cp.float16)[:] = vx.astype(cp.float16)
                            #     d_vz_fwd_snap_shot[sid, lxpad:lxpad + nx, :].view(cp.float16)[:] = vz.astype(cp.float16)
                            # else:
                            #     d_vx_fwd_snap_shot[sid, lxpad:lxpad + nx, :] = vx
                            #     d_vz_fwd_snap_shot[sid, lxpad:lxpad + nx, :] = vz

                            # if snap_dtype_cp == cp.float16:
                            #     d_vx_fwd_snap_shot[sid, :, :] = vx.astype(cp.float16)
                            #     d_vz_fwd_snap_shot[sid, :, :] = vz.astype(cp.float16)
                            # else:
                            #     d_vx_fwd_snap_shot[sid, :, :] = vx
                            #     d_vz_fwd_snap_shot[sid, :, :] = vz

                            if do_checkpoint:
                                if self.stream is not None:
                                    self.stream.synchronize()
                                else:
                                    cp.cuda.runtime.deviceSynchronize()
                                if snap_dtype_cp == cp.float16:
                                    arr_vx = cp.asnumpy(vx.astype(cp.float16))
                                    arr_vz = cp.asnumpy(vz.astype(cp.float16))
                                else:
                                    arr_vx = cp.asnumpy(vx.astype(cp.float32))
                                    arr_vz = cp.asnumpy(vz.astype(cp.float32))
                                snap_memmap[0][sid,:,:] = arr_vx
                                snap_memmap[1][sid,:,:] = arr_vz
                            else:
                                if snap_dtype_cp == cp.float16:
                                    d_vx_fwd_snap_shot[sid,:,:] = vx.astype(cp.float16)
                                    d_vz_fwd_snap_shot[sid,:,:] = vz.astype(cp.float16)
                                else:
                                    d_vx_fwd_snap_shot[sid,:,:] = vx
                                    d_vz_fwd_snap_shot[sid,:,:] = vz

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

                    if self.verbose:
                        self.log("[Backward] trace max/min:", float(traces_gpu.max()), float(traces_gpu.min()))

                    if elastic_mode:
                        vx_rec.fill(0.0)
                        vz_rec.fill(0.0)
                        sxx_rec.fill(0.0)
                        szz_rec.fill(0.0)
                        sxz_rec.fill(0.0)

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            # Vectorized injection into stresses at receiver depth
                            ix_arr = cp.arange(ntr, dtype=cp.int32)
                            ix_global = ix_arr + lxpad
                            valid = (ix_global >= 0) & (ix_global < nx) & (rec_depth >= 0) & (rec_depth < nz)
                            if valid.any():
                                idx = ix_global[valid]
                                sxx_rec[idx, rec_depth] += trace_t[valid]
                                szz_rec[idx, rec_depth] += trace_t[valid]

                            # Propagate backward one step
                            vx_rec, vz_rec, sxx_rec, szz_rec, sxz_rec = self.Elastic_TimeDomain_Propagation(
                                vx_rec, vz_rec, sxx_rec, szz_rec, sxz_rec,
                                d_rho, d_lam, d_mu,
                                nx, dx, nz, dz, dt,
                                fd_order_radius=3,
                                use_gpu=True,
                                # pad_mode='edge',
                                # pad_mode='nearest',
                                pad_mode='reflect',
                                absorb_R=0,
                                src=None,
                                inplace=True,
                                use_external_abc=False
                            )

                            # Apply Absorbing_Boundary_Condition
                            try:
                                pres_rec = 0.5 * (sxx_rec + szz_rec)
                                # p_rec_mod = self.Absorbing_Boundary_Condition(
                                #     nx, dx, nz, dz, dt, d_vp, None, pres_rec,
                                #     pm_pres, pres_rec, (1,1,1,1)
                                # )
                                # pp_tmp = pres_rec.copy()
                                # p_rec_mod = self.Absorbing_Boundary_Condition(
                                #     nx=nx, dx=dx, nz=nz, dz=dz, dt=dt,
                                #     dvv=d_vp, od=None, p=pres_rec, pm=pm_pres, pp=pp_tmp,
                                #     abs_flags=(1, 1, 1, 1)
                                # )
                                mask_b = (p_rec_mod != pres_rec)
                                if mask_b.any():
                                    sxx_rec[mask_b] = p_rec_mod[mask_b]
                                    szz_rec[mask_b] = p_rec_mod[mask_b]
                                pm_pres[...] = pres_rec.copy()
                            except Exception:
                                try:
                                    if "absorbgpuup" in self.k:
                                        self.k["absorbgpuup"](d_vp, sxx_rec, sxx_rec, nx, nz, dx, dz, dt, iz=1)
                                        self.k["absorbgpuup"](d_vp, szz_rec, szz_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            # Apply external ABC
                            if "absorbgpuup" in self.k:
                                try:
                                    self.k["absorbgpuup"](d_vp, vx_rec, vx_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, vz_rec, vz_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, sxx_rec, sxx_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, szz_rec, szz_rec, nx, nz, dx, dz, dt, iz=1)
                                    self.k["absorbgpuup"](d_vp, sxz_rec, sxz_rec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            # Imaging at snapshot times
                            if imaging and (it % snap_stride == 0):
                                sid = it // snap_stride
                                if do_checkpoint:
                                    # vx_fwd_local = cp.asarray(snap_memmap[sid, :, :].astype(np.float32))
                                    # vz_fwd_local = cp.asarray(snap_memmap[sid, :, :].astype(np.float32))  # best-effort
                                    vx_host = snap_memmap[0][sid,:,:].astype(np.float32)
                                    vz_host = snap_memmap[1][sid,:,:].astype(np.float32)
                                    # Copy to device
                                    vx_fwd_local = cp.asarray(vx_host)
                                    vz_fwd_local = cp.asarray(vz_host)
                                else:
                                    if d_vx_fwd_snap_shot is None:
                                        raise RuntimeError("Forward snapshots are not available for imaging. Ensure do_forward=True and imaging=True.")
                                    # vx_fwd_local = d_vx_fwd_snap_shot[sid, lxpad:lxpad + nx, :].astype(cp.float32)
                                    # vz_fwd_local = d_vz_fwd_snap_shot[sid, lxpad:lxpad + nx, :].astype(cp.float32)
                                    vx_fwd_local = d_vx_fwd_snap_shot[sid, :, :].astype(cp.float32)
                                    vz_fwd_local = d_vz_fwd_snap_shot[sid, :, :].astype(cp.float32)

                                # Derivatives
                                def d_dx(a, out=None):
                                    if out is None:
                                        out = cp.empty_like(a)
                                    out[1:-1, :] = (a[2:, :] - a[:-2, :]) * (0.5 / dx)
                                    out[0, :] = out[1, :]
                                    out[-1, :] = out[-2, :]
                                    return out

                                def d_dz(a, out=None):
                                    if out is None:
                                        out = cp.empty_like(a)
                                    out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * (0.5 / dz)
                                    out[:, 0] = out[:, 1]
                                    out[:, -1] = out[:, -2]
                                    return out

                                div_fwd = d_dx(vx_fwd_local) + d_dz(vz_fwd_local)
                                div_bwd = d_dx(vx_rec) + d_dz(vz_rec)
                                curl_bwd = d_dx(vz_rec) - d_dz(vx_rec)

                                # d_result_PP = div_fwd * div_bwd
                                # d_result_PS = div_fwd * curl_bwd

                                d_result_PP = self.Cross_Correlation(div_fwd, div_bwd, use_gpu=True, accumulate=False)
                                d_result_PS = self.Cross_Correlation(div_fwd, curl_bwd, use_gpu=True, accumulate=False)

                                image_PP[ix22:ix22 + nx, :] += d_result_PP
                                image_PS[ix22:ix22 + nx, :] += d_result_PS
                                illum_PP[ix22:ix22 + nx, :] += (vx_fwd_local * vx_fwd_local + vz_fwd_local * vz_fwd_local)

                    else:
                        # Acoustic backward imaging (Cross_Correlation)
                        d_pnrec.fill(0.0); d_pprec.fill(0.0)
                        d_result = cp.zeros((nx, nz), dtype=cp.float32)

                        for it in range(ntnewrec - 1, -1, -1):
                            if it < traces_gpu.shape[1]:
                                trace_t = traces_gpu[:, it]
                            else:
                                trace_t = cp.zeros((ntr,), dtype=cp.float32)

                            self.k["receivedata"](d_pnrec, trace_t, nz, ntr, lxpad, 0)

                            # Step acoustic backward
                            self.k["TimeDomain_Propagation"](d_v, d_pnrec, d_pprec, nx=nx, dx=dx, nz=nz, dz=dz, dt=dt)
                            if "absorbgpuup" in self.k:
                                try:
                                    self.k["absorbgpuup"](d_v, d_pnrec, d_pprec, nx, nz, dx, dz, dt, iz=1)
                                except Exception:
                                    pass

                            if imaging and (it % snap_stride == 0):
                                sid = it // snap_stride
                                if do_checkpoint:
                                    u_fwd_local = cp.asarray(snap_memmap[sid, :, :].astype(np.float32))
                                else:
                                    if d_vx_fwd_snap_shot is None:
                                        raise RuntimeError("Forward snapshots are not available for imaging. Ensure do_forward=True and imaging=True.")
                                    u_fwd_local = d_vx_fwd_snap_shot[sid, lxpad:lxpad + nx, :].astype(cp.float32)

                                # Cross-correlation
                                corr = self.Cross_Correlation(u_fwd_local, d_pnrec, image=None, use_gpu=True)
                                d_result += corr
                                illum[ix22:ix22 + nx, :] += u_fwd_local * u_fwd_local

                            # Swap buffers
                            d_pnrec, d_pprec = d_pprec, d_pnrec

                        image_global[ix22:ix22 + nx, :] += d_result

                # Free shot-level GPU arrays
                try:
                    del d_vp
                    if elastic_mode:
                        del d_vs, d_rho, d_mu, d_lam
                    del pdata_shot_d
                    if (d_vx_fwd_snap_shot is not None) and (not do_checkpoint):
                        del d_vx_fwd_snap_shot, d_vz_fwd_snap_shot
                except Exception:
                    pass
                cp.cuda.runtime.deviceSynchronize()

        # End shot loop

        # Normalize and final denoise
        if elastic_mode:
            image_PP = image_PP / (illum_PP + EPS)
            image_PS = image_PS / (illum_PP + EPS)

            # Applying Preconditioning
            image_PP = self.Preconditioning(image_PP, nx=nx_max, minnz=src_depth+2, maxnz=nz, nz=nz, use_gpu=True)
            image_PS = self.Preconditioning(image_PS, nx=nx_max, minnz=src_depth+2, maxnz=nz, nz=nz, use_gpu=True)

            # if "Elastic_Denoise" in self.k:
            #     try:
            #         d_out_PP = cp.zeros_like(image_PP)
            #         self.k.get("Elastic_Denoise", lambda *a, **k: None)(
            #             cp.ones_like(image_PP), image_PP, d_out_PP,
            #             nx=nx_max, dx=dx, nz=nz, dz=dz
            #         )
            #         image_PP = d_out_PP
            #     except Exception:
            #         pass

            # image_PP = self.Image_Denoise_Normalize_Cross_Correlation(image_PP, sigma=SIGMA, local_sigma=LOCAL_SIGMA, gamma=GAMMA, use_gpu=True)
            # image_PS = self.Image_Denoise_Normalize_Cross_Correlation(image_PP, sigma=SIGMA, local_sigma=LOCAL_SIGMA, gamma=GAMMA, use_gpu=True)

            out = cp.asnumpy((image_PP + 0.5 * image_PS))
            # out = out / (np.max(np.abs(out)) + 1e-12)
            out = out / (np.max(np.abs(out)))

        else:
            image_global = image_global / (illum + EPS)

            # Applying Preconditioning
            image_global = self.Preconditioning(image_global, nx=nx_max, minnz=src_depth+2, maxnz=nz, nz=nz, use_gpu=True)

            d_out = cp.zeros_like(image_global)

            # if "Helmholtz_Denoise" in self.k:
            #     try:
            #         self.k["Helmholtz_Denoise"](
            #             cp.ones_like(image_global),
            #             image_global,
            #             d_out,
            #             nx=nx_max, dx=dx, nz=nz, dz=dz
            #         )
            #         out = cp.asnumpy(d_out)
            #     except Exception:
            #         out = cp.asnumpy(image_global)
            # else:
            #     out = cp.asnumpy(image_global)

            out = out / (np.max(np.abs(out)) + 1e-12)

        if return_pdata and pdata_host_out is not None:
            return out, pdata_host_out
        return out








