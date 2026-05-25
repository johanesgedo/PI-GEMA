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

"""
Natural Frequency Calibration with Physics-Informed Neural Network
"""

import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional, Tuple, Dict, Any

NFFT = 4096
FMIN = 0.1      # Minimum frequency taken
FMAX = 20.0     # Maximum frequency taken
NPERSEG = 2048
OVERLAP = 0.5
EPOCHS = 2000
LEARNING_RATE = 1e-3
LAMBDA_SMOOTH = 1e-4
LAMBDA_PHYSICS = 1e-4
HIDDEN_LAYERS = (16,16,16)
N_SAMPLES = 2000


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



class Ambient_Noise_Statistical_Model:

    def __init__(
        self,
        fs: float,
        nfft: int = NFFT,
        fmin: float = FMIN,
        fmax: float = FMAX,
        nperseg: int = NPERSEG,
        overlap: float = OVERLAP,
        dtype: str = "float32",
        device: int = 0,
        verbose: bool = False
    ):
        self.cp = cp
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.nperseg = int(nperseg)
        self.overlap = float(overlap)
        self.dtype = getattr(cp, dtype)
        self.device = device
        self.verbose = verbose
        cp.cuda.Device(device).use()
        self.freqs = None
        self.band_mask = None
        self.S_ref_mean = None
        self.S_ref_std = None
        self.metadata = {}

    def Build_Frequency_Axis(self):
        cp = self.cp
        freqs = cp.fft.rfftfreq(self.nfft, d=1/self.fs)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        self.freqs = freqs[mask]
        self.band_mask = mask
        return self.freqs

    def Compute_PSD(self, Z, N, E):
        cp = self.cp
        Z = cp.asarray(Z, dtype=self.dtype)
        N = cp.asarray(N, dtype=self.dtype)
        E = cp.asarray(E, dtype=self.dtype)
        if not (Z.size == N.size == E.size):
            raise ValueError("Z, N, E must have same length")
        hop = int(self.nperseg * (1-self.overlap))
        if hop <= 0:
            hop = 1
        npts = Z.size
        n_frames = (npts - self.nperseg) // hop + 1
        # Build frames via indexing
        idx = cp.arange(self.nperseg)[None,:] + cp.arange(n_frames)[:,None] * hop
        Zf = Z[idx]
        Nf = N[idx]
        Ef = E[idx]
        # Apply Hann window
        win = cp.hanning(self.nperseg).astype(self.dtype)
        Zf *= win
        Nf *= win
        Ef *= win
        # FFT (batched)
        FZ = cp.fft.rfft(Zf, n=self.nfft, axis=1)
        FN = cp.fft.rfft(Nf, n=self.nfft, axis=1)
        FE = cp.fft.rfft(Ef, n=self.nfft, axis=1)
        # PSD
        PSD_Z = cp.abs(FZ) ** 2
        PSD_N = cp.abs(FN) ** 2
        PSD_E = cp.abs(FE) ** 2
        # Average 3C
        PSD_3C = (PSD_Z + PSD_N + PSD_E) / 3.0
        # Welch average
        PSD_mean = cp.mean(PSD_3C, axis=0)
        # Apply band mask
        PSD_band = PSD_mean[self.band_mask]
        return PSD_band

    def Build_Statistical_Reference(
        self,
        list_of_3C_data
    ):
        cp = self.cp
        if self.freqs is None:
            self.Build_Frequency_Axis()
        PSD_list = []
        for Z, N, E in list_of_3C_data:
            PSD = self.Compute_PSD(Z, N, E)
            PSD_list.append(PSD)
        PSD_stack = cp.stack(PSD_list, axis=0)
        self.S_ref_mean = cp.mean(PSD_stack, axis=0)
        self.S_ref_std = cp.std(PSD_stack, axis=0)
        self.metadata = {
            "n_records": len(list_of_3C_data),
            "fs": self.fs,
            "nfft": self.nfft,
            "band": (self.fmin, self.fmax)
        }
        return self.S_ref_mean, self.S_ref_std

    def Save_Reference():
        if self.S_ref_mean is None:
            raise RuntimeError("Reference not built yet")
        np.save(
            filename,
            freqs=self.cp.asnumpy(self.freqs),
            S_ref_mean=self.cp.asnumpy(self.S_ref_mean),
            S_ref_std=self.cp.asnumpy(self.S_ref_std),
            metadata=self.metadata
        )

    def Load_Reference(
        self,
        filename
    ):
        data = np.load(filename, allow_pickle=True)
        self.freqs = self.cp.asarray(data["freqs"])
        self.S_ref_mean = self.cp.asarray(data["S_ref_std"])
        self.metadata = data["metadata"].item()
        full_freqs = self.cp.fft.rfftfreq(self.nfft, d=1/self.fs)
        self.band_mask = (full_freqs >= self.fmin) & (full_freqs <= self.fmax)
        return self.S_ref_mean


"""
class PINN_Instrument_Model:

    def __init__(
        self,
        freqs,
        S_reference,
        lr=1e-3,
        lambda_smooth=1e-4,
        lambda_physics=1e-4,
        device=0,
        verbose=False
    ):
        cp.cuda.Device(device).use()
        self.cp = cp
        self.freqs = cp.asarray(freqs, dtype=cp.float32)
        self.S_ref = cp.asarray(S_reference, dtype=cp.float32)
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics
        self.verbose = verbose
        self.f0 = cp.array(30.0, dtype=cp.float32)
        self.zeta = cp.array(0.7, dtype=cp.float32)
        self.loss_history = []

    def Build_Network(self):
        return {"f0": self.f0, "zeta":self.zeta}

    def SDoF_Transfer(self):
        cp = self.cp
        w = 2 * cp.pi * self.freqs
        w0 = 2 * cp.pi * self.f0
        denom = (w0**2 - w**2) + 2j * self.zeta * w0 * w
        H = (w0**2) / denom
        return H

    def Loss_Function(self):
        cp = self.cp
        H = self.SDoF_Transfer()
        H_mag2 = cp.abs(H) ** 2
        loss_data = cp.mean((H_mag2 - self.S_ref) ** 2)
        grad = cp.gradient(cp.abs(H_mag2), self.freqs)
        loss_smooth = cp.mean(grad**2)
        loss_phys = cp.maximum(0, -self.zeta)**2
        total_loss = (
            loss_data + self.lambda_smooth * loss_smooth
                      + self.lambda_physics * loss_phys
        )
        return total_loss

    def Compute_Gradients(self, eps=1e-3):
        cp = self.cp
        L0 = self.Loss_Function()
        f0_orig = self.f0.copy()
        self.f0 += eps
        L_f0 = self.Loss_Function()
        grad_f0 = (L_f0 - L0) / eps
        self.f0 = f0_orig
        zeta_orig = self.zeta.copy()
        self.zeta += eps
        L_zeta = self.Loss_Function()
        grad_zeta = (L_zeta - L0) / eps
        self.zeta = zeta_orig
        return grad_f0, grad_zeta

    def Train(self, epochs=2000):
        cp = self.cp
        for epoch in range(epochs):
            grad_f0, grad_zeta = self.Compute_Gradients()
            self.f0 -= self.lr * grad_f0
            self.zeta -= self.lr * grad_zeta
            self.f0 = cp.maximum(self.f0, 1.0)
            self.zeta = cp.maximum(self.zeta, 0.01)
            loss = self.Loss_Function()
            self.loss_history.append(float(loss.get()))
            if self.verbose and epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss={float(loss.get()):.6e} | "
                      f"f0={float(self.f0.get()):.3f} | "
                      f"zeta={float(self.zeta.get()):.3f}")

    def Predict_Transfer_Function(self, return_numpy=True):
        H = self.SDoF_Transfer()
        if return_numpy:
            return self.cp.asnumpy(H)
        return H
"""


class PINN_Instrument_Model(nn.Module):

    def __init__(
        self,
        freqs,
        S_reference,
        hidden_layers=HIDDEN_LAYERS,
        lr=LEARNING_RATE,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_physics=LAMBDA_PHYSICS,
        device=None,
        learn_physical_params=True,
        verbose=False
    ):
        super().__init__()

        if isinstance(device, torch.device):
            dev = device
        else:
            if device is None:
                dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            elif isinstance(device, int):
                if torch.cuda.is_available():
                    dev_str = f"cuda:{device}"
                else:
                    warnings.warn("Requested GPU index but CUDA is not available; falling back to CPU")
                    dev_str = "cpu"
            elif isinstance(device, str):
                if device.startswith("cuda") and not torch.cuda.is_available():
                    warnings.warn(f"Requested device '{device}' but CUDA is not available; falling back to CPU")
                    dev_str = "cpu"
                else:
                    dev_str = device
            else:
                dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

        self.device = dev
        self.verbose = verbose

        freqs_t = torch.tensor(freqs, dtype=torch.float32, device=self.device).view(-1,1)
        Sref_t  = torch.tensor(S_reference, dtype=torch.float32, device=self.device).view(-1,1)
        freqs_t.requires_grad_(True)

        self.freqs = freqs_t
        self.S_ref = Sref_t
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics

        layers = []
        in_ch = 1
        self.freq_scale = float(freqs_t.abs().max().item()) if freqs_t.numel() > 0 else 1.0
        for h in hidden_layers:
            layers.append(nn.Linear(in_ch, h))
            layers.append(nn.Tanh())
            in_ch = h
        layers.append(nn.Linear(in_ch, 2))
        self.net = nn.Sequential(*layers).to(self.device)

        if learn_physical_params:
            self.log_f0 = nn.Parameter(torch.log(torch.tensor(30.0, dtype=torch.float32, device=self.device)))
            self.zeta_param = nn.Parameter(torch.tensor(0.7, dtype=torch.float32, device=self.device))
        else:
            self.register_buffer("log_f0", torch.log(torch.tensor(30.0, dtype=torch.float32, device=self.device)))
            self.register_buffer("zeta_param", torch.tensor(0.7, dtype=torch.float32, device=self.device))

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_history = []


    def Forward(self, freqs=None):
        if freqs is None:
            freqs = self.freqs
        x = (freqs / self.freq_scale).to(self.device)
        out = self.net(x)
        Hr = out[:,0:1]
        Hi = out[:,1:2]
        return Hr, Hi

    def Physical_Pamarameters(self):
        f0 = torch.exp(self.log_f0)
        zeta = torch.nn.functional.softplus(self.zeta_param)
        return f0, zeta

    def Residual(
        self,
        Hr,
        Hi
    ):
        w = 2.0 * torch.pi * self.freqs
        f0, zeta = self.Physical_Pamarameters()
        w0 = 2.0 * torch.pi * f0
        A = (w0**2 - w**2)
        B = 2.0 * zeta * w0 * w
        R_re = A * Hr - B * Hi - (w0**2)
        R_im = A * Hi + B * Hr
        return R_re, R_im

    def Loss_Function(self):
        Hr, Hi = self.Forward()
        H_mag2 = Hr**2 + Hi**2
        loss_data = torch.mean((H_mag2 - self.S_ref)**2)
        R_re, R_im = self.Residual(Hr, Hi)
        res_sq = R_re**2 + R_im**2
        loss_phys = torch.mean(res_sq)
        grad_H = torch.autograd.grad(H_mag2.sum(), self.freqs, create_graph=True)[0]
        loss_smooth = torch.mean(grad_H**2)
        total_loss = loss_data + self.lambda_physics * loss_phys + self.lambda_smooth * loss_smooth
        comps = {"loss_data": loss_data.item(), "loss_phys": loss_phys.item(), "loss_smooth": loss_smooth.item()}
        return total_loss, comps

    def Train_Model(
        self,
        epochs=2000,
        print_every=200
    ):
        self.train()
        for ep in range(epochs):
            self.optimizer.zero_grad()
            loss, comps = self.Loss_Function()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.loss_history.append(float(loss.detach().cpu().numpy()))
            if self.verbose and (ep % print_every == 0 or ep == epochs-1):
                f0_val, zeta_val = (self.Physical_Pamarameters()[0].item(), self.Physical_Pamarameters()[1].item())
                print(f"Epoch {ep:5d} | "
                      f"Loss={loss.item():.6e} | "
                      f"data={comps['loss_data']:.3e} | "
                      f"phys={comps['loss_phys']:.3e} | "
                      f"smooth={comps['loss_smooth']:.3e} | "
                      f"f0={f0_val:.3f} | "
                      f"zeta={zeta_val:.3f}")

    def Predict_Transfer(
        self,
        return_numpy=True
    ):
        self.eval()
        with torch.no_grad():
            Hr, Hi = self.Forward()
            if return_numpy:
                Hr_n = Hr.cpu().numpy().ravel()
                Hi_n = Hi.cpu().numpy().ravel()
                return Hr_n + 1j * Hi_n
            else:
                return Hr + 1j * Hi




class Smartphone_Calibration:

    def __init__(
        self,
        fs,
        S_reference,
        freqs,
        nfft=NFFT,
        regularization=1e-6,
        device=0,
        verbose=False
    ):
        cp.cuda.Device(device).use()
        self.cp = cp
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.verbose = verbose
        self.S_ref = cp.asarray(S_reference, dtype=cp.float32)
        self.freqs = cp.asarray(freqs, dtype=cp.float32)
        self.regularization = regularization
        self.H_empirical = None
        self.H_regularized = None

    def Estimate_Empirical_Ratio(
        self,
        Z,
        N,
        E
    ):
        cp = self.cp
        Z = cp.asarray(Z, dtype=cp.float32)
        N = cp.asarray(N, dtype=cp.float32)
        E = cp.asarray(E, dtype=cp.float32)
        # FFT
        FZ = cp.fft.rfft(Z, n=self.nfft)
        FN = cp.fft.rfft(N, n=self.nfft)
        FE = cp.fft.rfft(E, n=self.nfft)
        # 3C PSD
        PSD_phone = (cp.abs(FZ) ** 2 + cp.abs(FN) ** 2 + cp.abs(FE) ** 2) / 3.0
        freqs_full = cp.fft.rfftfreq(self.nfft, d=1/self.fs)
        mask = (freqs_full >= self.freqs[0]) & (freqs_full <= self.freqs[-1])
        PSD_phone_band = PSD_phone[mask]
        ratio = PSD_phone_band / (self.S_ref + 1e-12)
        H_emp = cp.sqrt(cp.abs(ratio))
        self.H_empirical = H_emp
        return H_emp

    def Apply_PINN_Regularization(self, H_pinn, alpha=0.5):
        cp = self.cp
        if self.H_empirical is None:
            raise RuntimeError("Run Estimate Empirical Ratio first")
        # H_pinn = cp.asarray(H_pinn, dtype=cp.float32)
        H_pinn = cp.asarray(H_pinn, dtype=cp.complex64)
        # Match frequency size
        H_pinn = H_pinn[:self.H_empirical.size]
        # Weighted blending
        H_final = alpha * self.H_empirical + (1-alpha) * cp.abs(H_pinn)
        # Smoothness regularization
        grad = cp.gradient(H_final)
        H_final -= 0.01 * grad
        self.H_regularized = cp.maximum(H_final, 1e-6)
        return self.H_regularized

    def Deconvolution(
        self,
        Z,
        N,
        E
    ):
        cp = self.cp
        if self.H_regularized is None:
            raise RuntimeError("Run Apply PINN Regularization first")
        Z = cp.asarray(Z, dtype=cp.float32)
        N = cp.asarray(N, dtype=cp.float32)
        E = cp.asarray(E, dtype=cp.float32)
        FZ = cp.fft.rfft(Z, n=self.nfft)
        FN = cp.fft.rfft(N, n=self.nfft)
        FE = cp.fft.rfft(E, n=self.nfft)
        freqs_full = cp.fft.rfftfreq(self.nfft, d=1 / self.fs)
        H_full = cp.ones_like(freqs_full, dtype=cp.float32)
        mask = (freqs_full >= self.freqs[0]) & (freqs_full <= self.freqs[-1])
        H_full[mask] = self.H_regularized
        H_stable = H_full + self.regularization
        FZ_corr = FZ / H_stable
        FN_corr = FN / H_stable
        FE_corr = FE / H_stable
        return FZ_corr, FN_corr, FE_corr

    # IFFT
    def Reconstruct_3C(
        self,
        FZ_corr,
        FN_corr,
        FE_corr,
        original_length
    ):
        cp = self.cp
        Z_corr = cp.fft.irfft(FZ_corr, n=self.nfft)
        N_corr = cp.fft.irfft(FN_corr, n=self.nfft)
        E_corr = cp.fft.irfft(FE_corr, n=self.nfft)
        # Trim padding
        Z_corr = Z_corr[:original_length]
        N_corr = N_corr[:original_length]
        E_corr = E_corr[:original_length]
        return Z_corr, N_corr, E_corr


"""
class PINN_Empirical_Instrument_Response:

    def __init__(
        self,
        fs,
        nfft=4096,
        fmin=0.1,
        fmax=20.0,
        device=0,
        verbose=False
    ):
        self.fs = fs
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.device = device
        self.verbose = verbose
        # Placeholders
        self.ambient_model = None
        self.pinn_model = None
        self.calibration_engine = None
        self.freqs = None
        self.S_ref_mean = None
        self.S_ref_std = None

    def Run_Training_Pipeline(
        self,
        list_of_doi_3C,
        nperseg=2048,
        overlap=0.5,
        lr=1e-3,
        epochs=2000,
        lambda_smooth=1e-4,
        lambda_physics=1e-4
    ):
        # 1) Build Ambient Statistical Reference
        self.ambient_model = Ambient_Noise_Statistical_Model(
            fs=self.fs,
            nfft=self.nfft,
            fmin=self.fmin,
            fmax=self.fmax,
            nperseg=nperseg,
            overlap=overlap,
            device=self.device,
            verbose=self.verbose
        )
        self.freqs = self.ambient_model.Build_Frequency_Axis()
        self.S_ref_mean, self.S_ref_std = self.ambient_model.Build_Statistical_Reference(list_of_doi_3C)
        if self.verbose:
            print("Ambient statistical reference built.")

        # 2) Train PINN Instrument Model
        self.pinn_model = PINN_Instrument_Model(
            freqs=self.freqs,
            S_reference=self.S_ref_mean,
            lambda_smooth=lambda_smooth,
            lambda_physics=lambda_physics,
            device=self.device,
            verbose=self.verbose
        )
        self.pinn_model.Build_Network()
        self.pinn_model.Train(epochs=epochs)
        if self.verbose:
            print("PINN instrument training complete.")
        return {
            "f0": float(self.pinn_model.f0.get()),
            "zeta": float(self.pinn_model.zeta.get()),
            "loss_history": self.pinn_model.loss_history
        }


    def Run_Calibration_Pipeline(
        self,
        Z,
        N,
        E,
        alpha=0.5,
        regularization=1e-6
    ):
        if self.S_ref_mean is None or self.pinn_model is None:
            raise RuntimeError("Run training pipeline first.")

        # 1) Initialize calibration engine
        self.calibration_engine = Smartphone_Calibration(
            fs=self.fs,
            S_reference=self.S_ref_mean,
            freqs=self.freqs,
            nfft=self.nfft,
            regularization=regularization,
            device=self.device,
            verbose=self.verbose
        )

        # 2) Empirical Estimation
        H_emp = self.calibration_engine.Estimate_Empirical_Ratio(Z, N, E)

        # 3) PINN Transfer Function
        H_pinn = self.pinn_model.Predict_Transfer_Function(return_numpy=False)

        # 4) Regularize Empirical with PINN
        H_reg = self.calibration_engine.Apply_PINN_Regularization(H_pinn, alpha=alpha)

        # 5) Deconvolution
        FZ_corr, FN_corr, FE_corr = self.calibration_engine.Deconvolution(Z, N, E)

        # 6) Reconstruct 3C
        Z_corr, N_corr, E_corr = self.calibration_engine.Reconstruct_3C(
                                    FZ_corr, FN_corr, FE_corr,
                                    original_length=len(Z)
                                 )
        if self.verbose:
            print("Smartphone calibration complete.")

        return {
            "Z_corrected": Z_corr,
            "N_corrected": N_corr,
            "E_corrected": E_corr,
            "H_empirical": H_emp,
            "H_regularized": H_reg
        }
"""


class PINN_Empirical_Instrument_Response:

    def __init__(
        self,
        fs,
        nfft: int = NFFT,
        fmin: float = FMIN,
        fmax: float = FMAX,
        device=0,
        verbose: bool = False,
    ):
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.device = device
        self.verbose = verbose

        self.ambient_model = None
        self.pinn_model = None
        self.calibration_engine = None

        self.freqs = None
        self.S_ref_mean = None
        self.S_ref_std = None

    #-------------------------
    # Utilities (internal)
    #-------------------------
    def Call_Train(
        self,
        model,
        epochs,
        **kwargs
    ):
        candidates = [
            "Train_Model",
            "Train",
            "train_model",
            "train",
            "fit",
        ]
        for name in candidates:
            if hasattr(model, name):
                fn = getattr(model, name)
                try:
                    # Try common signature
                    return fn(epochs=epochs, **kwargs)
                except TypeError:
                    # fallback: call with only epochs or without arguments
                    try:
                        return fn(epochs)
                    except TypeError:
                        try:
                            return fn()
                        except Exception as e:
                            raise RuntimeError(f"Training method '{name}' failed: {e}")
        raise AttributeError("No known train method found on the PINN model")

    def Get_Physical_Parameters(
        self,
        model
    ):
        candidates = [
            "Physical_Pamarameters",
            "PhysicalParameters",
            "physical_parameters",
            "physical_params",
            "physicalParameters",
            "physical_param",
        ]
        for name in candidates:
            if hasattr(model, name):
                try:
                    f0, zeta = getattr(model, name)()
                    return float(f0.item()) if hasattr(f0, "item") else float(f0), float(zeta.item()) if hasattr(zeta, "item") else float(zeta)
                except Exception:
                    out = getattr(model, name)
                    if isinstance(out, tuple) and len(out) >= 2:
                        f0, zeta = out[0], out[1]
                        return float(f0.item()) if hasattr(f0, "item") else float(f0), float(zeta.item()) if hasattr(zeta, "item") else float(zeta)
        if hasattr(model, "log_f0") and hasattr(model, "zeta_param"):
            try:
                f0 = float(torch.exp(model.log_f0).item())
                zeta = float(torch.nn.functional.softplus(model.zeta_param).item())
                return f0, zeta
            except Exception:
                pass
        raise AttributeError("Could not find physical parameters method/attributes on model")

    def Predict_Transfer(
        self,
        model,
        return_numpy=True
    ):
        preds = None
        candidates = ["Predict_Transfer",
                      "Predict_Transfer_Function",
                      "predict_transfer",
                      "predict",
                      "forward",
                      "Forward"]
        for name in candidates:
            if hasattr(model, name):
                fn = getattr(model, name)
                try:
                    preds = fn(return_numpy=return_numpy) if "return_numpy" in fn.__code__.co_varnames else fn()
                    break
                except TypeError:
                    try:
                        preds = fn()
                        break
                    except Exception:
                        continue
        if preds is None:
            try:
                preds = model()
            except Exception:
                raise AttributeError("No suitable prediction method found on PINN model")

        if return_numpy:
            if isinstance(preds, tuple) or (hasattr(preds, "__len__") and len(preds) == 2 and not np.iscomplexobj(preds)):
                Hr, Hi = preds
                if isinstance(Hr, torch.Tensor):
                    Hr = Hr.detach().cpu().numpy().ravel()
                    Hi = Hi.detach().cpu().numpy().ravel()
                return Hr + 1j * Hi
            if isinstance(preds, torch.Tensor):
                if preds.is_complex():
                    return preds.detach().cpu().numpy().ravel()
                else:
                    return preds.detach().cpu().numpy().ravel().astype(np.complex64)
            return np.asarray(preds)
        else:
            return preds

    def to_cupy(self, arr):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(np.asarray(arr))


    def Run_Training_Pipeline(
        self,
        list_of_doi_3C,
        nperseg=NPERSEG,
        overlap=0.5,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        lambda_smooth=LAMBDA_SMOOTH,
        lambda_physics=LAMBDA_PHYSICS,
    ):
        # 1) Build Ambient Statistical Reference
        self.ambient_model = Ambient_Noise_Statistical_Model(
            fs=self.fs,
            nfft=self.nfft,
            fmin=self.fmin,
            fmax=self.fmax,
            nperseg=nperseg,
            overlap=overlap,
            device=self.device,
            verbose=self.verbose,
        )
        self.freqs = self.ambient_model.Build_Frequency_Axis()
        self.S_ref_mean, self.S_ref_std = self.ambient_model.Build_Statistical_Reference(list_of_doi_3C)
        if self.verbose:
            print("Ambient statistical reference built.")

        S_ref_np = to_numpy(self.S_ref_mean)

        # 2) Train PINN Instrument Model
        self.pinn_model = PINN_Instrument_Model(
            freqs=self.freqs,
            S_reference=self.S_ref_mean,
            lambda_smooth=lambda_smooth,
            lambda_physics=lambda_physics,
            device=self.device,
            verbose=self.verbose,
        )

        # if hasattr(self.pinn_model, "Build_Network"):
        #     try:
        #         self.pinn_model.Build_Network()
        #     except Exception as e:
        #         if self.verbose:
        #             print("Warning: Build_Network() exists but raised:", e)

        self.Call_Train(self.pinn_model, epochs=epochs)
        f0_val, zeta_val = self.Get_Physical_Parameters(self.pinn_model)

        loss_history = None
        for name in ("loss_history", "losses", "history"):
            if hasattr(self.pinn_model, name):
                loss_history = getattr(self.pinn_model, name)
                break

        if self.verbose:
            print("PINN instrument training complete.")

        return {"f0": float(f0_val), "zeta": float(zeta_val), "loss_history": loss_history}


    def Run_Calibration_Pipeline(
        self,
        Z,
        N,
        E,
        alpha=0.5,
        regularization=1e-6
    ):
        if self.S_ref_mean is None or self.pinn_model is None:
            raise RuntimeError("Run training pipeline first.")

        # 1) Initialize calibration engine
        S_ref_cp = self.to_cupy(self.S_ref_mean)
        freqs_cp = self.to_cupy(self.freqs)

        self.calibration_engine = Smartphone_Calibration(
            fs=self.fs,
            S_reference=S_ref_cp,
            freqs=self.freqs,
            nfft=self.nfft,
            regularization=regularization,
            device=self.device,
            verbose=self.verbose,
        )

        # 2) Empirical Estimation
        H_emp = self.calibration_engine.Estimate_Empirical_Ratio(Z, N, E)

        # 3) PINN Transfer Function
        H_pinn_np = self.Predict_Transfer(self.pinn_model, return_numpy=True)

        # 4) Regularize Empirical with PINN
        H_pinn_cp = self.to_cupy(H_pinn_np)
        H_reg = self.calibration_engine.Apply_PINN_Regularization(H_pinn_cp, alpha=alpha)

        # 5) Deconvolution
        FZ_corr, FN_corr, FE_corr = self.calibration_engine.Deconvolution(Z, N, E)

        # 6) Reconstruct 3C (returns cupy arrays) -> convert to numpy
        Z_corr_cp, N_corr_cp, E_corr_cp = self.calibration_engine.Reconstruct_3C(FZ_corr, FN_corr, FE_corr, original_length=len(Z))

        Z_corr = to_numpy(Z_corr_cp)
        N_corr = to_numpy(N_corr_cp)
        E_corr = to_numpy(E_corr_cp)
        H_emp_np = to_numpy(H_emp)
        H_reg_np = to_numpy(H_reg)

        if self.verbose:
            print("Smartphone calibration complete.")

        return {
            "Z_corrected": Z_corr,
            "N_corrected": N_corr,
            "E_corrected": E_corr,
            "H_empirical": H_emp_np,
            "H_regularized": H_reg_np,
        }



class Calibration_Plotter:
    
    def __init__(
        self,
        freqs: Optional[Any] = None,
        S_ref_mean: Optional[Any] = None,
        S_ref_std: Optional[Any] = None,
        pinn_H: Optional[Any] = None,
        H_empirical: Optional[Any] = None,
        H_regularized: Optional[Any] = None,
        loss_history: Optional[Any] = None,
        verbose: bool = False
    ):
        self.freqs = to_numpy(freqs)
        self.S_ref_mean = to_numpy(S_ref_mean)
        self.S_ref_std = to_numpy(S_ref_std)
        self.pinn_H = to_numpy(pinn_H)
        self.H_empirical = to_numpy(H_empirical)
        self.H_regularized = to_numpy(H_regularized)
        self.loss_history = to_numpy(loss_history)
        self.verbose = verbose

    #-------------------------
    # Basic plotting helpers
    #-------------------------
    def Magnitude(
        self,
        H
    ):
        H = to_numpy(H)
        if H is None:
            return None
        return np.abs(H)

    def _ensure_freqs(self, n):
        if self.freqs is None:
            return np.arange(n)
        return self.freqs[:n]

    #-------------------------
    # Single-plot functions
    #-------------------------
    def Plot_Reference(
        self,
        ax=None,
        logx: bool=False,
        logy: bool=True,
        title: str="Ambient reference PSD"
    ):
        freqs = self.freqs
        S_mean = self.S_ref_mean
        S_std = self.S_ref_std

        if freqs is None or S_mean is None:
            raise ValueError("freqs and S_ref_mean must be provided to plot reference.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7,4))
        else:
            fig = ax.figure

        ax.plot(freqs, S_mean, label="S_ref_mean", linewidth=1.5)
        if S_std is not None:
            ax.fill_between(freqs, S_mean - S_std, S_mean + S_std, alpha=0.25, label="±1 std")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (arb. units)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", lw=0.5)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        return fig, ax


    def Plot_Transfer_Functions(
        self,
        ax=None,
        freq_log: bool=False,
        title: str="Transfer functions magnitude"
    ):
        H_pin = self.Magnitude(self.pinn_H)
        H_emp = self.Magnitude(self.H_empirical)
        H_reg = self.Magnitude(self.H_regularized)

        n = max(0 if H_pin is None else H_pin.size,
                0 if H_emp is None else H_emp.size,
                0 if H_reg is None else H_reg.size
        )
        if n == 0:
            raise ValueError("No transfer functions available to plot.")
        freqs = self._ensure_freqs(n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8,4.5))
        else:
            fig = ax.figure

        if H_pin is not None:
            ax.plot(freqs[:H_pin.size], H_pin, label="PINN H (|H|)", linewidth=1.5)
        if H_emp is not None:
            ax.plot(freqs[:H_emp.size], H_emp, label="Empirical H (|H_emp|)", linestyle="--", linewidth=1.2)
        if H_reg is not None:
            ax.plot(freqs[:H_reg.size], H_reg, label="Regularized H (|H_reg|)", linestyle="-.", linewidth=1.2)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.5)
        if freq_log:
            ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()
        return fig, ax


    def Plot_Loss_History(
        self,
        ax=None,
        title: str="Training Loss"
    ):
        loss = to_numpy(self.loss_history)
        if loss is None:
            raise ValueError("loss_history not provided.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,3.5))
        else:
            fig = ax.figure
        ax.plot(np.arange(len(loss)), loss, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.4)
        fig.tight_layout()
        return fig, ax


    def Plot_PSD_Comparison(
        self,
        original_psd: Any,
        corrected_psd: Any,
        freqs: Optional[Any]=None,
        ax=None,
        logx: bool=False,
        logy: bool=True,
        title: str="PSD before/after calibration"
    ):
        orig = to_numpy(original_psd)
        corr = to_numpy(corrected_psd)
        if orig is None or corr is None:
            raise ValueError("original_psd and corrected_psd must be provided")
        if freqs is None:
            freqs = self.freqs
        else:
            freqs = to_numpy(freqs)

        if freqs is None:
            freqs = np.arange(orig.size)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7,4))
        else:
            fig = ax.figure

        ax.plot(freqs[:orig.size], orig, label="Original PSD", linewidth=1.2)
        ax.plot(freqs[:corr.size], corr, label="Corrected PSD", linewidth=1.2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (arb. units)")
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.4)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        return fig, ax


    def Plot_Time_Series_Comparison(
        self,
        orig_ts: Any,
        corr_ts: Any,
        fs: float,
        channel_label: str="Z",
        ax=None,
        title: str="Time series: original vs corrected",
        n_samples: int=N_SAMPLES
    ):
        orig = to_numpy(orig_ts)[:n_samples]
        corr = to_numpy(corr_ts)[:n_samples]
        t = np.arange(orig.size) / float(fs)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,3.5))
        else:
            fig = ax.figure
        ax.plot(t, orig, label=f"Original {channel_label}", linewidth=1.0)
        ax.plot(t, corr, label=f"Corrected {channel_label}", linewidth=0.9, alpha=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.4)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    #-------------------------
    # Dashboard (multi-panel)
    #-------------------------
    def Dashboard(
        self,
        figsize: Tuple[float,float]=(12,8),
        freq_log: bool=False
    ):
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[1,1.0])

        # Reference
        ax0 = fig.add_subplot(gs[0,0])
        try:
            self.Plot_Reference(ax=ax0, logx=False, logy=True, title="Ambient reference PSD")
        except Exception as e:
            ax0.text(0.5, 0.5, f"No reference available\n{e}", ha="center")
            ax0.set_axis_off()

        # Transfers
        ax1 = fig.add_subplot(gs[0,1])
        try:
            self.Plot_Transfer_Functions(ax=ax1, freq_log=freq_log, title="Transfer functions")
        except Exception as e:
            ax1.text(0.5, 0.5, f"No transfer functions\n{e}", ha="center")
            ax1.set_axis_off()

        # Loss
        ax2 = fig.add_subplot(gs[1,0])
        try:
            self.Plot_Loss_History(ax=ax2, title="Training loss")
        except Exception as e:
            ax2.text(0.5, 0.5, f"No loss history\n{e}", ha="center")
            ax2.set_axis_off()

        # Empirical vs Regularized or PSD Comparison
        ax3 = fig.add_subplot(gs[1,1])
        H_emp = self.Magnitude(self.H_empirical)
        H_reg = self.Magnitude(self.H_regularized)
        if H_emp is not None or H_reg is not None:
            try:
                freqs = self._ensure_freqs(max(
                    0 if H_emp is None else H_emp.size,
                    0 if H_reg is None else H_reg.size
                ))
                if H_emp is not None:
                    ax3.plot(freqs[:H_emp.size], H_emp, label="Empirical |H|", linewidth=1.1)
                if H_reg is not None:
                    ax3.plot(freqs[:H_reg.size], H_reg, label="Regularized |H|", linewidth=1.1)
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Magnitude")
                ax3.set_title("Empirical vs Regularized")
                ax3.grid(True, ls=":", lw=0.4)
                if freq_log:
                    ax3.set_xscale("log")
                ax3.legend()
            except Exception as e:
                ax3.text(0.5, 0.5, f"Plot error\n{e}", ha="center")
                ax3.set_axis_off()
        else:
            ax3.text(0.5, 0.5, "No empirical/regularized H available", ha="center")
            ax3.set_axis_off()

        fig.suptitle("Calibration Dashboard", fontsize=14)
        return fig

    #-------------------------
    # Utility: save figure
    #-------------------------
    @staticmethod
    def save_figure(fig, filename: str, dpi: int = 720):
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        if isinstance(fig, plt.Figure):
            plt.close(fig)











