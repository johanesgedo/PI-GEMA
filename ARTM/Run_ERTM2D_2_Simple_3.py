
import os
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ARTM2D import Acoustic_Reverse_Time_Migration_2D


def make_simple_elastic_model(
    nx=301,
    nz=151,
    dx=10.0,
    dz=10.0,
    seed=42
):
    np.random.seed(seed)
    z = np.arange(nz, dtype=np.float32) * dz
    vp_base = 1800.0 + 0.8 * z[None, :]
    vs_base = 1000.0 + 0.5 * z[None, :]
    rho_base = 1900.0 + 0.3 * z[None, :]

    vp = np.tile(vp_base, (nx, 1))
    vs = np.tile(vs_base, (nx, 1))
    rho = np.tile(rho_base, (nx, 1))

    z1 = int(0.25 * nz)
    z2 = int(0.50 * nz)
    z3 = int(0.75 * nz)

    vp[:, z1:z2] += 300.0
    vs[:, z1:z2] += 150.0
    rho[:, z1:z2] += 120.0

    vp[:, z2:z3] += 700.0
    vs[:, z2:z3] += 350.0
    rho[:, z2:z3] += 200.0

    vp[:, z3:] += 1200.0
    vs[:, z3:] += 600.0
    rho[:, z3:] += 300.0

    for ix in range(nx):
        shift = int(10 * np.sin(2 * np.pi * ix / nx))
        z_dip = z2 + shift
        z_dip = max(0, min(nz - 1, z_dip))
        vp[ix, z_dip:] += 300
        vs[ix, z_dip:] += 150
        rho[ix, z_dip:] += 100

    ixg, izg = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    cx1, cz1 = int(0.35 * nx), int(0.55 * nz)
    r1 = min(nx, nz) // 10
    mask1 = (ixg - cx1) ** 2 + (izg - cz1) ** 2 < r1 ** 2

    vp[mask1] += 1000.0
    vs[mask1] += 500.0
    rho[mask1] += 250.0

    cx2, cz2 = int(0.65 * nx), int(0.40 * nz)
    r2 = min(nx, nz) // 12
    mask2 = (ixg - cx2) ** 2 + (izg - cz2) ** 2 < r2 ** 2

    vp[mask2] -= 600.0
    vs[mask2] -= 300.0
    rho[mask2] -= 150.0

    noise = np.random.normal(0.0, 50.0, size=(nx, nz)).astype(np.float32)
    vp += noise
    vs += 0.5 * noise
    rho += 0.3 * noise

    vp = np.clip(vp, 1500.0, 5000.0)
    vs = np.clip(vs, 800.0, 3000.0)
    rho = np.clip(rho, 1800.0, 2600.0)

    vs = np.minimum(vs, vp * 0.58)

    return vp.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32)


#---------------------------
# 1) Parameters
#---------------------------
nx, nz = 301, 151
dx = dz = 10.0
seed = 42

lxpad = 80
rxpad = 80

checkpoint_dir = "./snapshots"
os.makedirs(checkpoint_dir, exist_ok=True)

snap_stride = 4
snap_dtype = "float16"
max_gpu_snap_bytes = 64 * 1024 * 1024
src_depth = 5
rec_depth = 0
freq = 25.0

#----------------------------------
# 2) Create Model (vp, vs, rho)
#----------------------------------
vp, vs, rho = make_simple_elastic_model(nx=nx, nz=nz, dx=dx, dz=dz, seed=seed)

#---------------------------
# 3) Init migrator (ERTM)
#---------------------------
artm = Acoustic_Reverse_Time_Migration_2D(
    device=0,
    verbose=True,
    enable_stream=True,
    register_kernels=True
)

# Enable profiler if available
if hasattr(artm, "profiler") and artm.profiler is not None:
    try:
        artm.profiler.Enable()
    except Exception:
        pass

#---------------------------
# 4) Forward modelling
#---------------------------
print("[MAIN] Running forward modelling to create synthetic data (pdata)...")
image_dummy, pdata = artm.run(
    vp=vp,
    dx=dx,
    dz=dz,
    lxpad=lxpad,
    rxpad=rxpad,
    dt=None,
    snap_stride=snap_stride,
    imaging=False,
    checkpoint_dir=checkpoint_dir,
    src_depth=src_depth,
    rec_depth=rec_depth,
    vs=vs,
    rho=rho,
    do_forward=True,
    do_backward=False,
    return_pdata=True,
    snap_dtype=snap_dtype,
    max_gpu_snap_bytes=max_gpu_snap_bytes
)
print("[MAIN] Synthetic pdata shape:", getattr(pdata, "shape", None))

#---------------------------
# 5) Running RTM
#---------------------------
print("[MAIN] Running RTM imaging (backward stack) using synthesized pdata...")
image = artm.run(
    vp=vp,
    pdata=pdata,
    dx=dx,
    dz=dz,
    lxpad=lxpad,
    rxpad=rxpad,
    dt=None,
    snap_stride=snap_stride,
    imaging=True,
    src_depth=src_depth,
    rec_depth=rec_depth,
    vs=vs,
    rho=rho,
    do_forward=True,
    do_backward=True,
    return_pdata=True,
    snap_dtype=snap_dtype,
    max_gpu_snap_bytes=max_gpu_snap_bytes
)

print("[MAIN] RTM finished. Image shape:", getattr(image, "shape", None))

#------------------------
# 6) Postprocessing
#------------------------
out_image_path = "RTM_image.npy"
np.save(out_image_path, image)
print("[MAIN] Saved RTM image to", out_image_path)

# Normalize for plotting
img = image.copy()
img = img / (np.max(np.abs(img)) + 1e-12)

plt.figure(figsize=(10,6))
extent = [0.0, (img.shape[0] * dx) / 1000.0, (img.shape[1] * dz) / 1000.0, 0.0]  # km
plt.imshow(img.T, cmap="seismic", aspect="auto", origin="upper", extent=extent)
plt.colorbar(label="Normalized Amplitude")
plt.title("Acoustic RTM Image")
plt.xlabel("Distance (km)")
plt.ylabel("Depth (km)")
plt.tight_layout()
png_path = "rtm_image.png"
plt.savefig(png_path, dpi=1048)
plt.close()
print("[MAIN] Saved RTM image plot to", png_path)

#---------------------------
# 7) Profiler report
#---------------------------
if hasattr(artm, "profiler") and artm.profiler is not None:
    try:
        agg = artm.profiler.Report(show_plots=True, save_prefix="ERTM2D_Profile")
        print("[MAIN] Profiler aggregated metrics:", agg.keys())
    except Exception as e:
        print("[MAIN] Profiler report error:", e)

print("[MAIN] DONE")













