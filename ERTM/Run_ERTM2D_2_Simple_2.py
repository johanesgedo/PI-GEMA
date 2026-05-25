
import os
import numpy as np
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ERTM2D_2 import Elastic_Reverse_Time_Migration_2D
from ERTM2D_2_Profiler import Elastic_Reverse_Time_Migration_2D_Profiler

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


def _to_numpy(arr):
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr).astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


def load_snapshots_from_checkpoint(
    checkpoint_dir,
    shot=0,
    varname="vx",
    n_snapshots=None,
    nx=None,
    nz=None,
    dtype=np.float32
):
    fname = os.path.join(checkpoint_dir, f"shot_{shot:04d}_snaps_{varname}.dat")
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    # Infer n_snapshots if not provided
    if n_snapshots is None:
        if nx is None or nz is None:
            raise ValueError("nx and nz required to infer n_snapshots from file size")
        filesize = os.path.getsize(fname)
        bytes_per = np.dtype(dtype).itemsize
        total_elems = filesize // bytes_per
        if total_elems % (nx * nz) != 0:
            raise ValueError(f"File size {filesize} not compatible with nx*nx*nz ({nx}x{nz}) and dtype {dtype}")
        n_snapshots = total_elems // (nx * nz)

    shape = (int(n_snapshots), int(nx), int(nz))
    arr = np.memmap(fname, dtype=dtype, mode="r", shape=shape)
    return np.asarray(arr).astype(np.float32)


def plot_wavefield_snapshots(
    snaps_vx,
    snaps_vz,
    dx,
    dz,
    outdir="waveframes",
    cmap="seismic",
    normalize=True,
    combine="divergence",
    vmin_vmax=None,
    extent_in_km=True,
    save_mp4=True,
    fps=10,
    dpi=200
):
    os.makedirs(outdir, exist_ok=True)

    snaps_vx_np = _to_numpy(snaps_vx)
    snaps_vz_np = _to_numpy(snaps_vz)

    if snaps_vx_np.ndim != 3 or snaps_vz_np.ndim != 3:
        raise ValueError("snap arrays must be 3D: (nframes, nx, nz)")

    nframes, nx, nz = snaps_vx_np.shape
    if snaps_vz_np.shape != (nframes, nx, nz):
        raise ValueError("snaps_vx and snaps_vz must have the same shape")

    frames = np.empty((nframes, nz, nx), dtype=np.float32)

    for i in range(nframes):
        vx = snaps_vx_np[i]
        vz = snaps_vz_np[i]

        if combine == "divergence":
            dvx_dx = np.gradient(vx, dx, axis=0)
            dvz_dz = np.gradient(vz, dz, axis=1)
            field = dvx_dx + dvz_dz
        elif combine == "amplitude":
            field = np.sqrt(vx * vx + vz * vz)
        elif combine == "vx":
            field = vx
        elif combine == "vz":
            field = vz
        else:
            raise ValueError("combine must be 'divergence', 'amplitude', 'vx', or 'vz'")

        frames[i] = field.T.astype(np.float32)

    if vmin_vmax is None:
        absmax = np.nanmax(np.abs(frames)) + 1e-12
        if combine in ("divergence", "vx", "vz"):
            vmin, vmax = -absmax, absmax
        else:
            vmin, vmax = 0.0, absmax
    else:
        vmin, vmax = vmin_vmax

    if extent_in_km:
        extent = [0.0, (nx * dx) / 1000.0, (nz * dz) / 1000.0, 0.0]  # xmin,xmax,ymin,ymax
        xlabel = "Distance (km)"
        ylabel = "Depth (km)"
    else:
        extent = [0, nx, nz, 0]
        xlabel = "Distance (grid)"
        ylabel = "Depth (grid)"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = ax.set_title("Frame = 0", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Amplitude")

    for i in range(nframes):
        im.set_data(frames[i])
        title.set_text(f"Frame = {i}")
        outpath = os.path.join(outdir, f"frame_{i:04d}.png")
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    print(f"[plot_wavefield_snapshots] Saved {nframes} PNG frames to {outdir}")

    if save_mp4:
        try:
            try:
                FFMpegWriter = animation.writers['ffmpeg']
                writer = FFMpegWriter(fps=fps, metadata=dict(artist='rtm'), bitrate=4000)
            except Exception:
                # fallback to writer class
                writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='rtm'), bitrate=4000)

            def update(k):
                im.set_data(frames[k])
                title.set_text(f"Frame = {k}")
                return (im,)

            anim = animation.FuncAnimation(fig, update, frames=nframes, interval=1000 / fps)
            mp4_path = os.path.join(outdir, "wavefield.mp4")
            anim.save(mp4_path, writer=writer, dpi=dpi)
            print(f"[plot_wavefield_snapshots] Saved MP4 -> {mp4_path}")
        except Exception as e:
            print("[plot_wavefield_snapshots] Could not write MP4 (ffmpeg missing?). Error:", e)
            try:
                def update2(k):
                    im.set_data(frames[k])
                    title.set_text(f"Frame = {k}")
                    return (im,)

                anim = animation.FuncAnimation(fig, update2, frames=nframes, interval=1000 / fps)
                gif_path = os.path.join(outdir, "wavefield.gif")
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"[plot_wavefield_snapshots] Saved GIF -> {gif_path}")
            except Exception as e2:
                print("[plot_wavefield_snapshots] Could not save GIF either. Error:", e2)

    plt.close(fig)
    return outdir


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


#-------------------------------------------
# Example usage (with consistent padding)
#-------------------------------------------
nx, nz = 301, 151
dx = dz = 10.0

vp, vs, rho = make_simple_elastic_model(nx, nz, dx, dz, seed=42)

ertm = Elastic_Reverse_Time_Migration_2D(
    device=0,
    verbose=True,
    enable_stream=True
)

# Use same paddings when reading snapshots that were used when creating them
lxpad = 80
rxpad = 80
nx_local = nx + lxpad + rxpad
nz_local = nz

os.makedirs("./snapshots", exist_ok=True)
checkpoint_dir = "./snapshots"

image_dummy, pdata = ertm.run(
    vp=vp,
    dx=dx,
    dz=dz,
    lxpad=lxpad,
    rxpad=rxpad,
    dt=None,
    snap_stride=4,
    imaging=True,
    checkpoint_dir=checkpoint_dir,
    src_depth=5,
    rec_depth=0,
    vs=vs,
    rho=rho,
    do_forward=True,
    do_backward=True,
    return_pdata=True,
    snap_dtype="float16",
    max_gpu_snap_bytes=64 * 1024 * 1024
)

print("[TEST] Synthetic data generated:", getattr(pdata, "shape", None))

if hasattr(ertm, "profiler") and ertm.profiler is not None:
    ertm.profiler.Enable()

nt_est = pdata.shape[2]
n_snapshots = nt_est // 4

snaps_vx = load_snapshots_from_checkpoint(
    checkpoint_dir=checkpoint_dir,
    shot=0,
    varname="vx",
    n_snapshots=n_snapshots,
    nx=nx_local,
    nz=nz_local,
    dtype=np.float16
)

snaps_vz = load_snapshots_from_checkpoint(
    checkpoint_dir=checkpoint_dir,
    shot=0,
    varname="vz",
    n_snapshots=n_snapshots,
    nx=nx_local,
    nz=nz_local,
    dtype=np.float16
)

plot_wavefield_snapshots(
    snaps_vx,
    snaps_vz,
    dx=dx,
    dz=dz,
    outdir="wavefield_frames",
    combine="divergence",
    save_mp4=True,
    fps=10
)

image = ertm.run(
    vp=vp,
    pdata=pdata,
    dx=dx,
    dz=dz,
    lxpad=lxpad,
    rxpad=rxpad,
    dt=None,
    snap_stride=4,
    imaging=True,
    src_depth=5,
    rec_depth=0,
    vs=vs,
    rho=rho,
    do_forward=True,
    do_backward=True,
    snap_dtype="float16",
    max_gpu_snap_bytes=64 * 1024 * 1024
)

print("[TEST] RTM finished. Image shape =", getattr(image, "shape", None))

if hasattr(ertm, "profiler") and ertm.profiler is not None:
    agg = ertm.profiler.report(show_plots=True, save_prefix="ERTM2D_Profile")

# plot image (example)
plt.figure(figsize=(10, 6))
plt.imshow(image.T, cmap="seismic", aspect="auto", origin="upper")
plt.colorbar(label="Normalized Amplitude")
plt.title("Elastic RTM Image")
plt.xlabel("Distance (km)")
plt.ylabel("Depth (km)")
plt.tight_layout()
plt.show()


#----------------------------------------
# Plot Velocity Model
#----------------------------------------
plt.figure(figsize=(10, 6))

plt.imshow(
    vp.T,
    cmap="jet",
    aspect="auto",
    origin="upper"
)

plt.colorbar(label="Vp (m/s)")
plt.title("True Velocity Model")
plt.xlabel("Distance (km)")
plt.ylabel("Depth (km)")

plt.tight_layout()
plt.show()


#----------------------------------------
# Plot Synthetic Shot Gather
#----------------------------------------
shot_id = 0

plt.figure(figsize=(10, 6))

plt.imshow(
    pdata[shot_id].T,
    cmap="gray",
    aspect="auto",
    origin="upper"
)

plt.colorbar(label="Amplitude")
plt.title(f"Synthetic Shot Gather (Shot {shot_id})")
plt.xlabel("Receiver index")
plt.ylabel("Time index")

plt.tight_layout()
plt.show()




if __name__ == "__main__":
    import types
    try:
        from ERTM2D_2 import Elastic_Reverse_Time_Migration_2D
        owner = Elastic_Reverse_Time_Migration_2D(device=0, verbose=False)
        print("[Profiler] Using real ERTM2D engine")
    except Exception:
        owner = types.SimpleNamespace(
            has_cupy=False,
            device=0,
            DX=2,
            DZ=2,
            SIGMA=1.5,
            TRUNCATE=3.0,
            verbose=False
        )
        print("[Profiler] Using dummy owner")

    prof = Elastic_Reverse_Time_Migration_2D_Profiler(owner=owner, enable=True, verbose=True)

    try:
        prof.wrap_methods(owner, method_names=prof.targets)
    except Exception as e:
        print("[Profiler] wrap failed:", e)

    demo_grid_list = [(512, 512), (576, 576), (624, 624)]
    demo_tiles = [(16, 16), (32, 8)]

    results = prof.benchmark_all_and_plot(
        targets=None,
        grid_list=demo_grid_list,
        tiles=demo_tiles,
        repeats=4,
        warmups=1,
        save_prefix="benchmark_image_gaussian",
        show_combined=True,
        per_target_plots=True
    )

    print("Benchmark finished. Results keys:", list(results.keys()))



