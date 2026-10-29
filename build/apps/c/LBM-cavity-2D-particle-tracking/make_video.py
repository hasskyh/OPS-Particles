import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
import os
import sys

def read_ops_grid(filename):
    with open(filename) as f:
        lines = f.readlines()
    nx, ny = 101, 101
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('block Dims'):
            dims = re.findall(r'\[(\d+)\]', line)
            nx, ny = int(dims[0]), int(dims[1])
        if line.startswith('elem size'):
            data_start = i + 1
            break
    values = []
    try:
        for line in lines[data_start:]:
            values.extend(float(x) for x in line.split())
    finally:
        return np.array(values).reshape(ny, nx)

def read_particles(filename):
    with open(filename) as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if 'x' in line and 'y' in line and line.strip().startswith('x'):
            data_start = i + 1
            break
    rows = []
    try:
        for line in lines[data_start:]:
            vals = line.split()
            if vals:
                rows.append([float(v) for v in vals])
    finally:
        return np.array(rows) if rows else np.empty((0, 2))

u_files = sorted(
    glob.glob('u_timestep_*.txt'),
    key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group())
)
if not u_files:
    print("No u_timestep_*.txt files found.")
    sys.exit(1)

timesteps = [re.search(r'\d+', os.path.basename(f)).group() for f in u_files]
print(f"Found {len(timesteps)} timesteps: {timesteps[0]} → {timesteps[-1]}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.suptitle('2D Lid-Driven Cavity — LBM Simulation', fontsize=12)

# Pre-compute speed range across all frames for a consistent colormap
print("Scanning speed range across all frames...")
speed_max = 0.0
for ts in timesteps:
    try:
        u = read_ops_grid(f'u_timestep_{ts}.txt')
        v = read_ops_grid(f'v_timestep_{ts}.txt')
        speed_max = max(speed_max, np.sqrt(u**2 + v**2).max())
    except Exception:
        pass
print(f"  max speed: {speed_max:.6f}")

def make_frame(ts):
    for ax in axes:
        ax.cla()
    u   = read_ops_grid(f'u_timestep_{ts}.txt')
    v   = read_ops_grid(f'v_timestep_{ts}.txt')
    speed = np.sqrt(u**2 + v**2)
    x = np.linspace(0, 1, u.shape[1])
    y = np.linspace(0, 1, u.shape[0])

    im = axes[0].imshow(speed, origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 1], aspect='equal',
                        vmin=0, vmax=speed_max)
    axes[0].streamplot(x, y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
    axes[0].set_title(f'Velocity field  (t = {ts})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    pfile = f'particle_data_{ts}.txt'
    if os.path.exists(pfile):
        pts = read_particles(pfile)
        if pts.size > 0:
            axes[1].scatter(pts[:, 0], pts[:, 1], s=4, c='steelblue', alpha=0.7)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Particle positions  (t = {ts})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    return axes

print("Building animation...")
# Add a persistent colorbar on first frame
make_frame(timesteps[0])
fig.colorbar(
    plt.cm.ScalarMappable(cmap='viridis',
                          norm=plt.Normalize(vmin=0, vmax=speed_max)),
    ax=axes[0], label='Speed |u|'
)
plt.tight_layout()

def update(frame_idx):
    ts = timesteps[frame_idx]
    sys.stdout.write(f'\r  frame {frame_idx+1}/{len(timesteps)}  (t={ts})')
    sys.stdout.flush()
    make_frame(ts)

anim = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=100)

outfile = 'cavity_simulation.gif'
base, ext = os.path.splitext(outfile)
counter = 1
while os.path.exists(outfile):
    outfile = f'{base}_{counter}{ext}'
    counter += 1
print(f"\nSaving {outfile}  ({len(timesteps)} frames) ...")
anim.save(outfile, writer='pillow', fps=10, dpi=100)
print(f"Saved {outfile}")

#For an MP4, use this instead of the previous two lines:
#anim.save('cavity_simulation.mp4', writer='ffmpeg', fps=10, dpi=100)
