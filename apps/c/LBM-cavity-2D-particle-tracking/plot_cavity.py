import numpy as np
import matplotlib.pyplot as plt
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
    for line in lines[data_start:]:
        values.extend(float(x) for x in line.split())
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
    for line in lines[data_start:]:
        vals = line.split()
        if vals:
            rows.append([float(v) for v in vals])
    return np.array(rows) if rows else np.empty((0, 2))

# Data files are written to the repo root; resolve relative to this script's location
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')

# Pick timestep: last available, or pass one as argument
u_files = sorted(glob.glob(os.path.join(DATA_DIR, 'u_timestep_*.txt')))
if not u_files:
    print("No output files found — run the simulation first, then run this script from the same directory.")
    sys.exit(1)

if len(sys.argv) > 1:
    timestep = sys.argv[1]
else:
    timestep = os.path.basename(u_files[-1]).replace('u_timestep_', '').replace('.txt', '')

print(f"Plotting timestep {timestep}")

u   = read_ops_grid(os.path.join(DATA_DIR, f'u_timestep_{timestep}.txt'))
v   = read_ops_grid(os.path.join(DATA_DIR, f'v_timestep_{timestep}.txt'))
rho = read_ops_grid(os.path.join(DATA_DIR, f'rho_timestep_{timestep}.txt'))

speed = np.sqrt(u**2 + v**2)
x = np.linspace(0, 1, u.shape[1])
y = np.linspace(0, 1, u.shape[0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: velocity field ---
im = axes[0].imshow(speed, origin='lower', cmap='viridis',
                    extent=[0, 1, 0, 1], aspect='equal')
plt.colorbar(im, ax=axes[0], label='Speed |u|')
axes[0].streamplot(x, y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
axes[0].set_title(f'Velocity field  (t = {timestep})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# --- Right: particle positions ---
pfile = os.path.join(DATA_DIR, f'particle_data_{timestep}.txt')
if os.path.exists(pfile):
    pts = read_particles(pfile)
    if pts.size > 0:
        axes[1].scatter(pts[:, 0], pts[:, 1], s=4, c='steelblue', alpha=0.7)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Particle positions  (t = {timestep})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
else:
    axes[1].text(0.5, 0.5, f'{pfile}\nnot found', ha='center', va='center',
                 transform=axes[1].transAxes)

plt.suptitle('2D Lid-Driven Cavity — LBM Simulation', fontsize=12)
plt.tight_layout()

outfile = f'cavity_t{timestep}.png'
plt.savefig(outfile, dpi=150)
print(f"Saved {outfile}")
plt.show()
