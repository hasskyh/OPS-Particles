import glob, os

for f in glob.glob("particle_data_*.txt") + glob.glob("rho_timestep_*.txt") + glob.glob("u_timestep_*.txt") + glob.glob("v_timestep_*.txt"):
    os.remove(f)
    print(f"Deleted {f}")
