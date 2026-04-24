import torch
import shutil

src = r"C:\Users\Admin\TmrlData\weights\SAC_lidar_v4_seed0_t.tmod"
dst = r"C:\Users\Admin\TmrlData\weights\SAC_lidar_v4_seed0_t_straight.tmod"

# Copy the actor weights to new stage name
shutil.copy2(src, dst)
print(f"Done: {dst}")