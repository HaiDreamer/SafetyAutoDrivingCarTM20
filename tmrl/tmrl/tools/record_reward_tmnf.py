"""
Record a TMNF reward trajectory by driving the track manually.
Usage:
    python record_reward_tmnf.py

Drive from START to FINISH, then press Ctrl+C to stop and save.
Saves to ~/TmrlData/reward/reward.pkl  (same path RewardFunction expects)
"""

import pickle
import time
import numpy as np
import pathlib
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.tmnf_interfaces import TMNFClient, TMI_PORT


REWARD_PATH = pathlib.Path(cfg.REWARD_PATH)
SAMPLE_INTERVAL = 0.01  # small interval to capture many raw positions

# add this function directly in record_reward_tmnf.py:
def line(pt1, pt2, dist):
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, dist - norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0

def record():
    print("[Record] Connecting to TMInterface...")
    client = TMNFClient(port=TMI_PORT, game_speed=1.0)
    print("[Record] Connected. Drive the track from START to FINISH.")
    print("[Record] Press Ctrl+C when you cross the finish line to save.")

    positions = []
    prev_time = -1

    try:
        while True:
            race_time, cp, cp_target, finished = client.get_state()

            if race_time > 0 and race_time != prev_time:
                x, y, z = client.position
                positions.append([x, y, z])
                prev_time = race_time
                print(f"\r[Record] Positions: {len(positions)}  CP: {cp}/{cp_target}", end="")

            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Record] Stopped.")

    if len(positions) < 10:
        print("[Record] Too few positions recorded, aborting.")
        return

    # ── resample to equally spaced points (same as original record_reward_dist) ──
    positions = np.array(positions)
    final_positions = [positions[0]]
    dist_between_points = 0.1
    j, move_by, pt1 = 1, dist_between_points, final_positions[-1]

    while j < len(positions):
        pt2 = positions[j]
        pt, dst = line(pt1, pt2, move_by)
        if pt is not None:
            final_positions.append(pt)
            move_by = dist_between_points
            pt1 = pt
        else:
            pt1 = pt2
            j += 1
            move_by = dst

    final_positions = np.array(final_positions)
    print(f"[Record] Raw positions: {len(positions)} → resampled: {len(final_positions)} waypoints")

    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(final_positions, open(REWARD_PATH, "wb"))
    print(f"[Record] Saved to {REWARD_PATH}")


if __name__ == "__main__":
    record()