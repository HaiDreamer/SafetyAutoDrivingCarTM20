import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = Path(r"C:\Users\Admin\TmrlData\SAC_lidar_v4_seed0\train_episodes.csv")
ROW_FROM = 10051     # inclusive, 1-based row number (excluding header)
ROW_TO   = 11077      # inclusive
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

total_rows = len(df)
row_from = max(1, ROW_FROM)
row_to   = min(total_rows, ROW_TO)

df = df.iloc[row_from - 1 : row_to].reset_index(drop=True)

# ── Derived metrics ───────────────────────────────────────────────────────────
df["success"] = df["total_reward"] > 150            # after done bug telemetry "finished bool[8]", fix: df["success"] = df["finished"].astype(bool)

n = len(df)
success_rate   = df["success"].mean() * 100
avg_crash      = df["crash_count"].mean()
avg_reward     = df["total_reward"].mean()
avg_ep_length  = df["steps"].mean()

# ── Print stats ───────────────────────────────────────────────────────────────
print(f"Rows analysed : {row_from} – {row_to}  ({n} episodes)")
print(f"{'─' * 40}")
print(f"Success rate      : {success_rate:.1f}%")
print(f"Crashes/episode   : {avg_crash:.2f}")
print(f"Average reward    : {avg_reward:.4f}")
print(f"Avg episode length: {avg_ep_length:.1f} steps")

# ── Charts ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle(f"Training - complex turns", fontsize=14)

x = df.index + row_from   # x-axis = file row number

def rolling(series, w=20):
    return series.rolling(w, min_periods=1).mean()

# 1. Success rate (rolling %)
ax = axes[0, 0]
roll_success = rolling(df["success"].astype(float) * 100)
ax.plot(x, df["success"].astype(float) * 100, color="#888780", linewidth=0.6, alpha=0.5, label="Raw")  # ← add raw
ax.plot(x, roll_success, color="#378ADD", linewidth=1.5, label="Rolling avg")
ax.set_title("Success rate")
ax.set_ylabel("%")
ax.legend(fontsize=8, loc="upper left")                                                                  # ← add legend
ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# 2. Crash count per episode
ax = axes[0, 1]
ax.plot(x, df["crash_count"], color="#888780", linewidth=0.6, alpha=0.5, label="Raw")
ax.plot(x, rolling(df["crash_count"]), color="#D85A30", linewidth=1.5, label="Rolling avg")
ax.set_title("Crash count per episode")
ax.set_ylabel("Crashes")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 1))

# 3. Average reward
ax = axes[1, 0]
ax.plot(x, df["total_reward"], color="#888780", linewidth=0.6, alpha=0.5, label="Raw")
ax.plot(x, rolling(df["total_reward"]), color="#378ADD", linewidth=1.5, label="Rolling avg")
ax.set_title("Reward per episode")
ax.set_ylabel("Reward")
ax.legend(fontsize=8, loc="upper left") 

# 4. Episode length (steps)
ax = axes[1, 1]
ax.plot(x, df["steps"], color="#888780", linewidth=0.6, alpha=0.5, label="Raw")
ax.plot(x, rolling(df["steps"]), color="#1D9E75", linewidth=1.5, label="Rolling avg")
ax.set_title("Episode length (steps)")
ax.set_ylabel("Steps")
ax.legend(fontsize=8, loc="upper left") 

for ax in axes.flat:
    ax.set_xlabel("Episode")
    ax.grid(True, linewidth=0.4, alpha=0.4)

plt.tight_layout()
plt.savefig("D:\Internship-AutoDrivingCar\mermaid_code\episodes_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved → episodes_analysis.png")