"""
SAC Training Log Analyzer for TMRL / TrackMania 20 Lidar
Usage: python analyze_sac_log.py
       or: python analyze_sac_log.py path/to/your/sac_loss_log.csv
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = r"D:\Internship-AutoDrivingCar\sac_loss_log.csv"
SMOOTH_WINDOW = 50          # rolling average window (steps)
SPIKE_STD_MULT = 3.0        # flag values beyond mean ± N*std as spikes
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"✅ Loaded {len(df):,} rows from: {path}")
    print(f"   Columns: {list(df.columns)}\n")
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def detect_spikes(series: pd.Series, multiplier: float = SPIKE_STD_MULT):
    mean, std = series.mean(), series.std()
    return series[(series > mean + multiplier * std) | (series < mean - multiplier * std)]


def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    total_steps = len(df)
    print(f"  Total steps logged : {total_steps:,}")

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
            print(f"  Duration           : {duration}")
        except Exception:
            pass

    metrics = {
        "loss_critic":      "Critic Loss",
        "loss_actor":       "Actor Loss",
        "entropy":          "Entropy",
        "q_gap":            "Q-Gap",
        "backup_mean":      "Backup Mean (Reward signal)",
        "entropy_coef":     "Entropy Coef (α)",
    }

    print()
    for col, label in metrics.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        first_q = s.iloc[:max(1, len(s)//4)].mean()
        last_q  = s.iloc[-max(1, len(s)//4):].mean()
        trend   = "↑ improving" if last_q > first_q else "↓ declining"
        # flip trend label for losses (lower = better)
        if "loss" in col:
            trend = "↓ improving" if last_q < first_q else "↑ worsening"
        spikes = detect_spikes(s)
        print(f"  {label}")
        print(f"    Range  : [{s.min():.4f}, {s.max():.4f}]")
        print(f"    Mean   : {s.mean():.4f}  |  Std: {s.std():.4f}")
        print(f"    Trend  : {trend}  (first-quarter avg {first_q:.4f} → last-quarter avg {last_q:.4f})")
        if len(spikes) > 0:
            print(f"    ⚠️  Spikes detected: {len(spikes)} points (steps: {list(spikes.index[:5])}{'...' if len(spikes)>5 else ''})")
        print()

    # Convergence heuristic
    print("=" * 60)
    print("  CONVERGENCE DIAGNOSIS")
    print("=" * 60)
    issues = []

    if "loss_critic" in df.columns:
        critic = df["loss_critic"].dropna()
        late_critic = critic.iloc[-max(1, len(critic)//5):].mean()
        if late_critic > 1.0:
            issues.append("❌ Critic loss is still high in late training (>1.0) — may not have converged.")
        else:
            print("  ✅ Critic loss looks stable in late training.")

    if "backup_mean" in df.columns:
        bm = df["backup_mean"].dropna()
        if len(bm) > 10:
            first_half = bm.iloc[:len(bm)//2].mean()
            second_half = bm.iloc[len(bm)//2:].mean()
            if second_half > first_half:
                print("  ✅ backup_mean is trending upward — agent is improving.")
            else:
                issues.append("⚠️  backup_mean is not increasing — reward signal may be stagnating.")

    if "entropy" in df.columns:
        ent = df["entropy"].dropna()
        if len(ent) > 10:
            late_ent = ent.iloc[-max(1, len(ent)//5):].mean()
            early_ent = ent.iloc[:max(1, len(ent)//5)].mean()
            if late_ent < early_ent * 0.9:
                print("  ✅ Entropy is decreasing — policy is sharpening over time.")
            else:
                issues.append("⚠️  Entropy is not decreasing — agent may still be exploring or stuck.")

    if "q_gap" in df.columns:
        qg = df["q_gap"].dropna()
        if qg.max() > 1.0:
            issues.append(f"⚠️  q_gap peaked at {qg.max():.3f} — possible Q-value overestimation.")
        else:
            print("  ✅ q_gap is small — no obvious Q-value divergence.")

    if issues:
        print()
        for issue in issues:
            print(f"  {issue}")
    print()


def plot_dashboard(df: pd.DataFrame, save_path: str = None):
    x = df["step"] if "step" in df.columns else df.index

    fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
    fig.suptitle("SAC Training Dashboard — TMRL TrackMania Lidar",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_configs = [
        # (col,              title,                   color,     good_direction)
        ("loss_critic",      "Critic Loss",            "#f97316", "down"),
        ("loss_actor",       "Actor Loss",             "#a78bfa", "up"),
        ("entropy",          "Entropy",                "#38bdf8", "down"),
        ("q_gap",            "Q-Gap (Q1 vs Q2)",       "#facc15", "down"),
        ("backup_mean",      "Backup Mean",            "#4ade80", "up"),
        ("backup_min",       "Backup Min",             "#fb7185", None),
        ("q1_mean",          "Q1 Mean",                "#67e8f9", None),
        ("q2_mean",          "Q2 Mean",                "#c084fc", None),
        ("entropy_coef",     "Entropy Coef (α)",       "#fbbf24", None),
    ]

    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

    for ax, (col, title, color, direction) in zip(axes, plot_configs):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#9ca3af", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3148")

        if col not in df.columns or df[col].dropna().empty:
            ax.set_title(f"{title}\n(no data)", color="#6b7280", fontsize=9)
            continue

        s = df[col].dropna()
        s_x = x.loc[s.index]

        # Raw (faint)
        ax.plot(s_x, s, color=color, alpha=0.2, linewidth=0.8)
        # Smoothed
        s_smooth = smooth(s, SMOOTH_WINDOW)
        ax.plot(s_x, s_smooth, color=color, linewidth=1.8, label="smoothed")

        # Spike markers
        spikes = detect_spikes(s)
        if not spikes.empty:
            ax.scatter(x.loc[spikes.index], spikes, color="#ef4444",
                       s=12, zorder=5, label=f"spikes ({len(spikes)})")
            ax.legend(fontsize=6, facecolor="#1a1d27", labelcolor="white", framealpha=0.6)

        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("step", color="#6b7280", fontsize=7)
        ax.grid(True, color="#2d3148", linewidth=0.5, linestyle="--")

    out = save_path or "sac_training_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"📊 Dashboard saved → {out}")
    plt.show()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("   Usage: python analyze_sac_log.py [path/to/csv]")
        sys.exit(1)

    df = load_csv(csv_path)
    print_summary(df)

    out_img = os.path.join(os.path.dirname(csv_path), "sac_training_dashboard.png")
    plot_dashboard(df, save_path=out_img)


if __name__ == "__main__":
    main()