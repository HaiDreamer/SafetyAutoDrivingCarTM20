"""
SAC Training Log Analyzer for TMRL / TrackMania 20 Lidar
    every run pls change name of sac_training_dashboard.png, otherwise it is overwritten 

    
EXPLAIN
Rolling mean smoothing: a sliding window average — for each point, replace its value with the mean of the surrounding N points.
    deal with noisy RL curves, avoid NaN at start
Spike detection: flags any point that sits more than N standard deviations away from the series mean, spike if:x > mean + N·std or x < mean - N·std
    the spikes most want to catch — sharp critic loss explosions that signal instability 
Trend - First-quarter vs last-quarter trend heuristic: splits the series into its first 25% and last 25%, averages each chunk, then compares the two averages to decide if the metric is improving or worsening.
        backup_mean: split into two equal halves by index, computes the average of each half, then compares them
Critic loss convergence threshold: the mean of the final 20% of training and warns if it exceeds 1.0


CITATION
TD Backup / Bellman Target: backup = r + γ · (min(Q1', Q2') - α · log π)
    Fujimoto et al., ICML 2018, pp. 1582-1591
The entropy term bonus −α · log π
    Haarnoja et al., ICML 2018, pp. 1861-1870
Auto-α tuning
    Haarnoja et al., ICML 2019 (SAC v2)
Q-gap diagnostic
    Fujimoto et al., ICML 2018, pp. 1582-1591
Rolling Mean Smoothing series.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    Hyndman, R.J. (2009). Moving Averages. Published technical report, Monash University.
Spike Detection (z-score / standard deviation threshold)
    Grubbs, F.E. (1969). Procedures for Detecting Outlying Observations in Samples. Technometrics, 11(1), 1-21.
    Iglewicz, B., & Hoaglin, D.C. (1993). How to Detect and Handle Outliers. ASQ Quality Press. (ASQC Basic References in Quality Control, Vol. 16.)
Trend Heuristic (first-quarter vs last-quarter avg)
    Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press. — (Section on monitoring training curves and convergence)


"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = r"D:\Internship-AutoDrivingCar\sac_loss_log_straight_2.csv"
SMOOTH_WINDOW = 50          # rolling average window (steps)
# After
SPIKE_STD_MULT = {      # flag values beyond mean ± N*std as spikes
    "loss_critic":  2.5,
    "loss_actor":   3.0,
    "entropy":      2.5,
    "q_gap":        2.0,
    "backup_mean":  4.0,
    "default":      3.0,
}
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    # Count max columns across all lines to handle mid-run schema changes
    with open(path, "r") as f:
        lines = f.readlines()

    header_cols = lines[0].strip().split(",")
    max_cols = max(len(l.strip().split(",")) for l in lines[1:] if l.strip())

    # Pad header with generic names for any extra columns added mid-run
    extra = max_cols - len(header_cols)
    if extra > 0:
        print(f"CSV has {extra} extra column(s) added mid-run — padding header as col_extra_0 ... col_extra_{extra-1}")
        header_cols += [f"col_extra_{i}" for i in range(extra)]

    df = pd.read_csv(path, names=header_cols, skiprows=1, on_bad_lines="warn")
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df):,} rows from: {path}")
    print(f"Columns: {list(df.columns)}\n")
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def detect_spikes(series: pd.Series, multiplier: float = 3.0):
    mean, std = series.mean(), series.std()
    return series[(series > mean + multiplier * std) | (series < mean - multiplier * std)]


def print_summary(df: pd.DataFrame):
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    total_steps = len(df)
    print(f"Total steps logged : {total_steps:,}")

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
        "entropy_coef":     "Entropy Coef (alpha)",
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
        spikes = detect_spikes(s, SPIKE_STD_MULT.get(col, SPIKE_STD_MULT["default"]))
        print(f"{label}")
        print(f"Range  : [{s.min():.4f}, {s.max():.4f}]")
        print(f"Mean   : {s.mean():.4f}  |  Std: {s.std():.4f}")
        print(f"Trend  : {trend}  (first-quarter avg {first_q:.4f} → last-quarter avg {last_q:.4f})")
        if len(spikes) > 0:
            print(f"Spikes detected: {len(spikes)} points (steps: {list(spikes.index[:5])}{'...' if len(spikes)>5 else ''})")
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
            issues.append("Critic loss is still high in late training (>1.0) — may not have converged.")
        else:
            print("Critic loss looks stable in late training.")
        mid = critic.iloc[len(critic)//4 : -len(critic)//4]
        if len(mid) > 10 and mid.std() > critic.std() * 2.0:
            issues.append(f"High volatility in mid-run critic loss (mid std {mid.std():.4f} vs overall {critic.std():.4f}) — possible instability event.")

    if "backup_mean" in df.columns:
        bm = df["backup_mean"].dropna()
        if len(bm) > 10:
            first_half = bm.iloc[:len(bm)//2].mean()
            second_half = bm.iloc[len(bm)//2:].mean()
            if second_half > first_half:
                print("backup_mean is trending upward — agent is improving.")
            else:
                issues.append("backup_mean is not increasing — reward signal may be stagnating.")

    if "entropy" in df.columns:
        ent = df["entropy"].dropna()
        if len(ent) > 10:
            late_ent = ent.iloc[-max(1, len(ent)//5):].mean()
            if late_ent < 0.1:
                issues.append(f"Entropy collapsed near zero ({late_ent:.4f}) — policy may be deterministic or stuck.")
            else:
                print(f"Entropy is stable ({late_ent:.4f}) — normal for SAC with automatic α tuning.")

    if "q_gap" in df.columns:
        qg = df["q_gap"].dropna()
        if qg.max() > 1.0:
            issues.append(f"q_gap peaked at {qg.max():.3f} — possible Q-value overestimation.")
        else:
            print("q_gap is small — no obvious Q-value divergence.")
        mid = qg.iloc[len(qg)//4 : -len(qg)//4]
        if len(mid) > 10 and mid.std() > qg.std() * 2.0:
            issues.append(f"High volatility in mid-run q_gap (mid std {mid.std():.4f} vs overall {qg.std():.4f}) — possible Q-value divergence event.")

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
        ("loss_actor",       "Actor Loss",             "#a78bfa", "down"),
        ("entropy",          "Entropy",                "#38bdf8", "down"),
        ("q_gap",            "Q-Gap (Q1 vs Q2)",       "#facc15", "down"),
        ("backup_mean",      "Backup Mean",            "#4ade80", "up"),
        ("backup_min",       "Backup Min",             "#fb7185", None),
        ("q1_mean",          "Q1 Mean",                "#67e8f9", None),
        ("q2_mean",          "Q2 Mean",                "#c084fc", None),
        ("entropy_coef",     "Entropy Coef (alpha)",   "#fbbf24", None),
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
        spikes = detect_spikes(s, SPIKE_STD_MULT.get(col, SPIKE_STD_MULT["default"]))
        if not spikes.empty:
            ax.scatter(x.loc[spikes.index], spikes, color="#ef4444",
                       s=12, zorder=5, label=f"spikes ({len(spikes)})")
            ax.legend(fontsize=6, facecolor="#1a1d27", labelcolor="white", framealpha=0.6)

        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("step", color="#6b7280", fontsize=7)
        ax.grid(True, color="#2d3148", linewidth=0.5, linestyle="--")

    out = save_path or "sac_training_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Dashboard saved → {out}")
    plt.show()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        print("Usage: python analyze_sac_log.py [path/to/csv]")
        sys.exit(1)

    df = load_csv(csv_path)
    print_summary(df)

    out_img = os.path.join(os.path.dirname(csv_path), "sac_training_dashboard.png")
    plot_dashboard(df, save_path=out_img)


if __name__ == "__main__":
    main()