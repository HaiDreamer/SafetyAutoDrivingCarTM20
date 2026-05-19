"""
TMRL Debug Dashboard
====================
In-process dashboard using a background thread + shared state dict.
Mimics the Openplanet dashboard arrow button layout.

To disable entirely, set:
    DASHBOARD_ENABLED = False

Usage in networking.py:
    1. Import at the top:
           from tmrl.debug_dashboard import start_dashboard, update_dashboard

    2. In RolloutWorker.__init__(), add at the end:
           start_dashboard()

    3. In RolloutWorker.step(), after self.env.step(act), add:
           update_dashboard(new_obs, act)
"""

import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrow
from matplotlib.animation import FuncAnimation

# ── Toggle this to disable the dashboard completely ──────────────────────────
DASHBOARD_ENABLED = False
# ─────────────────────────────────────────────────────────────────────────────

_state = {"speed": 0.0, "gas": 0.0, "brake": 0.0, "steer": 0.0}
_lock  = threading.Lock()

PINK  = "#FF2D8B"
WHITE = "#FFFFFF"
DIM   = "#2a2a2a"
BG    = "#111111"


def update_dashboard(new_obs, act):
    """Call each step. new_obs[0]=speed (m/s), act=[gas, brake, steer]."""
    if not DASHBOARD_ENABLED:
        return
    speed = float(np.array(new_obs[0]).flatten()[0]) * 3.6  # m/s → km/h
    with _lock:
        _state["speed"] = speed
        _state["gas"]   = float(act[0])   # 0.0 – 1.0
        _state["brake"] = float(act[1])   # 0.0 – 1.0
        _state["steer"] = float(act[2])   # -1.0 (left) – 1.0 (right)


def _run_dashboard():
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=BG)
    fig.canvas.manager.set_window_title("TMRL Dashboard")
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    # ── Layout constants ──────────────────────────────────────────────────
    cx, cy   = 5.0, 4.8   # center of the button cross
    bw, bh   = 1.6, 1.4   # button width / height
    gap      = 0.18        # gap between buttons
    side_w   = 2.0         # width of left/right arrow
    side_h   = 2.0         # height of left/right arrow

    def make_rect(x, y, w, h, color):
        """Return a rounded rectangle patch centered at (x,y)."""
        from matplotlib.patches import FancyBboxPatch
        return FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            linewidth=1.5, edgecolor=WHITE,
            facecolor=color, zorder=3
        )

    def make_triangle(cx, cy, direction, color):
        """Return a triangle patch. direction: 'up','down','left','right'."""
        from matplotlib.patches import Polygon
        s = 0.55  # half-size
        if direction == "up":
            pts = [(cx, cy + s), (cx - s, cy - s * 0.7), (cx + s, cy - s * 0.7)]
        elif direction == "down":
            pts = [(cx, cy - s), (cx - s, cy + s * 0.7), (cx + s, cy + s * 0.7)]
        elif direction == "left":
            pts = [(cx - s, cy), (cx + s * 0.7, cy + s), (cx + s * 0.7, cy - s)]
        else:  # right
            pts = [(cx + s, cy), (cx - s * 0.7, cy + s), (cx - s * 0.7, cy - s)]
        return plt.Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=5)

    # ── Up button ─────────────────────────────────────────────────────────
    up_y = cy + bh / 2 + gap + bh / 2
    rect_up  = make_rect(cx, up_y, bw, bh, DIM)
    arr_up   = make_triangle(cx, up_y, "up", WHITE)

    # ── Down button ───────────────────────────────────────────────────────
    dn_y = cy - bh / 2 - gap - bh / 2
    rect_dn  = make_rect(cx, dn_y, bw, bh, DIM)
    arr_dn   = make_triangle(cx, dn_y, "down", WHITE)

    # ── Left button ───────────────────────────────────────────────────────
    lx = cx - bw / 2 - gap - side_w / 2
    rect_lt  = make_rect(lx, cy, side_w, side_h, DIM)
    arr_lt   = make_triangle(lx, cy, "left", WHITE)

    # ── Right button ──────────────────────────────────────────────────────
    rx = cx + bw / 2 + gap + side_w / 2
    rect_rt  = make_rect(rx, cy, side_w, side_h, DIM)
    arr_rt   = make_triangle(rx, cy, "right", WHITE)

    for patch in [rect_up, rect_dn, rect_lt, rect_rt,
                  arr_up, arr_dn, arr_lt, arr_rt]:
        ax.add_patch(patch)

    # ── Speed text ────────────────────────────────────────────────────────
    speed_txt = ax.text(cx, 9.5, "0 km/h",
                        ha="center", va="top", fontsize=14,
                        fontweight="bold", color=WHITE, zorder=6)



    def animate(_frame):
        with _lock:
            speed = _state["speed"]
            gas   = _state["gas"]
            brake = _state["brake"]
            steer = _state["steer"]

        # Highlight: gas → up, brake → down, steer left → left, steer right → right
        rect_up.set_facecolor(PINK  if gas   > 0.05 else DIM)
        rect_dn.set_facecolor(PINK  if brake > 0.05 else DIM)
        rect_lt.set_facecolor(PINK  if steer < -0.05 else DIM)
        rect_rt.set_facecolor(PINK  if steer >  0.05 else DIM)

        speed_txt.set_text(f"{speed:.1f} km/h")
        return rect_up, rect_dn, rect_lt, rect_rt, speed_txt

    _ani = FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
    plt.show()


def start_dashboard():
    """Call once from RolloutWorker.__init__() to open the dashboard window."""
    if not DASHBOARD_ENABLED:
        return
    t = threading.Thread(target=_run_dashboard, daemon=True)
    t.start()