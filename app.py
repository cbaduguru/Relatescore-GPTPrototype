import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math import pi

# ------------------------------------------------------------
# RelateScore™ Streamlit Prototype (RQ Wheel demo)
# - Ring-style RQ Wheel: per-category colors + score-scaled intensity
# - Labels placed inside the wheel segments
# - Simple stability smoothing (EMA + capped delta)
# ------------------------------------------------------------

st.set_page_config(page_title="RelateScore™", page_icon="●", layout="centered")

# -----------------------------
# UI / Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 760px; padding-top: 24px; }
      h1, h2, h3 { color: #1A1A1A; font-family: Inter, system-ui, -apple-system, sans-serif; }
      .small-muted { color:#666; font-size: 0.92rem; }
      .logo-wrap { text-align:center; margin-bottom: 10px; }
      .logo-title { font-size:22px; font-weight:700; }
      .logo-tag { margin-top:2px; }
      .stButton > button {
        background-color: #C6A667 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 18px !important;
        width: 100% !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_logo():
    st.markdown(
        """
        <div class="logo-wrap">
          <div class="logo-title">RelateScore™</div>
          <div class="small-muted logo-tag">Private reflection. Shared only by choice.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Categories + Color Index
# -----------------------------
CATEGORIES = [
    "Emotional Awareness",
    "Communication Style",
    "Conflict Tendencies",
    "Attachment Patterns",
    "Empathy & Responsiveness",
    "Self-Insight",
    "Trust & Boundaries",
    "Stability & Consistency",
]

CATEGORY_COLORS = {
    "Emotional Awareness": "#4A90E2",       # Soft Navy Blue
    "Communication Style": "#7ED321",       # Muted Green
    "Conflict Tendencies": "#FF6B6B",       # Warm Red-Orange
    "Attachment Patterns": "#A29BFE",       # Gentle Purple
    "Empathy & Responsiveness": "#FFD700",  # Sunny Yellow
    "Self-Insight": "#5A67D8",              # Deep Indigo
    "Trust & Boundaries": "#20C997",        # Earthy Teal
    "Stability & Consistency": "#A1887F",   # Neutral Taupe
}

# Split labels so they fit inside ring segments
CATEGORY_LABELS = {
    "Emotional Awareness": "Emotional\nAwareness",
    "Communication Style": "Communication\nStyle",
    "Conflict Tendencies": "Conflict\nTendencies",
    "Attachment Patterns": "Attachment\nPatterns",
    "Empathy & Responsiveness": "Empathy &\nResponsiveness",
    "Self-Insight": "Self-Insight",
    "Trust & Boundaries": "Trust &\nBoundaries",
    "Stability & Consistency": "Stability &\nConsistency",
}

# -----------------------------
# Color helpers
# -----------------------------
def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

def _blend_hex(a: str, b: str, t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    r = ar + (br - ar) * t
    g = ag + (bg - ag) * t
    bl = ab + (bb - ab) * t
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(bl * 255))

def dynamic_color(category: str, score: float) -> str:
    """Lower scores stay closer to neutral; higher scores saturate toward category color."""
    neutral = "#F5F5F5"
    base = CATEGORY_COLORS.get(category, "#4A90E2")
    intensity = float(np.clip((score - 20.0) / 70.0, 0.0, 1.0))  # 20->0, 90->1
    return _blend_hex(neutral, base, 0.35 + 0.65 * intensity)

# -----------------------------
# Stability smoothing (EMA + capped delta)
# -----------------------------
EMA_ALPHA = 0.25
MAX_STEP = 15.0  # cap per update (prototype)

def smooth_scores(raw: dict, prev: dict | None) -> dict:
    if not prev:
        return raw
    out = {}
    for k in CATEGORIES:
        new = float(raw[k])
        old = float(prev.get(k, new))
        delta = new - old
        # cap delta
        if abs(delta) > MAX_STEP:
            delta = np.sign(delta) * MAX_STEP
        out[k] = float(np.clip(old + EMA_ALPHA * delta, 20, 90))
    return out

def compute_scores() -> dict:
    # Demo generator: in production, replace with your real assessment scoring
    raw = {c: float(random.uniform(30, 85)) for c in CATEGORIES}
    prev = st.session_state.get("prev_scores")
    smoothed = smooth_scores(raw, prev)
    st.session_state.prev_scores = smoothed
    return smoothed

# -----------------------------
# RQ Wheel renderer (ring style)
# -----------------------------
def draw_rq_wheel(scores: dict):
    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * pi, n, endpoint=False)
    width = 2 * pi / n

    bg = "#FAF7F2"
    gold = "#C9A96E"
    text = "#1A1A1A"

    # Ring geometry
    r_inner = 30.0
    r_outer = 95.0
    ring_h = r_outer - r_inner

    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rlim(0, 100)
    ax.spines["polar"].set_visible(False)

    # Subtle concentric guides
    for r in [r_inner, r_inner + ring_h * 0.25, r_inner + ring_h * 0.5, r_inner + ring_h * 0.75, r_outer]:
        ax.plot(np.linspace(0, 2 * pi, 361), np.full(361, r), color=gold, lw=0.6, alpha=0.25, zorder=0)

    for i, cat in enumerate(CATEGORIES):
        theta = angles[i]
        score = float(scores.get(cat, 0.0))
        score01 = float(np.clip(score / 100.0, 0.0, 1.0))

        base_neutral = "#F5F5F5"
        base_color = CATEGORY_COLORS.get(cat, "#4A90E2")

        # Base wedge
        ax.bar(
            theta,
            ring_h,
            width=width * 0.98,
            bottom=r_inner,
            color=base_neutral,
            edgecolor=gold,
            linewidth=0.9,
            alpha=0.70,
            align="edge",
            zorder=1,
        )

        # Colored fill scaled by score
        fill_h = ring_h * score01
        col = dynamic_color(cat, score)
        ax.bar(
            theta,
            fill_h,
            width=width * 0.98,
            bottom=r_inner,
            color=col,
            edgecolor="none",
            alpha=0.25 + 0.60 * score01,
            align="edge",
            zorder=2,
        )

        # Marker at current score radius
        ax.scatter(
            [theta + width / 2],
            [r_inner + fill_h],
            s=50,
            c=[base_color],
            edgecolors=text,
            linewidths=0.7,
            zorder=5,
        )

        # Label INSIDE wedge
        label = CATEGORY_LABELS.get(cat, cat)
        ax.text(
            theta + width / 2,
            r_inner + ring_h * 0.52,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text,
            zorder=6,
        )

    # Inner + outer ring outlines
    ax.plot(np.linspace(0, 2 * pi, 361), np.full(361, r_outer), color=gold, lw=2.0, alpha=0.9, zorder=10)
    ax.plot(np.linspace(0, 2 * pi, 361), np.full(361, r_inner), color=gold, lw=1.4, alpha=0.7, zorder=10)

    st.pyplot(fig, use_container_width=True)

# -----------------------------
# App
# -----------------------------
render_logo()
st.header("RQ Wheel")

if "scores" not in st.session_state:
    st.session_state.scores = compute_scores()

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("Recalculate (simulate update)", key="recalc"):
        st.session_state.scores = compute_scores()
with c2:
    st.markdown(
        "<div class='small-muted' style='text-align:right;'>"
        "Color intensity reflects current strength."
        "</div>",
        unsafe_allow_html=True,
    )

draw_rq_wheel(st.session_state.scores)

with st.expander("Debug: raw values", expanded=False):
    st.json({k: round(float(v), 2) for k, v in st.session_state.scores.items()})
