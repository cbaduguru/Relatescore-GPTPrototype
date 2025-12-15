import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import time

# ------------------------------------------------------------
# RelateScore™ Streamlit Prototype (Cloud-safe navigation)
# - Entry screen: only Create Profile + Log In (no Enter Invite Code)
# - Home screen: exactly 3 buttons (Create Invite, Enter Invite Code, Withdraw and Reset)
# - Tip microcopy appears directly under every "Enter Invite Code" button
# - Invite codes work across sessions on the same Streamlit Cloud instance via shared in-memory store
# ------------------------------------------------------------
st.set_page_config(page_title="RelateScore™", page_icon="✅", layout="centered")
st.set_option("client.showErrorDetails", True)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .block-container { max-width: 520px; padding-top: 24px; }
        h1, h2, h3 { color: #1A1A1A; font-family: sans-serif; }
        .stButton > button {
            background-color: #C6A667 !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 10px 18px !important;
            width: 100% !important;
        }
        .insight-card {
            background-color: #FFFFFF;
            border: 1px solid #C6A667;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
        }
        .logo { text-align:center; margin-bottom: 10px; padding-top: 10px; overflow: visible !important; }
        .logo svg { display:block; margin:0 auto; overflow: visible !important; }
        .tagline { text-align:center; color:#3A3A3A; margin-bottom: 18px; }
        .rgi-big { font-size: 54px; font-weight: 800; color: #C6A667; text-align: center; line-height: 1.0; }
        .small-muted { color:#666; font-size: 0.92rem; }
        .tip-under-btn { margin-top: -10px; margin-bottom: 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

LOGO_SVG = """
<div class="logo">
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-label="RelateScore logo">
  <circle cx="32" cy="32" r="20" stroke="#C6A667" stroke-width="4" fill="none"/>
  <path d="M22 32 L29 39 L44 24" stroke="#C6A667" stroke-width="4" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<div style="font-size: 22px; font-weight: 700; margin-top: 6px;">RelateScore™</div>
</div>
"""

def display_logo():
    st.markdown(LOGO_SVG, unsafe_allow_html=True)

# -----------------------------
# Rerun compatibility (Cloud safe)
# -----------------------------
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def nav(to_page: str):
    st.session_state.page = to_page
    _rerun()

# -----------------------------
# Invite Store (shared across sessions)
# -----------------------------
INVITE_TTL_SECONDS = 60 * 30  # 30 minutes

@st.cache_resource
def get_invite_store():
    # { CODE: {"created_at": ts, "used": bool} }
    return {}

def _clean_expired_invites(store: dict):
    now = time.time()
    expired = [code for code, meta in store.items()
               if (now - meta.get("created_at", now)) > INVITE_TTL_SECONDS]
    for code in expired:
        store.pop(code, None)

def register_invite(code: str) -> None:
    store = get_invite_store()
    _clean_expired_invites(store)
    store[code] = {"created_at": time.time(), "used": False}

def validate_invite(code: str):
    """
    Returns (is_valid, reason)
    Reasons: ok | missing | expired | used
    """
    store = get_invite_store()
    _clean_expired_invites(store)
    meta = store.get(code)
    if not meta:
        return False, "missing"
    if (time.time() - meta.get("created_at", time.time())) > INVITE_TTL_SECONDS:
        store.pop(code, None)
        return False, "expired"
    if meta.get("used"):
        return False, "used"
    return True, "ok"

def consume_invite(code: str) -> None:
    store = get_invite_store()
    meta = store.get(code)
    if meta:
        meta["used"] = True

# -----------------------------
# Data
# -----------------------------
CATEGORIES = [
    "Emotional Awareness",
    "Communication Style",
    "Conflict Tendencies",
    "Attachment Patterns",
    "Empathy & Responsiveness",
    "Self-Insight",
    "Trust & Boundaries",
    "Stability & Consistency"
]

# -----------------------------
# RQ Wheel Color System (per category)
# - Uses RelateScore palette where possible (Accent Blue / Mint / Gold)
# - Adds distinct, premium-safe supporting colors for clear differentiation
# -----------------------------
CATEGORY_COLORS = {
    "Emotional Awareness": "#2E6AF3",        # Accent Blue
    "Communication Style": "#0C9A6F",        # Success Green
    "Conflict Tendencies": "#E54646",        # Error Red
    "Attachment Patterns": "#6B5B95",        # Deep Violet (supporting)
    "Empathy & Responsiveness": "#A6E3DA",   # Mint
    "Self-Insight": "#F4A623",               # Warning Amber
    "Trust & Boundaries": "#C6A667",         # Gold
    "Stability & Consistency": "#1A1A1A",    # Charcoal
}

def _hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _blend_hex(c1: str, c2: str, t: float) -> str:
    """Blend c1->c2 with t in [0,1]. Returns hex string."""
    t = float(np.clip(t, 0.0, 1.0))
    r1, g1, b1 = _hex_to_rgb01(c1)
    r2, g2, b2 = _hex_to_rgb01(c2)
    r = r1 + (r2 - r1) * t
    g = g1 + (g2 - g1) * t
    b = b1 + (b2 - b1) * t
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

def _category_dynamic_color(category: str, score: float) -> str:
    """Real-time color per category based on its score (0-100):
    - Low scores bias toward a warm neutral (subtle)
    - High scores move toward the category's base color
    """
    base = CATEGORY_COLORS.get(category, "#2E6AF3")
    warm_neutral = "#FAFAF8"  # Warm Surface
    # Map score to intensity; keep conservative so it stays premium
    intensity = float(np.clip((score - 20.0) / 70.0, 0.0, 1.0))  # 20->0, 90->1
    return _blend_hex(warm_neutral, base, intensity)

def draw_rq_wheel(ax, categories, scores_dict):
    """Draw an RQ Wheel with per-category colors + wedge fills."""
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values = np.array([float(scores_dict[c]) for c in categories], dtype=float)

    # Close the polygon
    angles_loop = np.concatenate([angles, [angles[0]]])
    values_loop = np.concatenate([values, [values[0]]])

    # Background + grid styling
    ax.set_facecolor("#FAFAF8")
    ax.grid(True, linewidth=0.8, alpha=0.25)
    ax.spines["polar"].set_alpha(0.25)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])

    # Colored wedges per category (gives the "real-time" multi-color feel)
    for i in range(n):
        a0 = angles[i]
        a1 = angles[(i + 1) % n]
        v0 = values[i]
        v1 = values[(i + 1) % n]

        # Handle wrap-around for the last wedge
        if i == n - 1:
            a1 = angles[0] + 2 * np.pi

        col = _category_dynamic_color(categories[i], v0)
        ax.fill([a0, a0, a1, a1], [0, v0, v1, 0], color=col, alpha=0.22, linewidth=0)

    # Outline polygon (neutral premium stroke)
    ax.plot(angles_loop, values_loop, linewidth=2.2, alpha=0.9)

    # Markers per axis in category color
    for i, cat in enumerate(categories):
        v = float(values[i])
        mcol = _category_dynamic_color(cat, v)
        ax.scatter([angles[i]], [v], s=60, c=[mcol], edgecolors="#1A1A1A", linewidths=0.6, zorder=5)

    # Category labels, colored to match
    ax.set_xticks(angles)
    labels = []
    for i, cat in enumerate(categories):
        v = float(values[i])
        labels.append(cat)
        # Apply colored tick labels after set_xticklabels
    ax.set_xticklabels(labels, fontsize=10)
    for tick, cat in zip(ax.get_xticklabels(), categories):
        tick.set_color(CATEGORY_COLORS.get(cat, "#1A1A1A"))
        tick.set_fontweight("medium")

LIKERT_QUESTIONS = {
    cat: [
        f"On a scale of 1–5, how important is {cat.lower()} to you in relationships?",
        f"How would you rate your current level in {cat.lower()}?",
        f"How often do you reflect on {cat.lower()}?"
    ]
    for cat in CATEGORIES
}

ASSESSMENT_QUESTIONS = {
    cat: [
        f"How often do you recognize patterns in {cat.lower()}?",
        f"How comfortable are you discussing {cat.lower()}?",
        f"How does {cat.lower()} impact your connections?"
    ]
    for cat in CATEGORIES
}

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    defaults = {
        "page": "entry",
        "logged_in": False,
        "consent_accepted": False,

        # Invite flow (local convenience)
        "invite_code": None,  # last generated code in THIS session
        "partner_code": "",

        # Assessment flow
        "use_mutual": False,
        "likert_responses": {},
        "assessment_responses": {},
        "scores": None,
        "raw_scores": None,
        "prev_scores": None,
        "prev_scores_ts": None,
        "score_history": [],
        "insights": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_state():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()

init_state()

# -----------------------------
# Helpers
# -----------------------------
# -----------------------------
# Stability Smoothing (EMA + Dampening)
# Notes:
# - In this Streamlit prototype we store prior scores in session_state (per browser session).
# - In production, persist these per-user in your backend so smoothing is consistent across devices/sessions.
EMA_ALPHA = 0.25  # 0<alpha<=1; lower = smoother, higher = more responsive
MAX_DAILY_CHANGE = 15.0  # max allowed change in score points per day (per category)
MIN_CHANGE_FLOOR = 2.0   # minimum allowed change even if dt is very small (prevents "stuck" feeling)
OUTLIER_SOFT_THRESHOLD = 25.0  # deltas above this get compressed ("dampened")

def _now_ts() -> float:
    return time.time()

def _dt_days(prev_ts: float | None) -> float:
    if not prev_ts:
        return 1.0
    dt = max(0.0, _now_ts() - float(prev_ts))
    return max(dt / 86400.0, 1.0 / 1440.0)  # at least 1 minute

def _dampen_delta(delta: float, threshold: float = OUTLIER_SOFT_THRESHOLD) -> float:
    """Soft dampening: compress very large deltas without hard-clipping."""
    ad = abs(delta)
    if ad <= threshold:
        return delta
    # Beyond threshold, compress using a square-root curve (smooth, monotonic)
    compressed = threshold + (ad - threshold) ** 0.5 * 5.0
    return float(np.sign(delta) * compressed)

def _cap_delta(delta: float, allowed: float) -> float:
    if abs(delta) <= allowed:
        return delta
    return float(np.sign(delta) * allowed)

def smooth_scores(new_scores: dict, prev_scores: dict | None, prev_ts: float | None) -> dict:
    """Apply EMA smoothing + outlier dampening + max-delta cap to category scores (not including RGI)."""
    if not prev_scores:
        return new_scores

    days = _dt_days(prev_ts)
    allowed = max(MIN_CHANGE_FLOOR, MAX_DAILY_CHANGE * days)

    smoothed = {}
    for cat in CATEGORIES:
        new_v = float(new_scores.get(cat, 0.0))
        old_v = float(prev_scores.get(cat, new_v))

        # 1) dampen outliers in the update step
        raw_delta = new_v - old_v
        damp_delta = _dampen_delta(raw_delta)

        # 2) EMA on the dampened target
        target = old_v + damp_delta
        ema = old_v + EMA_ALPHA * (target - old_v)

        # 3) cap maximum movement based on elapsed time
        capped_delta = _cap_delta(ema - old_v, allowed)
        smoothed[cat] = float(np.clip(old_v + capped_delta, 20, 90))

    return smoothed

def generate_invite_code(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def compute_scores():
    # --- Step 1: Compute "raw" category scores from the current assessment session
    raw_cat_scores = {}
    for cat in CATEGORIES:
        likert_vals = [st.session_state.likert_responses[q] for q in LIKERT_QUESTIONS[cat]]
        assess_vals = [st.session_state.assessment_responses[q] for q in ASSESSMENT_QUESTIONS[cat]]

        baseline = float(np.mean(likert_vals)) * 20.0
        raw = float(np.mean(assess_vals)) * 20.0

        score = (raw / baseline) * 50.0 if baseline > 0 else raw

        if st.session_state.use_mutual:
            mutual = float(np.random.uniform(40, 80))
            score = 0.4 * score + 0.6 * mutual

        raw_cat_scores[cat] = float(np.clip(score, 20, 90))

    st.session_state.raw_scores = dict(raw_cat_scores)

    # --- Step 2: Apply stability smoothing (EMA + dampening)
    prev_scores = st.session_state.get("prev_scores")
    prev_ts = st.session_state.get("prev_scores_ts")
    smoothed_cats = smooth_scores(raw_cat_scores, prev_scores, prev_ts)

    # --- Step 3: Compute RGI from the (smoothed) category scores
    weights = np.array([0.15, 0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10], dtype=float)
    rgi = float(np.sum(np.array([smoothed_cats[c] for c in CATEGORIES], dtype=float) * weights))

    final_scores = dict(smoothed_cats)
    final_scores["RGI"] = float(np.clip(rgi, 20, 90))

    # --- Step 4: Persist the smoothed state for next computation (prototype: per session)
    st.session_state.scores = final_scores
    st.session_state.prev_scores = dict(smoothed_cats)
    st.session_state.prev_scores_ts = _now_ts()

    # Optional: keep a short history for debugging / future UI
    hist = st.session_state.get("score_history", [])
    hist.append({
        "ts": st.session_state.prev_scores_ts,
        "raw": dict(raw_cat_scores),
        "smoothed": dict(smoothed_cats),
        "rgi": final_scores["RGI"],
    })
    st.session_state.score_history = hist[-20:]

def generate_insights():
    insights = []
    for cat in CATEGORIES:
        score = st.session_state.scores.get(cat, 0)
        if score > 70:
            type_ = "Strength"
            desc = "This is a strong foundation to build on."
        elif score < 40:
            type_ = "Blind Spot"
            desc = "This pattern may create misunderstandings."
        else:
            type_ = "Neutral"
            desc = "Balanced area with room for awareness."
        insights.append({
            "category": cat,
            "type": type_,
            "description": desc,
            "suggestion": "Consider a small experiment this week to shift this pattern by 1%."
        })
    st.session_state.insights = insights

def tip_microcopy():
    st.markdown(
        "<div class='small-muted tip-under-btn'>Tip: If you're joining via code, the sender must generate one first.</div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------

def home_footer_microcopy():
    st.markdown(
        "<hr style='margin-top:32px;margin-bottom:12px;'>"
        "<div class='small-muted' style='text-align:center;'>"
        "RelateScore™ provides private relational clarity that supports growth without judgment or exposure."
        "</div>",
        unsafe_allow_html=True
    )
def entry_page():
    display_logo()
    st.markdown('<div class="tagline">Private reflection. Shared only by choice.</div>', unsafe_allow_html=True)

    if st.button("Create Profile", key="entry_create"):
        nav("create_profile")

    if st.button("Log In", key="entry_login"):
        nav("log_in")

def create_profile_page():
    display_logo()
    st.header("Create your private profile")
    st.write("Your responses are encrypted and visible only by choice.")

    st.session_state.consent_accepted = st.checkbox(
        "I understand that my reflections are private, encrypted, and can be deleted at any time.",
        value=st.session_state.consent_accepted,
        key="consent_checkbox"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="create_back"):
            nav("entry")
    with c2:
        if st.button("Continue", key="create_continue", disabled=not st.session_state.consent_accepted):
            st.session_state.logged_in = True
            nav("home")

def log_in_page():
    display_logo()
    st.header("Welcome back")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="login_back"):
            nav("entry")
    with c2:
        if st.button("Log In", key="login_go"):
            st.session_state.logged_in = True
            nav("home")

def home_page():
    """
    Home shows exactly 3 buttons:
      1) Create Invite
      2) Enter Invite Code (with tip microcopy underneath)
      3) Withdraw and Reset
    """
    display_logo()
    st.header("Home")

    if not st.session_state.logged_in:
        st.warning("Please log in or create a profile to continue.")
        if st.button("Return to Entry", key="home_return_entry"):
            nav("entry")
        return

    if st.button("Create Invite", key="home_create_invite"):
        code = generate_invite_code()
        st.session_state.invite_code = code
        register_invite(code)
        nav("create_invite")

    if st.button("Enter Invite Code", key="home_enter_invite"):
        nav("enter_invite")
    tip_microcopy()

    if st.button("Withdraw and Reset", key="home_reset"):
        reset_state()
        nav("entry")

    home_footer_microcopy()


def create_invite_page():
    display_logo()
    st.header("Create Invite")

    if not st.session_state.invite_code:
        code = generate_invite_code()
        st.session_state.invite_code = code
        register_invite(code)

    st.write("Share this invitation code privately with your partner:")
    st.code(st.session_state.invite_code)

    st.markdown(
        f"<div class='small-muted'>This code expires in {INVITE_TTL_SECONDS//60} minutes and can be used once.</div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Home", key="invite_back_home"):
            nav("home")
    with c2:
        if st.button("Enter Invite Code", key="invite_go_enter"):
            nav("enter_invite")
    tip_microcopy()

def enter_invite_page():
    display_logo()
    st.header("Enter Invite")
    st.write("Entering the invitation code transitions you into the full application experience.")

    st.session_state.partner_code = st.text_input(
        "Invitation code",
        value=st.session_state.partner_code,
        key="partner_code_input"
    ).strip().upper()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="enter_invite_back"):
            nav("home" if st.session_state.logged_in else "entry")
    with c2:
        if st.button("Continue", key="enter_invite_continue"):
            if not st.session_state.partner_code:
                st.error("Please enter a code.")
                return

            is_ok, reason = validate_invite(st.session_state.partner_code)
            if is_ok:
                consume_invite(st.session_state.partner_code)
                nav("reflection_start")
            else:
                if reason == "expired":
                    st.error("This code has expired. Ask the sender to generate a new one.")
                elif reason == "used":
                    st.error("This code has already been used. Ask the sender to generate a new one.")
                else:
                    st.error("Code not recognized. Ask the sender to generate a new code and share it again.")

def reflection_start_page():
    display_logo()
    st.header("Begin when ready")
    st.write("There are no right or wrong answers.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="refstart_back"):
            nav("home" if st.session_state.logged_in else "entry")
    with c2:
        if st.button("Start Reflection", key="refstart_go"):
            nav("likert")

def likert_page():
    display_logo()
    st.header("Personal Calibration")

    for cat_i, cat in enumerate(CATEGORIES):
        st.subheader(cat)
        for q_i, q in enumerate(LIKERT_QUESTIONS[cat]):
            st.session_state.likert_responses[q] = st.slider(
                q, 1, 5, 3, key=f"likert_{cat_i}_{q_i}"
            )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="likert_back"):
            nav("reflection_start")
    with c2:
        if st.button("Proceed", key="likert_next"):
            nav("preview")

def preview_page():
    display_logo()
    st.header("Preview")

    st.write("You will receive:")
    st.write("- A private Relationship Growth Index (RGI)")
    st.write("- A wheel showing patterns")
    st.write("- Strengths, blind spots, and growth areas")

    st.session_state.use_mutual = st.checkbox(
        "Include simulated mutual reflection?",
        value=st.session_state.use_mutual,
        key="mutual_checkbox"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="preview_back"):
            nav("likert")
    with c2:
        if st.button("Proceed to Assessment", key="preview_next"):
            nav("assessment")

def assessment_page():
    display_logo()
    st.header("Relational Assessment")

    for cat_i, cat in enumerate(CATEGORIES):
        st.subheader(cat)
        for q_i, q in enumerate(ASSESSMENT_QUESTIONS[cat]):
            st.session_state.assessment_responses[q] = st.slider(
                q, 1, 5, 3, key=f"assess_{cat_i}_{q_i}"
            )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="assess_back"):
            nav("preview")
    with c2:
        if st.button("Submit", key="assess_submit"):
            if np.random.rand() < 0.1:
                st.error("Input blocked for toxicity. Please revise.")
            else:
                compute_scores()
                generate_insights()
                nav("dashboard")

def dashboard_page():
    display_logo()
    st.header("Dashboard")

    if not st.session_state.scores:
        st.warning("No results found yet. Please complete the assessment.")
        if st.button("Go to Assessment", key="dash_go_assessment"):
            nav("assessment")
        return

    st.markdown(f"<div class='rgi-big'>{st.session_state.scores['RGI']:.1f}</div>", unsafe_allow_html=True)
    st.caption("Relationship Growth Index")

    # Debug/verification: show smoothing behavior (optional)
    with st.expander("Stability smoothing (EMA) details", expanded=False):
        st.write(f"EMA alpha: {EMA_ALPHA}")
        st.write(f"Max daily change: {MAX_DAILY_CHANGE} points/day (min floor {MIN_CHANGE_FLOOR})")
        if st.session_state.raw_scores:
            st.caption("Raw vs smoothed category scores (prototype debug view)")
            rows = []
            for cat in CATEGORIES:
                raw_v = float(st.session_state.raw_scores.get(cat, np.nan))
                sm_v = float(st.session_state.scores.get(cat, np.nan))
                rows.append({
                    "Category": cat,
                    "Raw": round(raw_v, 1),
                    "Smoothed": round(sm_v, 1),
                    "Delta": round(sm_v - raw_v, 1),
                })
            st.dataframe(rows, use_container_width=True)

    # RQ Wheel (multi-color, real-time per category)
    scores = st.session_state.scores
    fig, ax = plt.subplots(figsize=(6.3, 6.3), subplot_kw=dict(polar=True))
    draw_rq_wheel(ax, CATEGORIES, scores)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Key Insights")
    for insight in (st.session_state.insights or []):
        st.markdown(
            f"""
            <div class="insight-card">
                <div style="font-weight:700;">{insight['category']}: {insight['type']}</div>
                <div>{insight['description']}</div>
                <div><i>Suggestion: {insight['suggestion']}</i></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("Withdraw and Reset", key="dash_reset"):
        reset_state()
        nav("entry")

    if st.button("Return to Home", key="dash_home"):
        nav("home")

# -----------------------------
# Router
# -----------------------------
PAGES = {
    "entry": entry_page,
    "create_profile": create_profile_page,
    "log_in": log_in_page,
    "home": home_page,
    "create_invite": create_invite_page,
    "enter_invite": enter_invite_page,
    "reflection_start": reflection_start_page,
    "likert": likert_page,
    "preview": preview_page,
    "assessment": assessment_page,
    "dashboard": dashboard_page,
}

page = st.session_state.get("page", "entry")
PAGES.get(page, entry_page)()
