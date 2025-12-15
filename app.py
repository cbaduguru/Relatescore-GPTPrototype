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
# RQ Wheel Color Index (per-category)
# -----------------------------
CATEGORY_COLORS = {
    "Emotional Awareness": "#4A90E2",      # Soft Navy Blue
    "Communication Style": "#7ED321",      # Muted Green
    "Conflict Tendencies": "#FF6B6B",      # Warm Red-Orange
    "Attachment Patterns": "#A29BFE",      # Gentle Purple
    "Empathy & Responsiveness": "#FFD700", # Sunny Yellow
    "Self-Insight": "#5A67D8",             # Deep Indigo
    "Trust & Boundaries": "#20C997",       # Earthy Teal
    "Stability & Consistency": "#A1887F",  # Neutral Taupe
}

# Line-broken labels so text fits inside each sector
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

def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _mix_hex(hex_a: str, hex_b: str, t: float) -> str:
    # Linear blend between two hex colors (t in [0,1])
    a = np.array(_hex_to_rgb01(hex_a), dtype=float)
    b = np.array(_hex_to_rgb01(hex_b), dtype=float)
    c = (1 - t) * a + t * b
    c = np.clip(c, 0, 1)
    return "#{:02X}{:02X}{:02X}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))

def draw_rq_wheel(scores: dict):
    """
    Ring-style RQ Wheel:
      - Each category is a colored sector wedge
      - Sector color intensity and fill length respond to the category score
      - Colored point marker per category
      - Colored label placed inside each sector
      - Warm neutral background + gold grid/outline (UI kit aligned)
    """
    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n

    # UI-kit aligned neutrals
    bg = "#FAF7F2"         # warm light beige
    gold = "#C6A667"       # matte gold accent (button color in this prototype)
    gold_grid = "#C9A96E"  # slightly warmer gold for grid lines

    r_inner = 22.0
    r_outer = 92.0
    ring_h = r_outer - r_inner

    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Remove default labels; we place custom labels inside wedges
    ax.set_xticks([])
    ax.set_yticks([])

    # Subtle radial guides
    for r in [r_inner, r_inner + ring_h*0.25, r_inner + ring_h*0.5, r_inner + ring_h*0.75, r_outer]:
        ax.plot(np.linspace(0, 2*np.pi, 361), np.full(361, r), color=gold_grid, lw=0.6, alpha=0.35, zorder=0)

    # Draw sector wedges
    for i, cat in enumerate(CATEGORIES):
        theta = angles[i]
        score = float(scores.get(cat, 0.0))
        score01 = max(0.0, min(1.0, score / 100.0))

        base_neutral = "#F5F5F5"
        base_color = CATEGORY_COLORS.get(cat, "#2E6AF3")

        # Neutral base ring for each wedge
        ax.bar(
            theta,
            ring_h,
            width=width * 0.98,
            bottom=r_inner,
            color=base_neutral,
            edgecolor=gold_grid,
            linewidth=0.8,
            alpha=0.60,
            align="edge",
            zorder=1,
        )

        # Color overlay: length + intensity respond to score
        fill_h = ring_h * score01
        # Mix toward neutral at low scores; more saturated at high scores
        color_mixed = _mix_hex("#F5F5F5", base_color, 0.35 + 0.65 * score01)
        ax.bar(
            theta,
            fill_h,
            width=width * 0.98,
            bottom=r_inner,
            color=color_mixed,
            edgecolor="none",
            alpha=0.25 + 0.55 * score01,
            align="edge",
            zorder=2,
        )

        # Point marker at score radius (colored)
        ax.scatter(
            [theta + width / 2],
            [r_inner + fill_h],
            s=40,
            c=base_color,
            edgecolors=gold,
            linewidths=0.9,
            zorder=5,
        )

        # Label inside wedge (colored, bold, fits inside)
        label = CATEGORY_LABELS.get(cat, cat)
        # Place label mid-wedge and mid-ring
        ax.text(
            theta + width / 2,
            r_inner + ring_h * 0.52,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=base_color,
            zorder=6,
        )

    # Outer ring outline (gold)
    ax.plot(np.linspace(0, 2*np.pi, 361), np.full(361, r_outer), color=gold, lw=2.0, alpha=0.9, zorder=10)
    ax.plot(np.linspace(0, 2*np.pi, 361), np.full(361, r_inner), color=gold, lw=1.4, alpha=0.65, zorder=10)

    # Limit radius for clean framing
    ax.set_rlim(0, 100)

    return fig

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
def generate_invite_code(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def compute_scores():
    cat_scores = {}
    for cat in CATEGORIES:
        likert_vals = [st.session_state.likert_responses[q] for q in LIKERT_QUESTIONS[cat]]
        assess_vals = [st.session_state.assessment_responses[q] for q in ASSESSMENT_QUESTIONS[cat]]

        baseline = float(np.mean(likert_vals)) * 20.0
        raw = float(np.mean(assess_vals)) * 20.0

        score = (raw / baseline) * 50.0 if baseline > 0 else raw

        if st.session_state.use_mutual:
            mutual = float(np.random.uniform(40, 80))
            score = 0.4 * score + 0.6 * mutual

        cat_scores[cat] = float(np.clip(score, 20, 90))

    weights = np.array([0.15, 0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10], dtype=float)
    rgi = float(np.sum(np.array([cat_scores[c] for c in CATEGORIES], dtype=float) * weights))
    cat_scores["RGI"] = rgi
    st.session_state.scores = cat_scores

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

    
    scores = st.session_state.scores

    # Render the upgraded RQ Wheel (ring style, per-category colors, score-responsive intensity)
    fig = draw_rq_wheel(scores)
    st.pyplot(fig)

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
