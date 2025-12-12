
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_NAME = "RelateScore™ Prototype"
MISSION = "RelateScore™ provides private relational clarity that supports growth without judgment or exposure."

DATA_DIR = ".data"
INVITES_PATH = os.path.join(DATA_DIR, "invites.json")

# -----------------------------
# Storage helpers (JSON file)
# -----------------------------
def _ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INVITES_PATH):
        with open(INVITES_PATH, "w", encoding="utf-8") as f:
            json.dump({"invites": {}}, f, indent=2)

def _load_store() -> Dict[str, Any]:
    _ensure_storage()
    with open(INVITES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_store(store: Dict[str, Any]) -> None:
    _ensure_storage()
    tmp = INVITES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    os.replace(tmp, INVITES_PATH)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -----------------------------
# Invite & user identity
# -----------------------------
def _rand_code(n: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    import secrets
    return "".join(secrets.choice(alphabet) for _ in range(n))

def get_or_create_user_id() -> str:
    if "user_id" not in st.session_state:
        import secrets
        st.session_state.user_id = secrets.token_hex(8)
    return st.session_state.user_id

# -----------------------------
# Safety / Toxicity filter (simple heuristic)
# NOTE: Prototype only — replace with robust moderation in production.
# -----------------------------
BANNED_PATTERNS = [
    r"\bkill\b", r"\bdie\b", r"\bworthless\b", r"\bhate you\b", r"\bshut up\b",
    r"\bstupid\b", r"\bidiot\b", r"\bslut\b", r"\bbitch\b", r"\basshole\b"
]
def toxicity_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for p in BANNED_PATTERNS if re.search(p, t))
    # Also add small penalty for excessive ALL CAPS yelling
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    return min(1.0, 0.18 * hits + 0.3 * max(0.0, caps_ratio - 0.25))

def enforce_toxicity_gate(texts: List[str], threshold: float = 0.55) -> Tuple[bool, float]:
    score = max(toxicity_score(t) for t in texts)
    return (score <= threshold), score

# -----------------------------
# Scoring logic (prototype)
# Flowchart alignment:
# - Normalize text to 0–100
# - Apply category weights (communication 35%, empathy 20%, etc.)
# - Weight mutual reflection heavier
# - Outlier dampening: trimmed mean/median blend
# - Time weighting: EMA alpha 0.3–0.8 across multiple reflections
# -----------------------------
CATEGORIES = ["Communication", "Empathy", "Reliability", "Conflict Navigation", "Connection"]

DEFAULT_WEIGHTS = {
    "Communication": 0.35,
    "Empathy": 0.20,
    "Reliability": 0.20,
    "Conflict Navigation": 0.15,
    "Connection": 0.10,
}

PROMPTS = [
    "In the past week, what did your partner do that helped you feel supported?",
    "What is one moment you felt misunderstood, and what did you need instead?",
    "What is one small change you can make next week to improve the relationship?"
]

EMPATHY_POS = ["understand", "heard", "care", "empath", "support", "validate", "compassion"]
COMM_POS = ["talk", "discuss", "communicat", "listen", "clarify", "share", "ask"]
REL_POS = ["reliable", "consistent", "follow through", "depend", "trust", "kept", "promise"]
CONN_POS = ["close", "love", "affection", "connect", "bond", "together", "intim"]
CONFLICT_SKILL = ["calm", "repair", "apolog", "resolve", "compromise", "boundary", "respect"]
CONFLICT_NEG = ["always", "never", "ignore", "silent", "yell", "fight", "blame"]

def _keyword_score(text: str, keywords: List[str]) -> float:
    t = (text or "").lower()
    return sum(1 for k in keywords if k in t)

def _len_score(text: str) -> float:
    # rewards reflection depth but caps
    n = len((text or "").strip())
    return min(1.0, n / 500.0)

def score_categories(effort_1_5: int, answers: List[str], attachment_flags: Dict[str, bool]) -> Dict[str, float]:
    joined = " ".join(answers or [])
    effort = np.clip((effort_1_5 - 1) / 4, 0, 1)  # 0..1

    # Base 0..1 signals
    comm = 0.35 * _len_score(joined) + 0.35 * np.tanh(_keyword_score(joined, COMM_POS) / 3) + 0.30 * effort
    emp  = 0.35 * _len_score(joined) + 0.35 * np.tanh(_keyword_score(joined, EMPATHY_POS) / 3) + 0.30 * effort
    rel  = 0.25 * _len_score(joined) + 0.45 * np.tanh(_keyword_score(joined, REL_POS) / 3) + 0.30 * effort
    conn = 0.25 * _len_score(joined) + 0.45 * np.tanh(_keyword_score(joined, CONN_POS) / 3) + 0.30 * effort

    conflict_pos = np.tanh(_keyword_score(joined, CONFLICT_SKILL) / 3)
    conflict_neg = np.tanh(_keyword_score(joined, CONFLICT_NEG) / 3)
    conflict = 0.35 * _len_score(joined) + 0.35 * conflict_pos + 0.30 * effort - 0.25 * conflict_neg

    # Attachment indicators (optional) — used as small modifiers
    # Flags: anxious, avoidant, secure
    if attachment_flags.get("secure"):
        comm += 0.04; emp += 0.04; rel += 0.04; conn += 0.04; conflict += 0.04
    if attachment_flags.get("anxious"):
        conflict -= 0.04
    if attachment_flags.get("avoidant"):
        conn -= 0.04
        comm -= 0.02

    raw = {
        "Communication": comm,
        "Empathy": emp,
        "Reliability": rel,
        "Conflict Navigation": conflict,
        "Connection": conn,
    }
    # Normalize to 0..100
    norm = {k: float(np.clip(v, 0, 1) * 100) for k, v in raw.items()}
    return norm

def rsq_from_categories(cat_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(cat_scores[c] * weights.get(c, 0) for c in CATEGORIES))

def trimmed_mean(values: List[float], trim: float = 0.1) -> float:
    if not values:
        return 0.0
    v = np.array(values, dtype=float)
    if len(v) < 5:
        return float(v.mean())
    k = int(len(v) * trim)
    v_sorted = np.sort(v)
    v_trim = v_sorted[k:len(v_sorted)-k] if (len(v_sorted) - 2*k) > 0 else v_sorted
    return float(v_trim.mean())

def median_blend(values: List[float]) -> float:
    if not values:
        return 0.0
    v = np.array(values, dtype=float)
    return float(0.6 * np.median(v) + 0.4 * np.mean(v))

def dampened_aggregate(values: List[float]) -> float:
    return float(0.5 * trimmed_mean(values, 0.1) + 0.5 * median_blend(values))

def ema(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    a = float(np.clip(alpha, 0.01, 0.99))
    s = values[0]
    for x in values[1:]:
        s = a * x + (1 - a) * s
    return float(s)

def compute_dashboard(reflections: List[Dict[str, Any]], weights: Dict[str, float], ema_alpha: float) -> Dict[str, Any]:
    """Return aggregates (RSQ and category scores) with outlier dampening + EMA across time."""
    if not reflections:
        return {
            "rsq_point": 0.0,
            "rsq_trend": 0.0,
            "category_point": {c: 0.0 for c in CATEGORIES},
            "category_trend": {c: 0.0 for c in CATEGORIES},
            "n_reflections": 0
        }

    # time order
    refl = sorted(reflections, key=lambda r: r.get("ts", ""))
    rsqs = [float(r.get("rsq", 0.0)) for r in refl]
    cats_by_c = {c: [float(r.get("categories", {}).get(c, 0.0)) for r in refl] for c in CATEGORIES}

    # Point estimate = dampened aggregate of last up to 10 entries
    tail_n = 10
    rsq_point = dampened_aggregate(rsqs[-tail_n:])
    category_point = {c: dampened_aggregate(vals[-tail_n:]) for c, vals in cats_by_c.items()}

    # Trend = EMA on entire history
    rsq_trend = ema(rsqs, ema_alpha)
    category_trend = {c: ema(vals, ema_alpha) for c, vals in cats_by_c.items()}

    return {
        "rsq_point": rsq_point,
        "rsq_trend": rsq_trend,
        "category_point": category_point,
        "category_trend": category_trend,
        "n_reflections": len(reflections)
    }

# -----------------------------
# UI / Branding
# -----------------------------
def inject_branding():
    st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
        :root{
          --rs-primary:#2C2A4A;
          --rs-accent:#3B82F6;
          --rs-soft:#F4F6FB;
          --rs-text:#0F172A;
          --rs-muted:#475569;
          --rs-border:#E2E8F0;
        }
        .rs-shell{background:var(--rs-soft); padding:18px 18px 6px 18px; border-radius:18px; border:1px solid var(--rs-border);}
        .rs-title{font-size:28px; font-weight:700; color:var(--rs-primary); margin-bottom:6px;}
        .rs-sub{color:var(--rs-muted); margin-top:0px; margin-bottom:0px;}
        .rs-card{background:white; padding:16px; border-radius:18px; border:1px solid var(--rs-border);}
        .rs-small{color:var(--rs-muted); font-size:13px;}
        .rs-footer{color:var(--rs-muted); font-size:12px; padding-top:20px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def footer():
    st.markdown(f"<div class='rs-footer'>{MISSION}<br/>Prototype for demonstration only — no clinical, legal, or safety guarantees.</div>", unsafe_allow_html=True)

# -----------------------------
# App views
# -----------------------------
def view_home(store: Dict[str, Any]):
    st.markdown("<div class='rs-shell'>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.75, 0.25], vertical_alignment="center")
    with col1:
        st.markdown(f"<div class='rs-title'>{APP_NAME}</div>", unsafe_allow_html=True)
        st.markdown("<p class='rs-sub'>Private, structured reflection with a lightweight scorecard. No raw text is shared by default.</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.markdown("**Logo slot**")
        st.caption("Replace with your brand asset later.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Start a new reflection thread")
        initiator = st.text_input("Your name (or alias)", value=st.session_state.get("name", ""))
        partner = st.text_input("Partner name (or alias)", value=st.session_state.get("partner_name", ""))
        st.session_state["name"] = initiator
        st.session_state["partner_name"] = partner

        if st.button("Create invite", type="primary"):
            code = _rand_code()
            user_id = get_or_create_user_id()
            store["invites"][code] = {
                "code": code,
                "created_at": _now_iso(),
                "initiator_name": initiator.strip() or "Initiator",
                "partner_name": partner.strip() or "Partner",
                "consents": {user_id: True},  # initiator consents by creating
                "withdrawn": False,
                "reflections": [],
                "toxicity_events": 0,
            }
            _save_store(store)
            st.success("Invite created.")
            st.info(f"Share this code with your partner: **{code}**")
            st.caption("Either party can withdraw consent; withdrawal clears the thread.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Join with an invite code")
        code = st.text_input("Invite code", value=st.session_state.get("code", "")).strip().upper()
        st.session_state["code"] = code
        if st.button("Join"):
            if code in store["invites"] and not store["invites"][code].get("withdrawn"):
                st.success("Invite found. Continue to Consent & Reflect in the sidebar.")
                st.session_state["active_code"] = code
            else:
                st.error("Invite code not found (or has been withdrawn).")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    footer()

def view_consent(store: Dict[str, Any], code: str):
    inv = store["invites"].get(code)
    if not inv or inv.get("withdrawn"):
        st.error("This invite is not available.")
        return

    st.markdown("<div class='rs-shell'>", unsafe_allow_html=True)
    st.markdown(f"<div class='rs-title'>Thread: {code}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='rs-sub'>Initiator: <b>{inv.get('initiator_name')}</b> • Partner: <b>{inv.get('partner_name')}</b></p>", unsafe_allow_html=True)

    user_id = get_or_create_user_id()

    st.divider()
    st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
    st.subheader("Consent checkpoint (dual consent required)")
    st.write("RelateScore only processes reflections if both parties consent. Raw text is not shared by default.")

    current = inv.get("consents", {})
    you_consented = bool(current.get(user_id, False))
    consent = st.checkbox("I consent to participate in this thread", value=you_consented)

    if st.button("Save consent", type="primary"):
        inv["consents"][user_id] = bool(consent)
        store["invites"][code] = inv
        _save_store(store)
        st.success("Consent saved.")

    # Dual consent status
    dual = len([k for k, v in inv.get("consents", {}).items() if v]) >= 2
    if dual:
        st.success("Dual consent obtained. You may proceed to Reflection.")
    else:
        st.warning("Waiting on dual consent. Invite the other party to join and consent.")

    st.divider()
    st.subheader("Withdraw consent (clears data)")
    st.caption("Withdrawal ends processing and deletes stored reflections for this invite code.")
    if st.button("Withdraw and clear thread", type="secondary"):
        inv["withdrawn"] = True
        inv["reflections"] = []
        store["invites"][code] = inv
        _save_store(store)
        st.success("Thread withdrawn and cleared.")
        st.session_state.pop("active_code", None)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    footer()

def view_reflection(store: Dict[str, Any], code: str):
    inv = store["invites"].get(code)
    if not inv or inv.get("withdrawn"):
        st.error("This invite is not available.")
        return

    user_id = get_or_create_user_id()
    consents = inv.get("consents", {})
    dual = len([k for k, v in consents.items() if v]) >= 2
    if not dual:
        st.warning("Dual consent is not yet obtained. Please complete Consent first.")
        return

    st.markdown("<div class='rs-shell'>", unsafe_allow_html=True)
    st.markdown(f"<div class='rs-title'>Reflection</div>", unsafe_allow_html=True)
    st.markdown("<p class='rs-sub'>Guided prompts + quick self-rating. A toxicity gate blocks harmful language.</p>", unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Inputs")
        effort = st.slider("Self-rating: effort & intent to improve (1–5)", 1, 5, 3)

        answers = []
        for i, p in enumerate(PROMPTS, start=1):
            answers.append(st.text_area(f"Prompt {i}", placeholder=p, height=110, key=f"ans_{i}"))

        st.caption("Optional: attachment-style indicators (for small scoring modifiers).")
        c1, c2, c3 = st.columns(3)
        with c1:
            anxious = st.checkbox("Anxious", value=False)
        with c2:
            avoidant = st.checkbox("Avoidant", value=False)
        with c3:
            secure = st.checkbox("Secure", value=False)

        attachments = {"anxious": anxious, "avoidant": avoidant, "secure": secure}

        toxicity_ok, tox_score = enforce_toxicity_gate([*answers], threshold=0.55)
        if not toxicity_ok:
            st.error(f"Input blocked by the toxicity gate (score {tox_score:.2f} > threshold). Please revise and resubmit.")
            # Count toxicity events (for red-flag dashboard)
            if st.button("Acknowledge & return to input"):
                inv["toxicity_events"] = int(inv.get("toxicity_events", 0)) + 1
                store["invites"][code] = inv
                _save_store(store)
        else:
            if st.button("Submit reflection", type="primary"):
                cat_scores = score_categories(effort, answers, attachments)
                rsq = rsq_from_categories(cat_scores, DEFAULT_WEIGHTS)

                inv["reflections"].append({
                    "ts": _now_iso(),
                    "user_id": user_id,
                    "effort": int(effort),
                    "answers": answers,  # stored for *your* history; not shared in UI
                    "attachments": attachments,
                    "categories": cat_scores,
                    "rsq": rsq,
                })
                store["invites"][code] = inv
                _save_store(store)
                st.success("Reflection saved.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Weights & time curve")
        st.caption("Prototype weights match the flowchart example: Communication 35%, Empathy 20%, Reliability 20%, Conflict 15%, Connection 10%.")
        ema_alpha = st.slider("Time-weighting alpha (EMA)", 0.30, 0.80, 0.50, 0.05, help="Higher = recent reflections influence more.")
        st.session_state["ema_alpha"] = ema_alpha
        st.divider()
        st.subheader("Privacy defaults")
        st.write("• Your raw text stays private in this prototype UI.\n• Dashboards show aggregates only.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    footer()

def view_dashboard(store: Dict[str, Any], code: str):
    inv = store["invites"].get(code)
    if not inv or inv.get("withdrawn"):
        st.error("This invite is not available.")
        return

    user_id = get_or_create_user_id()
    consents = inv.get("consents", {})
    dual = len([k for k, v in consents.items() if v]) >= 2
    if not dual:
        st.warning("Dual consent is not yet obtained. Please complete Consent first.")
        return

    ema_alpha = float(st.session_state.get("ema_alpha", 0.50))

    # reflections are not separated for aggregate scoring; "mutual reflection heavier" is implemented as:
    # if both parties have reflections within the last 7 days, apply a 1.10 multiplier on RSQ point estimate.
    refl = inv.get("reflections", [])
    dashboard = compute_dashboard(refl, DEFAULT_WEIGHTS, ema_alpha)

    # Mutual weighting check
    recent = sorted(refl, key=lambda r: r.get("ts", ""))[-10:]
    # identify unique users in last 7 days
    week_ago = datetime.now(timezone.utc).timestamp() - 7 * 86400
    users_recent = set()
    for r in recent:
        try:
            ts = datetime.fromisoformat(r["ts"]).timestamp()
        except Exception:
            ts = 0
        if ts >= week_ago:
            users_recent.add(r.get("user_id"))
    mutual_bonus = 1.10 if len(users_recent) >= 2 else 1.00

    rsq_point = float(np.clip(dashboard["rsq_point"] * mutual_bonus, 0, 100))
    rsq_trend = float(np.clip(dashboard["rsq_trend"] * mutual_bonus, 0, 100))

    st.markdown("<div class='rs-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='rs-title'>Private Output</div>", unsafe_allow_html=True)
    st.markdown("<p class='rs-sub'>RSQ score, insights, and a lightweight red-flag dashboard.</p>", unsafe_allow_html=True)
    st.divider()

    top1, top2, top3 = st.columns(3)
    top1.metric("RSQ (Dampened point estimate)", f"{rsq_point:0.1f}", help="Outlier dampening: trimmed mean + median blend (last 10).")
    top2.metric("RSQ (EMA trend)", f"{rsq_trend:0.1f}", help="Time weighting across history.")
    top3.metric("Reflections captured", f"{dashboard['n_reflections']}")

    st.divider()

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Category scorecard (0–100)")
        df = pd.DataFrame({
            "Category": CATEGORIES,
            "Point": [dashboard["category_point"][c] for c in CATEGORIES],
            "Trend": [dashboard["category_trend"][c] for c in CATEGORIES],
            "Weight": [DEFAULT_WEIGHTS[c] for c in CATEGORIES],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("Scores are indicative signals from text length + keyword heuristics + self-rated effort. Replace with calibrated model later.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='rs-card' style='margin-top:14px;'>", unsafe_allow_html=True)
        st.subheader("Growth hints (non-judgmental)")
        lowest = df.sort_values("Point").head(2)["Category"].tolist()
        tips = []
        if "Communication" in lowest:
            tips.append("Try a 10-minute weekly check-in: one appreciation, one request, one boundary.")
        if "Empathy" in lowest:
            tips.append("Use reflective listening: 'What I’m hearing is… Did I get that right?'")
        if "Reliability" in lowest:
            tips.append("Make one small promise only if you can keep it; track it in a shared note.")
        if "Conflict Navigation" in lowest:
            tips.append("Use a repair phrase: 'I want to get back on the same team—can we reset?'")
        if "Connection" in lowest:
            tips.append("Schedule one low-pressure connection ritual (walk, coffee, device-free dinner).")
        for t in tips[:3]:
            st.write(f"• {t}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        st.subheader("Red-flag dashboard")
        tox_events = int(inv.get("toxicity_events", 0))
        st.write(f"• Toxicity gate triggers: **{tox_events}**")

        # Simple risk flags from last 5 reflections
        last = sorted(refl, key=lambda r: r.get("ts", ""))[-5:]
        conflict_vals = [r.get("categories", {}).get("Conflict Navigation", 0.0) for r in last]
        empathy_vals = [r.get("categories", {}).get("Empathy", 0.0) for r in last]
        low_conflict_skill = sum(1 for v in conflict_vals if float(v) < 35)
        low_empathy = sum(1 for v in empathy_vals if float(v) < 35)

        st.write(f"• Low conflict-navigation signals (last 5): **{low_conflict_skill}**")
        st.write(f"• Low empathy signals (last 5): **{low_empathy}**")

        if tox_events >= 1 or low_conflict_skill >= 3:
            st.warning("This thread shows elevated friction signals. Consider pausing and using calmer, specific language.")
        else:
            st.success("No elevated red-flag signals detected in the latest window (prototype heuristic).")

        st.divider()
        st.subheader("Consent control")
        st.caption("Either party can withdraw and clear the thread at any time.")
        if st.button("Withdraw and clear thread", type="secondary"):
            inv["withdrawn"] = True
            inv["reflections"] = []
            store["invites"][code] = inv
            _save_store(store)
            st.success("Thread withdrawn and cleared.")
            st.session_state.pop("active_code", None)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
    st.subheader("Your reflection history (private)")
    st.caption("Shows only reflections created from this browser session (matched by your user_id).")
    mine = [r for r in refl if r.get("user_id") == user_id]
    if mine:
        hist = pd.DataFrame([{
            "Timestamp (UTC)": r.get("ts",""),
            "Effort": r.get("effort"),
            "RSQ": round(float(r.get("rsq", 0.0)), 1),
            **{c: round(float(r.get("categories", {}).get(c, 0.0)), 1) for c in CATEGORIES}
        } for r in sorted(mine, key=lambda r: r.get("ts",""), reverse=True)])
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.info("No reflections from this session yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    footer()

# -----------------------------
# Router
# -----------------------------
def main():
    inject_branding()
    store = _load_store()
    code = st.session_state.get("active_code") or st.session_state.get("code") or ""

    st.sidebar.title("RelateScore™")
    st.sidebar.caption("Prototype navigation")
    page = st.sidebar.radio("Go to", ["Home", "Consent", "Reflection", "Dashboard"], index=0)

    if page == "Home":
        view_home(store)
        return

    # For other pages, ensure a code is available
    if not code:
        st.sidebar.info("Join or create an invite code on Home first.")
        view_home(store)
        return

    st.sidebar.markdown(f"**Active code:** `{code}`")
    st.sidebar.caption("Tip: invite your partner to join with the same code.")

    if page == "Consent":
        view_consent(store, code)
    elif page == "Reflection":
        view_reflection(store, code)
    elif page == "Dashboard":
        view_dashboard(store, code)
    else:
        view_home(store)

if __name__ == "__main__":
    main()
