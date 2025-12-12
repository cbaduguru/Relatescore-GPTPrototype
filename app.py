import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom CSS for brand and mobile-like layout
st.markdown("""
<style>
    .main {
        max-width: 400px;
        margin: 0 auto;
        background-color: #F9F9F9;
        padding: 20px;
    }
    h1, h2 {
        font-family: 'Montserrat', sans-serif;
        color: #2D2D2D;
    }
    p, div {
        font-family: 'Open Sans', sans-serif;
        color: #2D2D2D;
    }
    .stButton > button {
        background-color: #BFAE99;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stSlider > div > div > div {
        background-color: #3A7DDF;
    }
    .insight-card {
        background-color: #FFFFFF;
        border: 1px solid #BFAE99;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Categories from RQ Wheel
categories = [
    "Emotional Awareness",
    "Communication Style",
    "Conflict Tendencies",
    "Attachment Patterns",
    "Empathy & Responsiveness",
    "Self-Insight",
    "Trust & Boundaries",
    "Stability & Consistency"
]

# Sample questions (5 per category, Likert 1-5)
questions = {}
for cat in categories:
    questions[cat] = [
        f"How often do you recognize your emotions in {cat.lower()}?",
        f"How would you rate your ability in {cat.lower()}?",
        f"Do you notice patterns in {cat.lower()}?",
        f"How comfortable are you with {cat.lower()}?",
        f"How does {cat.lower()} affect your relationships?"
    ]

# Session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'consented' not in st.session_state:
    st.session_state.consented = False
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

def home_page():
    st.title("RelateScore")
    st.write("Relationship clarity without exposure.")
    st.write("Founded by Charles Badu")  # From doc

    st.header("Welcome")
    st.write("Your relationships deserve clarity and truth.")  # Microcopy

    if st.button("Start Onboarding"):
        st.session_state.page = 'consent'

def consent_page():
    st.header("Consent & Privacy")
    st.write("Your insights stay private. No sharing. No judgment.")  # Microcopy
    st.write("We never sell or share your information.")
    consent = st.checkbox("I consent to private reflection. I understand data is encrypted and can be erased anytime.")
    if consent:
        if st.button("Proceed to Assessment"):
            st.session_state.consented = True
            st.session_state.page = 'assessment'

def assessment_page():
    st.header("Relational Assessment")
    st.write("Answer with honesty, not perfection.")  # Microcopy

    # Simulate dual consent
    st.write("For mutual reflection, invite partner (simulated here).")
    use_mutual = st.checkbox("Include simulated mutual reflection?")

    for cat in categories:
        st.subheader(cat)
        for q in questions[cat]:
            st.session_state.responses[q] = st.slider(q, 1, 5, 3)

    if st.button("Submit Assessment"):
        # Simulate toxicity filter
        if np.random.rand() < 0.1:  # 10% chance block
            st.error("Input blocked for toxicity. Please revise.")
        else:
            compute_scores(use_mutual)
            generate_insights()
            st.success("Assessment complete!")
            st.session_state.page = 'dashboard'

def compute_scores(use_mutual):
    cat_scores = {}
    for cat in categories:
        cat_vals = [st.session_state.responses[q] for q in questions[cat]]
        self_score = np.mean(cat_vals) * 20  # Normalize to 0-100

        # Simulate mutual
        if use_mutual:
            mutual_score = np.random.uniform(40, 80)  # Simulated partner
            score = 0.5 * self_score + 0.5 * mutual_score  # Weighted
        else:
            score = self_score

        # Outlier dampening (cap extreme)
        score = np.clip(score, 20, 90)

        cat_scores[cat] = score

    # RGI: average of categories with weights (from flowchart simulation)
    weights = [0.15, 0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.1]  # Approximate
    rgi = np.average(list(cat_scores.values()), weights=weights)

    cat_scores['RGI'] = rgi
    st.session_state.scores = cat_scores

def generate_insights():
    insights = []
    for cat, score in st.session_state.scores.items():
        if cat == 'RGI':
            continue
        if score > 70:
            insight = f"This is a strong foundation to build on in {cat}."  # Microcopy
            type_ = "Strength"
        elif score < 40:
            insight = f"This pattern may create misunderstandings in {cat}."  # Microcopy
            type_ = "Blind Spot"
        else:
            insight = f"Balanced area in {cat} with room for awareness."
            type_ = "Neutral"
        insights.append({"category": cat, "type": type_, "description": insight, "suggestion": "Consider experimenting with this new approach."})  # Microcopy
    st.session_state.insights = insights

def dashboard_page():
    st.header("Your Relational Dashboard")
    st.write("Your relational clarity at a glance.")  # Microcopy

    # RQ Wheel (Radar Chart)
    scores = st.session_state.scores
    labels = np.array(categories)
    values = np.array([scores[cat] for cat in categories])
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#BFAE99', alpha=0.25)
    ax.plot(angles, values, color='#3A7DDF', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    st.pyplot(fig)

    st.write(f"Your RGI: {scores['RGI']:.1f}")

    st.header("Key Insights")
    for insight in st.session_state.insights:
        with st.container():
            st.markdown(f"""
            <div class="insight-card">
                <h3>{insight['category']}: {insight['type']}</h3>
                <p>{insight['description']}</p>
                <p><i>Suggestion: {insight['suggestion']}</i></p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("New Reflection"):
        st.session_state.page = 'assessment'

# Page router
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'consent':
    consent_page()
elif st.session_state.page == 'assessment' and st.session_state.consented:
    assessment_page()
elif st.session_state.page == 'dashboard' and st.session_state.scores:
    dashboard_page()
else:
    st.error("Please complete onboarding.")
    st.session_state.page = 'home'