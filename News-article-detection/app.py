"""
app.py — Streamlit News Article Classifier Interface.
Run: streamlit run app.py
"""
import os
import sys
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Allow project-level imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    BEST_MODEL_PATH, LR_MODEL_PATH, NB_MODEL_PATH,
    TFIDF_VECTORIZER_PATH, METRICS_PATH, LABEL_COLORS
)
from src.utils import (
    load_artifact, load_metrics, predict_single, get_top_features
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — News Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e2e8f0;
  }

  /* Hero header */
  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
  }

  /* Prediction card */
  .pred-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 1.5rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
  }
  .pred-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 2px solid #ef4444;
    color: #fca5a5;
  }
  .pred-real {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(21,128,61,0.1));
    border: 2px solid #22c55e;
    color: #86efac;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.4rem 0;
  }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
  .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }

  /* Sidebar */
  .css-1d391kg { background: rgba(15,12,41,0.9) !important; }

  /* Text area */
  textarea { background: rgba(255,255,255,0.05) !important; color: #e2e8f0 !important; }

  /* Button */
  .stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.5);
  }

  .section-header {
    font-size: 1.15rem; font-weight: 700; color: #c4b5fd;
    border-left: 4px solid #7c3aed; padding-left: 0.6rem;
    margin-top: 1.5rem; margin-bottom: 0.8rem;
  }
  .info-box {
    background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.3);
    border-radius: 10px; padding: 1rem; color: #bfdbfe; font-size: 0.9rem;
    margin-top: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load vectorizer + both classifiers (cached across sessions)."""
    vectorizer = load_artifact(TFIDF_VECTORIZER_PATH)
    lr_model   = load_artifact(LR_MODEL_PATH)
    nb_model   = load_artifact(NB_MODEL_PATH)
    best_model = load_artifact(BEST_MODEL_PATH)
    return vectorizer, lr_model, nb_model, best_model


def models_available() -> bool:
    files = [TFIDF_VECTORIZER_PATH, LR_MODEL_PATH, NB_MODEL_PATH, BEST_MODEL_PATH]
    return all(os.path.exists(f) for f in files)


def gauge_chart(confidence: float, label: str):
    color = LABEL_COLORS.get(label, "#7c3aed")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":  {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(255,255,255,0.03)"},
                {"range": [50, 80], "color": "rgba(255,255,255,0.05)"},
                {"range": [80, 100],"color": "rgba(255,255,255,0.08)"},
            ],
        },
        title={"text": "Confidence", "font": {"color": "#94a3b8", "size": 14}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def feature_bar_chart(features: list):
    words   = [f[0] for f in features]
    weights = [f[1] for f in features]
    colors  = ["#22c55e" if w > 0 else "#ef4444" for w in weights]

    fig = go.Figure(go.Bar(
        x=weights, y=words, orientation="h",
        marker_color=colors,
        text=[f"{w:+.3f}" for w in weights],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top Influential Words (green → REAL, red → FAKE)",
        title_font={"color": "#c4b5fd", "size": 13},
        xaxis_title="LR Coefficient Weight",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis={"gridcolor": "rgba(255,255,255,0.07)"},
        yaxis={"autorange": "reversed"},
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ FakeShield")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    selected_model_name = st.selectbox(
        "Active Model",
        ["Best (Auto)", "Logistic Regression", "Naive Bayes"],
        help="Choose which classifier to use for prediction."
    )
    show_features = st.toggle("Show word-level explanation", value=True)
    st.markdown("---")

    # Metrics
    st.markdown("### 📊 Model Performance")
    if os.path.exists(METRICS_PATH):
        metrics = load_metrics(METRICS_PATH)
        for model_key in ["Logistic Regression", "Naive Bayes"]:
            if model_key in metrics:
                m = metrics[model_key]
                st.markdown(f"**{model_key}**")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy",    f"{m.get('accuracy', 0):.2f}%")
                col2.metric("F1 Weighted", f"{m.get('f1_weighted', 0):.2f}%")
    else:
        st.info("Run `python train.py` to see metrics here.")

    st.markdown("---")
    st.markdown(
        '<div class="info-box">📁 Place <b>Fake.csv</b> & <b>True.csv</b> in the '
        '<code>data/</code> folder, then run <code>python train.py</code> to train.</div>',
        unsafe_allow_html=True
    )


# ─── Main Area ────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🛡️ FakeShield</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered News Authenticity Classifier — '
    'Powered by NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# Check model availability
if not models_available():
    st.error(
        "⚠️ **Trained model files not found.**\n\n"
        "Please:\n"
        "1. Download the dataset: `python data/download_data.py`\n"
        "2. Train the models: `python train.py`\n"
        "3. Refresh this page."
    )
    st.stop()

# Load models (cached)
with st.spinner("Loading models …"):
    vectorizer, lr_model, nb_model, best_model = load_models()

model_map = {
    "Best (Auto)":         best_model,
    "Logistic Regression": lr_model,
    "Naive Bayes":         nb_model,
}
active_model = model_map[selected_model_name]

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📰 Paste Your News Article</div>', unsafe_allow_html=True)

col_input, col_examples = st.columns([3, 1])
with col_examples:
    st.markdown("**Quick Examples**")
    if st.button("🟢 Real sample", key="btn_real"):
        st.session_state["article_text"] = (
            "Washington (Reuters) — The U.S. Federal Reserve raised interest rates by "
            "25 basis points on Wednesday, signalling continued commitment to bringing "
            "inflation down toward its 2% target, as labour market data remained robust."
        )
    if st.button("🔴 Fake sample", key="btn_fake"):
        st.session_state["article_text"] = (
            "BREAKING: Scientists confirm that drinking bleach cures COVID-19 overnight. "
            "Government officials are hiding this miracle cure to protect Big Pharma profits. "
            "Share this before it gets deleted! The deep state doesn't want you to know the truth!"
        )

with col_input:
    article_text = st.text_area(
        "Enter full article text (headline + body for best results):",
        value=st.session_state.get("article_text", ""),
        height=220,
        placeholder="Paste the news article text here …",
        key="article_text",
        label_visibility="collapsed",
    )

word_count = len(article_text.split()) if article_text.strip() else 0
st.caption(f"Word count: **{word_count}**  |  Model: **{selected_model_name}**")

predict_btn = st.button("🔍 Analyse Article", key="analyse_btn")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not article_text.strip():
        st.warning("Please paste an article before clicking Analyse.")
    elif word_count < 10:
        st.warning("Please provide at least 10 words for reliable classification.")
    else:
        with st.spinner("Analysing …"):
            result   = predict_single(article_text, active_model, vectorizer)
            features = get_top_features(article_text, lr_model, vectorizer, top_n=12) \
                       if show_features else []

        label = result["label"]
        conf  = result["confidence"]

        # ── Verdict Card ──────────────────────────────────────────
        emoji = "🚨" if label == "FAKE" else "✅"
        css_cls = "pred-fake" if label == "FAKE" else "pred-real"
        verdict_msg = (
            "This article shows strong indicators of misinformation." if label == "FAKE"
            else "This article appears to be credible and fact-based."
        )
        st.markdown(
            f'<div class="pred-card {css_cls}">'
            f'{emoji}  {label} NEWS<br>'
            f'<span style="font-size:1rem;font-weight:400;color:#cbd5e1">{verdict_msg}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Metrics Row ───────────────────────────────────────────
        col_g, col_m1, col_m2 = st.columns([1.2, 1, 1])

        with col_g:
            st.plotly_chart(gauge_chart(conf, label), use_container_width=True)

        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{result['probabilities']['REAL']:.1f}%</div>
              <div class="metric-label">REAL probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{result['probabilities']['FAKE']:.1f}%</div>
              <div class="metric-label">FAKE probability</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability Bar ───────────────────────────────────────
        st.markdown('<div class="section-header">📈 Probability Breakdown</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Class": ["REAL", "FAKE"],
            "Probability (%)": [
                result["probabilities"]["REAL"],
                result["probabilities"]["FAKE"],
            ]
        })
        fig_bar = px.bar(
            prob_df, x="Probability (%)", y="Class", orientation="h",
            color="Class",
            color_discrete_map={"REAL": "#22c55e", "FAKE": "#ef4444"},
            text="Probability (%)", range_x=[0, 100],
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=160,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            xaxis={"gridcolor": "rgba(255,255,255,0.07)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Feature Explanation ───────────────────────────────────
        if show_features and features:
            st.markdown('<div class="section-header">🔍 Word-Level Explanation (LR)</div>', unsafe_allow_html=True)
            st.plotly_chart(feature_bar_chart(features), use_container_width=True)
            st.caption(
                "Words with **positive** coefficients push the prediction toward REAL; "
                "**negative** coefficients push toward FAKE. Only words present in this article are shown."
            )
        elif show_features and selected_model_name == "Naive Bayes":
            st.info("💡 Word-level explanations are only available for Logistic Regression. Switch the model in the sidebar.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:0.8rem;">'
    'FakeShield · News Article Classifier · Built with Streamlit, scikit-learn & NLTK'
    '</div>',
    unsafe_allow_html=True
)
