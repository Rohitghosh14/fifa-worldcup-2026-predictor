import pandas as pd
import pickle
import streamlit as st


def load_team_stats():
    return pd.read_csv("data/team_stats.csv")


def load_model():
    with open("models/match_predictor.pkl", "rb") as f:
        return pickle.load(f)

def _raw_predict(model, team_stats, home_team, away_team, neutral=True):
    row_h = team_stats[team_stats['team'] == home_team].iloc[0]
    row_a = team_stats[team_stats['team'] == away_team].iloc[0]
    X_new = pd.DataFrame([{
        'is_home_advantage': 0 if neutral else 1,
        'home_team_form': row_h['current_form'],
        'away_team_form': row_a['current_form'],
        'rank_difference': row_a['rank'] - row_h['rank']
    }])
    probs = model.predict_proba(X_new)[0]
    return dict(zip(model.classes_, probs))


def predict_match(model, team_stats, team_a, team_b, neutral=True):
    if not neutral:
        return _raw_predict(model, team_stats, team_a, team_b, neutral=False)

    # Neutral match: average both label orderings so the result doesn't
    # depend on which team happens to be "Team A" vs "Team B"
    forward = _raw_predict(model, team_stats, team_a, team_b, neutral=True)
    backward = _raw_predict(model, team_stats, team_b, team_a, neutral=True)

    p_a_win = (forward['H'] + backward['A']) / 2
    p_b_win = (forward['A'] + backward['H']) / 2
    p_draw = (forward['D'] + backward['D']) / 2

    return {'H': p_a_win, 'D': p_draw, 'A': p_b_win}


def apply_bauhaus_theme():
    """Injects the Bauhaus / Neo-Brutalist theme, with a dark/light toggle
    that stays in sync across every page via st.session_state."""

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    st.session_state.dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)
    dark_mode = st.session_state.dark_mode

    if dark_mode:
        bg, fg, card_bg, border, shadow = "#1a1a1a", "#f5f0e8", "#242424", "#f5f0e8", "#000000"
    else:
        bg, fg, card_bg, border, shadow = "#f5f0e8", "#1a1a1a", "#ffffff", "#1a1a1a", "#1a1a1a"

    accent_yellow = "#ffcc00"
    accent_red = "#e63b2e"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&family=Inter:wght@400;500;600&display=swap');

    .stApp {{ background-color: {bg}; font-family: 'Inter', sans-serif; }}

    h1 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        color: {fg} !important;
    }}
    h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: {fg} !important;
    }}
    p, span, label, li, .stCaption {{ color: {fg} !important; }}

    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 3px solid {border};
        border-radius: 0px;
        padding: 14px 16px;
        box-shadow: 6px 6px 0px {shadow};
    }}
    div[data-testid="stMetricLabel"] {{
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 11px !important;
        color: {fg} !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: {accent_yellow} !important;
    }}

    div[data-baseweb="select"] > div {{
        border-radius: 0px !important;
        border: none !important;
        border-bottom: 3px solid {border} !important;
        background-color: {card_bg} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 3px solid {border};
    }}

    hr {{ border-top: 2px solid {border} !important; }}

    div[data-testid="stAlert"] {{
        background-color: {accent_red}22;
        border: 3px solid {accent_red};
        border-radius: 0px;
        color: {fg} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    return dark_mode