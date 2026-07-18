import streamlit as st
import pandas as pd
from utils import load_team_stats, load_model, predict_match, apply_bauhaus_theme

st.set_page_config(page_title="FIFA 2026 AI", layout="wide")
apply_bauhaus_theme()   

team_stats = load_team_stats()
model = load_model()

# ---------- Title ----------
st.title("FIFA World Cup 2026 — AI Predictions")
st.caption("A machine learning model trained on real historical international football results.")

# ---------- Top stat tiles ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("2026 World Cup Teams", "48")
col2.metric("Training Matches", "49,505")
col3.metric("Live 2026 Validation Accuracy", "55%")
col4.metric("Algorithm", "Logistic Regression")

st.divider()

# ---------- Top Contenders table ----------
st.subheader("Top Contenders — 2026 World Cup")
st.caption("Ranked by FIFA World Ranking + current form (not a full tournament simulation)")
st.caption("Recent Form scale: 0 = lost all last 5 matches, 3 = won all last 5 matches")

wc2026_teams = [
    "Mexico","South Africa","South Korea","Czech Republic","Canada","Bosnia and Herzegovina",
    "Qatar","Switzerland","Brazil","Morocco","Haiti","Scotland","United States","Paraguay",
    "Australia","Turkey","Germany","Curaçao","Ivory Coast","Ecuador","Netherlands","Japan",
    "Sweden","Tunisia","Belgium","Egypt","Iran","New Zealand","Spain","Cape Verde","Saudi Arabia",
    "Uruguay","France","Senegal","Iraq","Norway","Argentina","Algeria","Austria","Jordan",
    "Portugal","DR Congo","Uzbekistan","Colombia","England","Croatia","Ghana","Panama"
]

contenders = team_stats[team_stats['team'].isin(wc2026_teams)].dropna(subset=['rank'])
contenders = contenders.sort_values('rank').head(10)

st.dataframe(
    contenders.rename(columns={'team': 'Team', 'rank': 'FIFA Rank', 'current_form': 'Recent Form'}),
    hide_index=True,
    use_container_width=True
)

st.divider()

# ---------- World Cup Final prediction ----------                     # now calls predict_match(model, team_stats, ...)
st.subheader("World Cup Final — Tomorrow, July 19")
st.caption("Real prediction from our trained model for the actual final")
st.caption("⚠️ This is a knockout match — a 'Draw' result means the match likely goes to penalties, not a final tied score.")

final_probs = predict_match(model, team_stats, "Spain", "Argentina")
c1, c2, c3 = st.columns(3)
c1.metric("Spain wins", f"{final_probs['H']:.0%}")
c2.metric("Draw → Penalties", f"{final_probs['D']:.0%}")
c3.metric("Argentina wins", f"{final_probs['A']:.0%}")

st.caption("Note: for knockout-stage matchups, a 'Draw' result means the match would likely go to penalties.")

st.divider()

# ---------- Third-place match prediction ----------
st.subheader("Third-Place Match — Today, July 18")
st.caption("Real prediction from our trained model for the actual match")
st.caption("⚠️ This is a knockout match — a 'Draw' result means the match likely goes to penalties, not a final tied score.")

third_probs = predict_match(model, team_stats, "France", "England")
c1, c2, c3 = st.columns(3)
c1.metric("France wins", f"{third_probs['H']:.0%}")
c2.metric("Draw → Penalties", f"{third_probs['D']:.0%}")
c3.metric("England wins", f"{third_probs['A']:.0%}")

st.caption("Note: for knockout-stage matchups, a 'Draw' result means the match would likely go to penalties.")

st.divider()
st.subheader("Head-to-Head Predictions")
st.info("Use **Compare Teams** in the sidebar to pick two teams and see the predicted outcome.")