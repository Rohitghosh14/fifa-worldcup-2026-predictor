import streamlit as st
import pandas as pd
from utils import load_team_stats, load_model, predict_match, apply_bauhaus_theme

st.set_page_config(page_title="Compare Teams", layout="wide")
apply_bauhaus_theme()   # ← NEW: this page now gets the same theme + toggle

team_stats = load_team_stats()
model = load_model()


wc2026_teams = sorted([
    "Mexico","South Africa","South Korea","Czech Republic","Canada","Bosnia and Herzegovina",
    "Qatar","Switzerland","Brazil","Morocco","Haiti","Scotland","United States","Paraguay",
    "Australia","Turkey","Germany","Curaçao","Ivory Coast","Ecuador","Netherlands","Japan",
    "Sweden","Tunisia","Belgium","Egypt","Iran","New Zealand","Spain","Cape Verde","Saudi Arabia",
    "Uruguay","France","Senegal","Iraq","Norway","Argentina","Algeria","Austria","Jordan",
    "Portugal","DR Congo","Uzbekistan","Colombia","England","Croatia","Ghana","Panama"
])

st.title("Compare Teams")
st.caption("Pick two teams and see the model's predicted outcome (neutral venue, like a real knockout match)")

col1, col2 = st.columns(2)
team_a = col1.selectbox("Team A", wc2026_teams, index=wc2026_teams.index("Brazil"))
team_b = col2.selectbox("Team B", wc2026_teams, index=wc2026_teams.index("France"))

st.divider()

if team_a == team_b:
    st.warning("Please select two different teams.")
else:
    probs = predict_match(model, team_stats, team_a, team_b)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_a} wins", f"{probs['H']:.0%}")
    c2.metric("Draw/Draw → Penalties", f"{probs['D']:.0%}")
    c3.metric(f"{team_b} wins", f"{probs['A']:.0%}")

    chart_data = pd.DataFrame({
        "Outcome": [f"{team_a} Win", "Draw → Penalties", f"{team_b} Win"],
        "Probability": [probs['H'], probs['D'], probs['A']]
    }).set_index("Outcome")

    st.bar_chart(chart_data)

    a_rank = team_stats[team_stats['team']==team_a].iloc[0]['rank']
    b_rank = team_stats[team_stats['team']==team_b].iloc[0]['rank']
    a_form = team_stats[team_stats['team']==team_a].iloc[0]['current_form']
    b_form = team_stats[team_stats['team']==team_b].iloc[0]['current_form']

    st.caption("Note: for knockout-stage matchups, a 'Draw' result means the match would likely go to penalties.")

    st.caption(f"{team_a}: FIFA Rank #{int(a_rank)}, recent form {a_form:.1f}/3.0  |  "
                f"{team_b}: FIFA Rank #{int(b_rank)}, recent form {b_form:.1f}/3.0")