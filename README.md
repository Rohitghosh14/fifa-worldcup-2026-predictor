# FIFA World Cup 2026 — AI Match Predictor

A machine learning project that predicts football match outcomes (Win / Draw / Loss)
using real historical international results, then validates its predictions against
the **actual, live 2026 FIFA World Cup**.

Built end-to-end from scratch: raw data → feature engineering → model training →
evaluation → live tournament validation → a deployed Streamlit web app.

---

## What it does

- Predicts the outcome of any match between two international football teams,
  with Win / Draw / Loss probabilities
- Trained on **49,505 real historical matches** (1872–2026)
- Validated against **102 real, unseen 2026 World Cup matches** — genuine
  out-of-sample performance, not just a random train/test split
- Two-page Streamlit app: a live dashboard (including a real prediction for the
  2026 World Cup Final) and a head-to-head team comparison tool

---

## How it works

### 1. Data
Historical match results come from a public dataset of international football
results (1872–present). Team strength is estimated using a FIFA World Ranking
snapshot. Team names are reconciled between the two sources where they differ
(e.g. `"South Korea"` vs `"Korea Republic"`).

### 2. Features
Four features are engineered from the raw data:

| Feature | What it captures |
|---|---|
| `is_home_advantage` | Whether the match was played on a team's home turf (vs. neutral ground) |
| `home_team_form` / `away_team_form` | Each team's average points from their last 5 matches (momentum) |
| `rank_difference` | The gap between the two teams' FIFA World Ranking positions (overall quality) |

### 3. Models

**Logistic Regression** — a linear classification algorithm. It looks at each
feature, learns how much weight to give it, and combines them into a probability
for each outcome (Home Win / Draw / Away Win). Simple, fast, and easy to
interpret — a strong baseline for tabular data like this.

**Random Forest** — an ensemble of many decision trees (100, in this project),
each trained on a random slice of the data. Every tree "votes" on the outcome,
and the majority vote wins. In theory, better at capturing non-linear
relationships between features than Logistic Regression.

**Result:** after tuning (limiting tree depth to prevent overfitting), both
models performed almost identically. Logistic Regression was chosen for the
final app for its speed and interpretability — a case where the simpler model
was the right call.

### 4. The balanced vs. unbalanced trade-off

Football results are imbalanced: home wins happen far more often than draws.
Trained "as-is," both models learned to essentially **ignore draws entirely**
(0% draw recall) in exchange for higher raw accuracy — a deceptively good-looking
number hiding a real blind spot.

Applying `class_weight='balanced'` forces the model to weigh mistakes on rarer
outcomes (draws) more heavily:

| | Accuracy | Draws correctly caught |
|---|---|---|
| Unbalanced | 56.5% | 0% |
| **Balanced (used in this app)** | 52% | 26–29% |

Overall accuracy drops slightly, but the model becomes meaningfully more useful —
it can now actually recognize a likely draw instead of defaulting to a win/loss
guess every time. This trade-off was chosen deliberately, prioritizing a model
that reflects real football outcomes over one that just chases a single accuracy
number.

### 5. Live validation

The model was retrained using **only pre-tournament data** (everything before
11 June 2026), then tested purely on the 2026 World Cup's actual results — matches
it had never seen. This avoids data leakage and reflects genuine forecasting
performance, not memorization.

| | Accuracy | Macro F1 |
|---|---|---|
| Naive baseline (always guess "Home Win") | 48.6% | — |
| **This model, on 102 real 2026 World Cup matches** | **55%** | **0.51** |

### 6. Known limitations

- FIFA ranking is a single fixed snapshot, not each team's rank *at the time* of
  every historical match (a simplification, not the full historical-accurate version)
- No player-level data (injuries, individual form, squad changes) — only team-level
  history is available
- For knockout-stage matches, a "Draw" prediction reflects a tie after regulation
  time — the dataset contains no penalty shootout data, so the model cannot predict
  shootout winners

---

## Tech Stack

- **Python**, **pandas**, **scikit-learn** — data processing & modeling
- **Streamlit** — web app framework
- **uv** — dependency management

---

## Project Structure

```
fifa-worldcup-2026-predictor/
├── main.py                   # Streamlit Page 1 — Dashboard
├── utils.py                  # Shared functions: model loading, prediction, theming
├── pages/
│   └── 1_Compare_Teams.py    # Streamlit Page 2 — head-to-head comparison
├── data/
│   ├── results.csv           # Historical match results
│   ├── fifa_ranking.csv      # FIFA World Ranking snapshot
│   └── team_stats.csv        # Precomputed per-team form + rank
├── models/
│   └── match_predictor.pkl   # Trained Logistic Regression model
├── notebook/
│   └── 01_dataset_exploration.ipynb   # Full data prep, feature engineering, and training process
├── pyproject.toml
└── uv.lock
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Run the app
streamlit run main.py
```

---

## Author

Built by [Rohit Ghosh](https://github.com/Rohitghosh14) as a hands-on machine
learning project — from raw data to a deployed, validated prediction app.