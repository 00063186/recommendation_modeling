# MobUpps Recommendation Modeling Take-Home

This repository contains a feature-based LambdaMART recommender for the MobUpps Senior Data Scientist take-home assignment.

The implementation is intentionally small: a data/feature layer, one ranking model, reproducible offline evaluation, SQL feature examples, and a concise written analysis.

## Repository structure

```text
.
├── README.md
├── ANALYSIS.md
├── requirements.txt
├── src/
│   ├── data.py
│   ├── features.py
│   └── recommender.py
├── scripts/
│   ├── evaluate.py                  # evaluator supplied with the assignment
│   ├── run_evaluation.py            # fixed global-time-cutoff evaluation
│   └── run_provided_evaluator.py    # compatibility with supplied split/evaluator
├── sql/
│   └── features.sql
├── notebooks/
│   └── exploration.ipynb
└── data/
    └── .gitkeep
```

The CSV data files are deliberately **not committed**.

## Expected data

Mount or copy the assignment files to:

```text
data/users.csv
data/apps.csv
data/interactions.csv
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Model

The model is `src.recommender.LambdaMARTRecommender` and exposes the required interface:

```python
model.fit(train_df)
model.recommend(user_id: str, n: int) -> list[str]
```

The recommender:

1. builds user and app features **only from the training interactions**;
2. aggregates observed `(user_id, app_id)` event histories into an ordinal engagement grade;
3. fits `LGBMRanker(objective="lambdarank")` grouped by user;
4. scores the full app catalog for each known user;
5. returns a global-popularity fallback for unknown users.

Unobserved user-app pairs are not labelled as true negatives. This avoids assuming that non-interaction means dislike, but it also means the model learns engagement strength among observed pairs more directly than interaction occurrence. This tradeoff is discussed in `ANALYSIS.md`.

## Run the fixed-cutoff evaluation

The development evaluation uses one global cutoff, matching the production question "train on the past, predict the future":

```bash
python -m scripts.run_evaluation --data-dir data --cutoff 2026-04-23 --k 10
```

The reference exploratory run produced:

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| LambdaMART | 0.0584 | 0.1025 |
| Popularity | 0.0779 | 0.1081 |

The personalized model did **not** beat popularity. I report that result rather than tune around it because the assignment emphasizes evaluation judgment over leaderboard-style optimization.

## Run with the supplied evaluator

The model is also compatible with the provided per-user leave-last-20%-out evaluation harness:

```bash
python -m scripts.run_provided_evaluator --data-dir data --k 10
```

To run only the supplied popularity smoke test:

```bash
python -m scripts.evaluate --interactions data/interactions.csv --k 10
```

## SQL

`sql/features.sql` contains two example warehouse queries for train-window user and app aggregates. Both explicitly filter on a training cutoff to prevent future leakage.

## Notes on evaluation

Recall@10 measures the fraction of held-out unique apps recovered in the top 10. NDCG@10 additionally rewards putting relevant apps earlier. Both metrics are binary with respect to event strength in the supplied evaluator: a future view and a future purchase both make an app relevant.

See `ANALYSIS.md` for target definition, model choice, evaluation limitations, hyperparameter reasoning, A/B-test design, and data-quality findings.
