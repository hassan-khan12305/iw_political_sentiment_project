# A Time-Varying Parameter Model of Twitter’s Alignment with Political Opinion During the 2024 Election -- Princeton COS Spring 2026 Independent Work

Dynamic latent variable model of political climate using 35.1M tweets from the 2024 US election (W18–W48).

**Core contribution:** A univariate TVP (Time-Varying Parameter) Kalman filter that models how strongly a composite Twitter signal tracks poll-measured political opinion over time - and shows this relationship is not constant but shifts systematically with political events.

---

## Research Question

Does a composite Twitter signal (combining sentiment, stance, leaning, and volume) track poll-measured political opinion with a fixed relationship, or does that relationship vary over time — specifically around major campaign events?

**Short answer from the data:** The relationship varies substantially. The dynamic TVP model explains 80% of Trump poll variance vs 37% for the static fixed-weight model (ΔR²=+0.43).

---

## Pipeline

Run in order:

```bash
# 1. Data ingestion and cleaning
iw/bin/python src/ingest_usc.py
iw/bin/python src/clean_usc.py
iw/bin/python src/aggregate_weekly.py

# 2. Semantic signals
iw/bin/python src/features/sentiment_vader.py
iw/bin/python src/features/merge_sentiment.py --source vader
iw/bin/python src/features/merge_sentiment.py --source tweetnlp
iw/bin/python src/features/political_leaning_hashtags.py
iw/bin/python src/features/merge_leaning.py --source hashtags
iw/bin/python src/features/merge_leaning.py --source stance

# 3. Topic modeling
iw/bin/python src/features/topic_lda.py 
iw/bin/python src/features/merge_topics.py

# 4. Models
iw/bin/python src/models/prepare_observation_matrix.py
iw/bin/python src/models/prepare_polls.py
iw/bin/python src/models/factor_static.py
iw/bin/python src/models/factor_dynamic.py
```

---

## Data Source

USC 2024 US Election Twitter/X dataset — `data/raw/x-24-us-election-usc.zip`

- ~35.1M clean tweets, W18–W48 (2024-04-29 – 2024-11-30)
- English only, original tweets only, deduplicated
- W01–W17 excluded (pre-collection noise, 0.3% of corpus)
- W35 has an anomalous volume dip (~48K vs ~180K surrounding weeks) — collection gap artifact

---

## Key Outputs

| File | Description |
|---|---|
| `data/usc_2024_weekly.parquet` | Main measurement matrix — 31 weeks × 37 columns |
| `data/observation_matrix.parquet` | Standardised 6-signal y_t for model input |
| `data/polls_weekly.parquet` | Weekly polling aggregates — Trump/Harris/Biden net favorability |
| `data/factor_dynamic_output.parquet` | Smoothed loading λ_t, 95% CI, static/dynamic fits per week |
| `results/factor_static_summary.txt` | PCA + SSM baseline diagnostics |
| `results/factor_dynamic_summary.txt` | TVP parameters, LR test, structural break, event-window table |
| `results/event_window_results.csv` | Δλ for 7 events |
| `docs/methodology_and_results.md` | Full technical methodology writeup |

---

## 6 Signals Used in Model

| Signal | Column | Source |
|---|---|---|
| VADER Sentiment | `avg_sentiment_vader` | Lexicon-based, CPU |
| RoBERTa Sentiment | `avg_sentiment_tweetnlp` | TweetNLP fine-tuned, GPU |
| Hashtag Lean | `lean_score_hashtags` | Curated hashtag dictionary |
| Trump Net Stance | `trump_net_stance` | DeBERTa NLI zero-shot, GPU |
| Harris Net Stance | `harris_net_stance` | DeBERTa NLI zero-shot, GPU |
| Log Volume | `log_tweet_count` | log(weekly tweet count) |

Topics (LDA K=5) excluded from primary model — orthogonal to sentiment/stance axis, reduce Factor 1 from 50.4% -> 33.5%.

---

## Key Events (W18–W48 window)

| Week | Date | Event |
|---|---|---|
| W26 | Jun 27 | 1st presidential debate (Biden-Trump) |
| W29 | Jul 15–21 | RNC + Biden withdraws |
| **W30** | **Jul 22** | **Structural break — first full Harris week** |
| W34 | Aug 19 | Democratic National Convention |
| W37 | Sep 10 | 2nd presidential debate (Trump-Harris) |
| W40 | Oct 1 | VP debate (Walz-Vance) |
| W45 | Nov 5 | Election Day |

---

## Key Findings

**Dynamic TVP model:**
- Composite: PLS1 of 6 signals (r=0.61 with Trump polls vs r≈0.05 for PC1)
- PLS1 weights: RoBERTa +0.65, VADER +0.50, Trump Stance +0.45, Hashtag Lean +0.33
- Static R² = 0.37 -> Dynamic R² = **0.80** (ΔR² = +0.43)
- σ_ν = 0.49 (loading drifts significantly — genuinely time-varying)
- LR statistic = 4.83 (significant, boundary χ²(1) p=0.05 threshold = 2.71)

**Structural break W30 (Biden->Harris):**
- Pre-W30 mean λ = +0.63 | Post-W30 mean λ = +0.45 | Δλ = −0.18
- Composite weakened as discourse context shifted from Biden-Trump to Harris-Trump frame

**Event-window results (Δλ = window mean − complement):**
- W26 (1st debate, Biden-Trump): **Δλ = +0.57** — peak informativeness
- W29 (Biden exit + RNC): Δλ = +0.19 — still informative during transition
- W34 (DNC): **Δλ = −0.49** — deep decoupling
- W37 (2nd debate, Trump-Harris): **Δλ = −0.52** — least informative event overall
- W40 (VP debate): Δλ = −0.35 — continued post-Harris decoupling
- W45 (Election Day): **Δλ = +0.45** — strong re-alignment with actual outcome
