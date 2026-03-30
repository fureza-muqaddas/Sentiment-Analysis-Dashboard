# FinSentinel — Media Sentiment Dashboard

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)
![VADER](https://img.shields.io/badge/NLP-VADER%20%7C%20FinBERT-F59E0B?style=flat-square)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render)
![License](https://img.shields.io/badge/License-MIT-6B7280?style=flat-square)

> Real-time media sentiment analysis for any company — scrapes news articles and Reddit posts, scores each one with NLP, and visualizes sentiment trends over time.

**[Live Demo →](https://sentiment-analysis-dashboard-usxx.onrender.com/)**

![FinSentinel Demo](demo.gif)

---

## Features

- **Dual data sources** — NewsAPI (financial news) + Reddit JSON API (no key required)
- **NLP sentiment scoring** — VADER by default; drop-in upgrade to FinBERT (domain-specific financial model)
- **Trend visualization** — Chart.js line chart (sentiment over time) + donut distribution
- **Live article feed** — filterable by positive / negative / neutral, with per-article score bars
- **Zero-friction demo** — Reddit works immediately without any API key
- **Render-ready** — `render.yaml` included for one-click deployment

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/finsentinel.git
cd finsentinel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and add your NewsAPI key (optional — Reddit works without it)

# 4. Run
python app.py
# → http://localhost:5000
```

---

## Environment Variables

| Variable        | Required | Description |
|-----------------|----------|-------------|
| `NEWS_API_KEY`  | Optional | [NewsAPI](https://newsapi.org) free tier — 100 requests/day |

Reddit requires no API key and works out of the box.

---

## Upgrading to FinBERT

For institutional-grade financial sentiment (trained on financial phrasebank):

```bash
# requirements.txt — uncomment these lines:
# transformers==4.41.2
# torch==2.3.1

pip install transformers torch
```

The app auto-detects FinBERT on startup. Falls back to VADER if unavailable.
FinBERT requires ~2GB RAM — use Render's **Standard** plan or run locally.

> **Why both?** VADER is fast and works on Render free tier. FinBERT understands financial language nuance ("shares fell short of expectations" → negative even with neutral words).

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Browser (UI)               │
│   Chart.js trend + donut │ Article feed │
└───────────────┬─────────────────────────┘
                │ POST /api/analyze
┌───────────────▼─────────────────────────┐
│           Flask Backend                 │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  NewsAPI     │  │  Reddit JSON API │ │
│  │  (with key)  │  │  (no key needed) │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         └────────┬──────────┘           │
│          ┌───────▼────────┐             │
│          │ Sentiment Engine│             │
│          │ VADER / FinBERT│             │
│          └───────┬────────┘             │
│                  │ JSON response        │
└──────────────────┘─────────────────────┘
```

---

## Tech Stack

| Layer       | Technology |
|-------------|------------|
| Backend     | Python, Flask |
| NLP         | VADER (`vaderSentiment`), FinBERT (`ProsusAI/finbert`) |
| Data        | NewsAPI, Reddit JSON API |
| Visualization | Chart.js 4, custom CSS |
| Deployment  | Render (`render.yaml`) |

---

## Deployment to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo
3. Render auto-detects `render.yaml` — click Deploy
4. Add `NEWS_API_KEY` in the Render environment variables dashboard (optional)

---

## Related Projects

- [InsightIQ](https://github.com/YOUR_USERNAME/insightiq) — AI-powered data analysis tool (Flask + LLaMA + Chart.js)
- [Student Performance Predictor](https://github.com/YOUR_USERNAME/student-performance-predictor)

---

## License

MIT — see [LICENSE](LICENSE)
