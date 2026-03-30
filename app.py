from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─── Sentiment Analyzer Setup ────────────────────────────────────────────────
ANALYZER = "none"
finbert_pipeline = None
vader = None

try:
    from transformers import pipeline as hf_pipeline
    print("Loading FinBERT... (this may take a moment on first run)")
    finbert_pipeline = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )
    ANALYZER = "finbert"
    print("✓ FinBERT loaded successfully")
except Exception as e:
    print(f"FinBERT unavailable ({e}), falling back to VADER")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        ANALYZER = "vader"
        print("✓ VADER loaded")
    except Exception as e2:
        print(f"VADER also unavailable: {e2}")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"


# ─── Sentiment Helpers ────────────────────────────────────────────────────────
def get_sentiment(text: str) -> tuple[float, str]:
    """Returns (score [-1,1], label ['positive'|'negative'|'neutral'])"""
    if not text or ANALYZER == "none":
        return 0.0, "neutral"

    if ANALYZER == "finbert" and finbert_pipeline:
        try:
            result = finbert_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "negative":
                return round(-score, 4), "negative"
            elif label == "positive":
                return round(score, 4), "positive"
            else:
                return 0.0, "neutral"
        except Exception:
            pass  # fall through to VADER

    if vader:
        scores = vader.polarity_scores(text)
        compound = round(scores["compound"], 4)
        if compound >= 0.05:
            return compound, "positive"
        elif compound <= -0.05:
            return compound, "negative"
        else:
            return compound, "neutral"

    return 0.0, "neutral"


# ─── Data Fetching ────────────────────────────────────────────────────────────
def fetch_news(company: str, days: int = 7) -> list[dict]:
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": f'"{company}"',
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json().get("articles", [])
        articles = []
        for a in raw:
            if not a.get("title"):
                continue
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description") or "",
                "url": a.get("url", "#"),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "publishedAt": a.get("publishedAt", ""),
                "source_type": "news",
            })
        return articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_reddit(company: str, days: int = 7) -> list[dict]:
    time_map = {1: "day", 3: "week", 7: "week", 14: "month", 30: "month"}
    t = time_map.get(days, "month")
    params = {
        "q": company,
        "sort": "new",
        "t": t,
        "limit": 100,
        "type": "link",
    }
    headers = {"User-Agent": "FinSentinel/1.0 (sentiment research tool)"}
    try:
        resp = requests.get(
            REDDIT_SEARCH_URL, params=params, headers=headers, timeout=10
        )
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        posts = []
        cutoff = datetime.now() - timedelta(days=days)
        for child in children:
            d = child.get("data", {})
            created_utc = d.get("created_utc", 0)
            created_dt = datetime.utcfromtimestamp(created_utc)
            if created_dt < cutoff:
                continue
            title = d.get("title", "")
            selftext = (d.get("selftext") or "")[:200]
            if not title:
                continue
            posts.append({
                "title": title,
                "description": selftext,
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "source": f"r/{d.get('subreddit', 'reddit')}",
                "publishedAt": created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_type": "reddit",
            })
        return posts
    except Exception as e:
        print(f"Reddit error: {e}")
        return []


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", analyzer=ANALYZER.upper())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    company = (body.get("company") or "").strip()
    days = max(1, min(int(body.get("days", 7)), 30))
    sources = body.get("sources", ["news", "reddit"])

    if not company:
        return jsonify({"error": "Company name is required"}), 400

    raw_articles: list[dict] = []
    if "news" in sources:
        raw_articles += fetch_news(company, days)
    if "reddit" in sources:
        raw_articles += fetch_reddit(company, days)

    if not raw_articles:
        msg = (
            "No articles found. "
            + ("Add a NewsAPI key to .env to enable news results. " if "news" in sources and not NEWS_API_KEY else "")
            + "Try a different company name or expand the date range."
        )
        return jsonify({"error": msg, "articles": [], "trend": [], "label_counts": {}}), 200

    # ── Sentiment pass ──────────────────────────────────────────────────────
    results = []
    daily: dict[str, dict] = {}

    for a in raw_articles:
        text = f"{a['title']} {a['description']}".strip()
        score, label = get_sentiment(text)
        pub_date = a["publishedAt"][:10] if a["publishedAt"] else ""

        results.append({
            **a,
            "score": score,
            "label": label,
            "date": pub_date,
        })

        if pub_date:
            if pub_date not in daily:
                daily[pub_date] = {
                    "scores": [], "positive": 0, "negative": 0, "neutral": 0
                }
            daily[pub_date]["scores"].append(score)
            daily[pub_date][label] += 1

    # ── Trend aggregation ───────────────────────────────────────────────────
    trend = []
    for date in sorted(daily.keys()):
        s = daily[date]["scores"]
        trend.append({
            "date": date,
            "avg_score": round(sum(s) / len(s), 4),
            "count": len(s),
            "positive": daily[date]["positive"],
            "negative": daily[date]["negative"],
            "neutral": daily[date]["neutral"],
        })

    all_scores = [r["score"] for r in results]
    overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0

    label_counts = {
        "positive": sum(1 for r in results if r["label"] == "positive"),
        "negative": sum(1 for r in results if r["label"] == "negative"),
        "neutral": sum(1 for r in results if r["label"] == "neutral"),
    }

    return jsonify({
        "company": company,
        "analyzer": ANALYZER,
        "days": days,
        "total_articles": len(results),
        "overall_score": overall,
        "overall_label": (
            "positive" if overall > 0.05 else
            "negative" if overall < -0.05 else "neutral"
        ),
        "articles": sorted(results, key=lambda x: x["publishedAt"], reverse=True)[:60],
        "trend": trend,
        "label_counts": label_counts,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
