import requests
from transformers import pipeline

def get_news(company, api_key):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=relevancy&pageSize=10&apiKey={api_key}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] for a in articles if a["title"] != "[Removed]"]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
COMPANY = "NVIDIA"

print(f"Fetching news for {COMPANY}...")
headlines = get_news(COMPANY, NEWS_API_KEY)
print(f"Found {len(headlines)} articles\n")

print("Loading FinBERT model...")
sentiment = pipeline("text-classification", model="ProsusAI/finbert")

results = sentiment(headlines)

print(f"\n── SENTIMENT RESULTS FOR {COMPANY} ──\n")
for headline, result in zip(headlines, results):
    label = result['label'].upper()
    score = round(result['score'], 3)
    emoji = "🟢" if label == "POSITIVE" else "🔴" if label == "NEGATIVE" else "⚪"
    print(f"{emoji} {label} ({score})")
    print(f"   {headline}\n")