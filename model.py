from transformers import pipeline

print("Loading FinBERT model... (first time is slow)")
sentiment = pipeline("text-classification", model="ProsusAI/finbert")

headlines = [
    "NVIDIA beats earnings expectations by 40%",
    "FTC opens antitrust probe into NVIDIA",
    "Chip stocks mixed ahead of Fed decision"
]

results = sentiment(headlines)

for headline, result in zip(headlines, results):
    label = result['label'].upper()
    score = round(result['score'], 3)
    print(f"{label} ({score}) — {headline}")