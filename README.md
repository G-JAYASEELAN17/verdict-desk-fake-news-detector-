# The Verdict Desk — Fake News Detector

A web app that classifies news text as **REAL** or **FAKE** using a
TF-IDF + Logistic Regression model, with per-prediction explainability
showing which words in the input pushed the verdict each way.

## How it works

1. **Dataset**: [ISOT Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   — 44,898 labeled news articles (23,481 FAKE / 21,417 REAL), loaded from
   `Fake.csv` + `True.csv`. Real articles are Reuters wire stories; fake
   articles come from sites flagged by Politifact and Wikipedia as unreliable.
2. **Preprocessing**: lowercase, strip URLs/punctuation, combine title + body.
   Reuters datelines (e.g. "WASHINGTON (Reuters) -") and the word "Reuters"
   itself are stripped out so the model can't just memorize the source's own
   byline instead of learning real fake-vs-real language.
3. **Vectorization**: TF-IDF over unigrams + bigrams (top 15,000 features).
4. **Model**: Logistic Regression — chosen deliberately over a deep learning
   model so every prediction is traceable back to specific word weights
   (no black box).
5. **Explainability**: at prediction time, the app looks up the learned
   coefficient for every word in the submitted text and surfaces the
   strongest words pushing toward FAKE and toward REAL.

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.3% |
| Precision | 98.8% |
| Recall | 98.0% |
| F1 Score | 98.4% |

Evaluated on a held-out 20% test split (8,980 articles) not seen during training.

## Important limitation (be upfront about this)

This model detects **linguistic patterns typical of fake news articles in its
training data** — sensational phrasing, certain recurring topics, certain
sourcing language. It does **not** fact-check claims, verify sources, or know
about events after its training data. A "FAKE" verdict is a signal to verify
elsewhere, not a ruling on truth.

## Project structure

```
fake_news_project/
├── train_model.py          # Trains the model, saves artifacts + metrics
├── app.py                  # Flask app: serves the page + /predict API
├── model.joblib             # Trained Logistic Regression model
├── vectorizer.joblib        # Fitted TF-IDF vectorizer
├── metrics.json              # Saved evaluation metrics + top global words
├── Fake.csv                 # Fake-article half of the training dataset
├── True.csv                  # Real-article half of the training dataset
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Running it locally

```bash
pip install -r requirements.txt

# (Optional) retrain the model — artifacts are already included
python train_model.py

# Start the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Deploying it (for a live resume link)

This is a standard Flask app, so it deploys as-is to any free-tier host that
supports Python web apps, e.g. **Render** or **Railway**:

1. Push this folder to a GitHub repo.
2. On Render/Railway, create a new Web Service from that repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py` (or `gunicorn app:app` for production — add
   `gunicorn` to requirements.txt if you use this)

## Talking points for interviews

- **Why TF-IDF + Logistic Regression over deep learning?** With ~6K articles,
  a simple linear model on sparse TF-IDF features avoids overfitting, trains
  in seconds, and — critically — stays interpretable. You can point to the
  exact words driving any prediction, which a neural net would obscure.
- **Why explainability matters here**: the whole risk of an unexplained fake
  news classifier is that it becomes another unaccountable black box. Showing
  which words drove a verdict lets a user sanity-check the model instead of
  blindly trusting it.
- **Known weakness**: the training data skews toward 2016–2017 US political
  news, and all real articles come from a single source (Reuters), so the
  model's "real" signal partly reflects Reuters' house style (AP-style
  attribution, datelines, wire-service phrasing) rather than truthfulness
  itself. Real news from other outlets with a different voice could be
  misclassified. This is a good example of dataset bias to discuss if asked
  about model limitations.
