import os
import pandas as pd
import numpy as np
import re
import sys
import argparse
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
import matplotlib.pyplot as plt

nltk.download('stopwords')

# --------- CLEAN SCRIPT TEXT ----------
def clean_script(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = re.findall(r'\b\w+\b', text)  # regex-based tokenization (avoids punkt error)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# --------- LOAD DATA ----------
df = pd.read_csv("metadata.csv")
df['log_budget'] = np.log1p(df['budget'])
df['log_box_office'] = np.log1p(df['box_office'])
df['title'] = df['title'].str.strip()

# --------- ADD SCRIPT CONTENT ----------
scripts_dir = "data/scripts"
scripts = []
for title in df['title']:
    script_path = os.path.join(scripts_dir, title.lower().replace(" ", "_") + ".txt")
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read()
            cleaned = clean_script(text)
            scripts.append(cleaned)
    else:
        scripts.append("")

df['clean_script'] = scripts

# --------- SCRIPT FEATURES ----------
df['script_length'] = df['clean_script'].apply(lambda x: len(x.split()))
df['sentiment'] = df['clean_script'].apply(lambda x: TextBlob(x).sentiment.polarity)

# --------- TF-IDF VECTORIZATION ----------
tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(df['clean_script']).toarray()
print(f"✅ TF-IDF shape: {X_text.shape}")

# --------- NUMERIC FEATURES ----------
X_numeric = df[['budget', 'runtime', 'rating', 'script_length', 'sentiment']].fillna(0).values
X = np.hstack((X_text, X_numeric))
y = df['box_office']

# --------- TRAINING MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds) ** 0.5
print(f"📉 RMSE: ${rmse:,.2f}")

# --------- PLOT FEATURE IMPORTANCE ----------
importances = model.feature_importances_[-4:]  # Last 4 are budget, runtime, rating,  sentiment
labels = ['Budget', 'Runtime', 'Rating', 'Sentiment']
plt.barh(labels, importances)
plt.title("Feature Importance")
plt.show()

# --------- USER INPUT MODE ----------
parser = argparse.ArgumentParser()
parser.add_argument("--custom", action="store_true", help="Enter custom movie input")
args = parser.parse_args()

if args.custom:
    print("\n🎬 Custom Mode: Enter your movie details below:")
    title = input("Enter movie title: ")
    budget = float(input("Enter budget (in USD): "))
    if budget > 500_000_000:
        print("⚠️ Budget is unrealistically high. Please enter realistic values.")
        sys.exit()

    runtime = int(input("Enter runtime (in minutes): "))
    if runtime > 210:
        print("⚠️ Runtime is very lengthy. Consider shortening.")
        sys.exit()

    rating = float(input("Enter IMDb rating (0-10): "))
    if rating > 10 or rating < 0:
        print("⚠️ Rating should be between 0 and 10.")
        sys.exit()

    script = input("Enter movie script or summary:\n")
    cleaned_script = clean_script(script)
    sentiment = TextBlob(cleaned_script).sentiment.polarity
    script_length = len(cleaned_script.split())

    X_new_text = tfidf.transform([cleaned_script]).toarray()
    X_new_numeric = np.array([[budget, runtime, rating, script_length, sentiment]])
    X_new = np.hstack((X_new_text, X_new_numeric))

    prediction = model.predict(X_new)[0]
    box_office_prediction = prediction * 0.25  # scaled prediction

    print(f"\n🎯 Predicted Box Office Collection: ${box_office_prediction:,.2f}")
    if box_office_prediction > 2 * budget:
        print("✅ THE MOVIE IS PREDICTED AS A HIT!")
    else:
        print("❌ THE MOVIE IS PREDICTED TO MAYBE A FLOP.")
