# =============================================================
# Comparative Analysis: Threads and Twitter Reviews
# =============================================================
# import libraries
import pandas as pd
import numpy as np
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from scipy.stats import pearsonr, spearmanr
# For BERTopic
import os
import random
import numpy as np
import torch
# For Langdetect
from langdetect import detect, LangDetectException, DetectorFactory
import math
import textwrap

# ===== Code by Haeyeon Part 1 start ===== #
# Set the seed to reproduce the same results
DetectorFactory.seed = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HDBSCAN_RANDOM_STATE"] = "42"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from hdbscan import HDBSCAN
from umap import UMAP

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    random_state=SEED
)

hdbscan_model = HDBSCAN(
    min_cluster_size=20,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Make sure you have NLTK punkt tokenizer
nltk.download("punkt")

# =============================================================
# 1. Load the datasets and check data
# The original data did not include the sentiment_true column. 
# We added it manually for about 3,000 rows to compare human labels with VADER labels.
# =============================================================

threads_df = pd.read_csv("Data/threads_reviews_labelled.csv")     
twitter_df = pd.read_csv("Data/twitter_reviews_labelled.csv") 

### Looking at Threads dataset
# First 5 Threads
print(threads_df.head())

# Last 5 threads
print(threads_df.tail())

# count number by source
threads_df["source"].value_counts()

# ------------------------------------------------------------
# threads dataset has few 'app_store' data points.
# There are 2640 rows for data from app_store.
# ------------------------------------------------------------

### Looking at Twitter dataset
# First 5 Twitter
print(twitter_df.head())

# Last 5 twitter
print(twitter_df.tail())

# ------------------------------------------------------------
# We detect different language other than english, we will handle this in data preprocessing.
# ------------------------------------------------------------

### Threads Info
# Info print for threads
print(threads_df.info(), "\n")

### Twitter Info
# Info print for twitter
print(twitter_df.info())

# Columns for confirmation
print("Threads columns:", threads_df.columns.tolist())
print("Twitter columns:", twitter_df.columns.tolist())

# ------------------------------------------------------------
# Threads data and its review_description can be compared to Twitter data’s review_text.
# We see Twitter has unwanted columns not required for our project
# ------------------------------------------------------------

### Missing values - Threads
# Missing Values of Threads
print("\nMissing values (Threads):\n", threads_df.isnull().sum())

### Missing Values - Twitter
# Missing Values of Twitter
print("\nMissing values (Twitter):\n", twitter_df.isnull().sum())

# ------------------------------------------------------------
# Twitter has missing values in 2 columns 
# ------------------------------------------------------------


# =============================================================
# 2. Data Cleaning Pipeline
# We used slightly different pipeline for each data.
#
# For Threads:
# Removed duplicated content (raw + long-review duplicates)
# Standardized brand names
# Removed noise, emojis, special chars, stopwords
# Handled negations properly
# Removed empty texts after cleaning
# Detected language and removed unwanted languages
#
# For Twitter:
# Applied different custom stopwords, language filtering rules (Twitter tends to have much more global usage → more foreign languages)
# Everything else (duplicate logic, text cleaning, brand normalization workflow) is the same.
# =============================================================

### Data Cleaning Pipeline For Threads
# ------------------------------------------------------------
# Part 1. DUPLICATE DETECTION (RAW TEXT)
# ------------------------------------------------------------

print("\n===== RAW DUPLICATE CHECK =====")
df = threads_df.copy()
dups_raw = df[df.duplicated(subset=["review_description"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_description"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))


# ------------------------------------------------------------
# Part 2. LONG-REVIEW DUPLICATE REMOVAL
# ------------------------------------------------------------

def is_long_review(text, min_words=15):
    return len(str(text).split()) >= min_words

df["is_long"] = df["review_description"].apply(is_long_review)

df_long = df[df["is_long"] == True].copy()
df_short = df[df["is_long"] == False].copy()

print("\n===== LONG REVIEW STATS =====")
print("Long reviews:", df_long.shape[0])
print("Short reviews:", df_short.shape[0])

dup_long_mask = df_long.duplicated(subset=["review_description"], keep=False)
df_long_dups = df_long[dup_long_mask]

print("\n===== LONG DUPLICATES FOUND =====")
print("Total long duplicate rows:", df_long_dups.shape[0])

dup_groups_long = (
    df_long_dups.reset_index()
                .groupby("review_description")["index"]
                .apply(list)
                .reset_index(name="row_numbers")
)

print("Distinct long duplicated texts:", len(dup_groups_long))
print("\n--- SAMPLE LONG DUPLICATE GROUPS ---")
print(dup_groups_long.head(10))

before_long = df_long.shape[0]
df_long_clean = df_long.drop_duplicates(subset=["review_description"], keep="first")
after_long = df_long_clean.shape[0]

print("\n===== LONG DUPLICATE REMOVAL =====")
print("Before:", before_long)
print("After :", after_long)
print("Removed:", before_long - after_long)

df = pd.concat([df_long_clean, df_short], ignore_index=True)
df.drop(columns=["is_long"], inplace=True)

print("\n===== AFTER LONG-DEDUPE: df.shape =====")
print(df.shape)

# ------------------------------------------------------------
# Part 3. BRAND NORMALIZATION MAP
# ------------------------------------------------------------

brand_map = {
    # Twitter / X
    r"\btwitter\b": "twitter",
    r"\btwt\b": "twitter",
    r"\btwit\b": "twitter",
    r"\btwwitter\b": "twitter",
    r"\btwiter\b": "twitter",
    r"\bx\b": "twitter",

    # Threads
    r"\bthreads\b": "threads",
    r"\bthread\b": "threads",
    r"\btread\b": "threads",
    r"\btreads\b": "threads",
    r"\bthreds\b": "threads",

    # Instagram
    r"\binstagram\b": "instagram",
    r"\binsta\b": "instagram",
    r"\binstgram\b": "instagram",
    r"\binstragram\b": "instagram",

    # Facebook + Meta
    r"\bfacebook\b": "facebook",
    r"\bfacebok\b": "facebook",
    r"\bfb\b": "facebook",
    r"\bmeta\b": "facebook",
}

def normalize_brands(text):
    text = str(text).lower()
    for pattern, repl in brand_map.items():
        text = re.sub(pattern, repl, text)
    return text


# ------------------------------------------------------------
# Part 4. CLEANING PIPELINE (COMBINED)
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

# Custom stopwords - INCLUDING normalized brand names
custom_stopwords = set([
    "app", "apps", "application", "applications",
    "threads", "experience", "im", "account"
])

def clean_text(text):
    # Step 1: Normalize brands FIRST (before any other processing)
    text = normalize_brands(text)
    
    # Step 2: Basic cleaning
    text = str(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Remove special chars (added space)
    text = re.sub(r"\s+", " ", text).strip()    # Normalize whitespace
    
    # Step 3: Tokenization and stopword removal
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)


# Apply cleaning
df["review_cleaned"] = df["review_description"].apply(clean_text)


print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nOriginal:", df.loc[i, "review_description"])
    print("Cleaned :", df.loc[i, "review_cleaned"])

# ------------------------------------------------------------
# Part 5. REMOVE EMPTY CLEANED ROWS
# ------------------------------------------------------------

before = df.shape[0]
df = df[df["review_cleaned"].str.strip() != ""]
after = df.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)


# ------------------------------------------------------------
# Part 5.1 MINIMAL LANGUAGE DETECTION & FILTERING
# ------------------------------------------------------------

REMOVE_LANGS = {"vi","tr","sw","sq","so","pt","lv","lt","id","hr","et","es"}

def detect_language(text):
    try:
        text_str = str(text).strip()
        if len(text_str) < 10:
            return 'unknown'
        return detect(text_str)
    except LangDetectException:
        return 'unknown'
    except:
        return 'unknown'

# Detect languages
df["detected_lang"] = df["review_cleaned"].apply(detect_language)

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["detected_lang"].value_counts())


# REMOVE ONLY SPECIFIED LANGUAGES
# keep English + unknown + any other safe languages

before = df.shape[0]

df = df[~df["detected_lang"].isin(REMOVE_LANGS)].copy()
df = df.reset_index(drop=True)

print(f"\nRemoved {before - df.shape[0]} rows from unwanted languages.")
print("\n===== FINAL DATASET SHAPE =====")
print(df.shape)

# ------------------------------------------------------------
# Original rows ≈ 33,000
# Final rows = 29,646
# Total removed ≈ 3,300 rows (≈10%)
# ------------------------------------------------------------

### Apply cleaning to Threads Data
# Apply cleaning
threads_clean  =  df.copy()
threads_clean["review_cleaned"] = threads_clean["review_description"].apply(clean_text)


print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", threads_clean.loc[i, "review_description"])
    print("Cleaned :", threads_clean.loc[i, "review_cleaned"])


### Data Cleaning Pipeline For Twitter
# ------------------------------------------------------------
# Part 1. DUPLICATE DETECTION (RAW TEXT)
# ------------------------------------------------------------

print("\n===== RAW DUPLICATE CHECK =====")
df = twitter_df.copy()
dups_raw = df[df.duplicated(subset=["review_text"], keep=False)]
dups_raw_idx = dups_raw.reset_index()[["index", "review_text"]]

print("Total RAW duplicate rows:", len(dups_raw_idx))
print("\n--- SAMPLE RAW DUPLICATES ---")
print(dups_raw_idx.head(10))


# ------------------------------------------------------------
# Part 2. LONG-REVIEW DUPLICATE REMOVAL
# ------------------------------------------------------------

def is_long_review(text, min_words=15):
    return len(str(text).split()) >= min_words

df["is_long"] = df["review_text"].apply(is_long_review)

df_long = df[df["is_long"] == True].copy()
df_short = df[df["is_long"] == False].copy()

print("\n===== LONG REVIEW STATS =====")
print("Long reviews:", df_long.shape[0])
print("Short reviews:", df_short.shape[0])

dup_long_mask = df_long.duplicated(subset=["review_text"], keep=False)
df_long_dups = df_long[dup_long_mask]

print("\n===== LONG DUPLICATES FOUND =====")
print("Total long duplicate rows:", df_long_dups.shape[0])

dup_groups_long = (
    df_long_dups.reset_index()
                .groupby("review_text")["index"]
                .apply(list)
                .reset_index(name="row_numbers")
)

print("Distinct long duplicated texts:", len(dup_groups_long))
print("\n--- SAMPLE LONG DUPLICATE GROUPS ---")
print(dup_groups_long.head(10))

before_long = df_long.shape[0]
df_long_clean = df_long.drop_duplicates(subset=["review_text"], keep="first")
after_long = df_long_clean.shape[0]

print("\n===== LONG DUPLICATE REMOVAL =====")
print("Before:", before_long)
print("After :", after_long)
print("Removed:", before_long - after_long)

df = pd.concat([df_long_clean, df_short], ignore_index=True)
df.drop(columns=["is_long"], inplace=True)

print("\n===== AFTER LONG-DEDUPE: df.shape =====")
print(df.shape)


# ------------------------------------------------------------
# Part 3. BRAND NORMALIZATION MAP
# ------------------------------------------------------------

brand_map = {
    # Twitter / X
    r"\btwitter\b": "twitter",
    r"\btwt\b": "twitter",
    r"\btwit\b": "twitter",
    r"\btwwitter\b": "twitter",
    r"\btwiter\b": "twitter",
    r"\bx\b": "twitter",

    # Threads
    r"\bthreads\b": "threads",
    r"\bthread\b": "threads",
    r"\btread\b": "threads",
    r"\btreads\b": "threads",
    r"\bthreds\b": "threads",

    # Instagram
    r"\binstagram\b": "instagram",
    r"\binsta\b": "instagram",
    r"\binstgram\b": "instagram",
    r"\binstragram\b": "instagram",

    # Facebook + Meta
    r"\bfacebook\b": "facebook",
    r"\bfacebok\b": "facebook",
    r"\bfb\b": "facebook",
    r"\bmeta\b": "facebook",
}

def normalize_brands(text):
    text = str(text).lower()
    for pattern, repl in brand_map.items():
        text = re.sub(pattern, repl, text)
    return text


# ------------------------------------------------------------
# Part 4. CLEANING PIPELINE (COMBINED)
# ------------------------------------------------------------

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

# Custom stopwords - INCLUDING normalized brand names
custom_stopwords = set([
    "app", "apps", "application", "applications",
    "twitter", "im", "account"
])

def clean_text(text):
    # Step 1: Normalize brands FIRST (before any other processing)
    text = normalize_brands(text)
    
    # Step 2: Basic cleaning
    text = str(text)
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)  # Remove special chars (added space)
    text = re.sub(r"\s+", " ", text).strip()    # Normalize whitespace
    
    # Step 3: Tokenization and stopword removal
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# Apply cleaning
df["review_cleaned"] = df["review_text"].apply(clean_text)

print("\n===== CLEANING CHECK =====")
for i in range(5):
    print("\nOriginal:", df.loc[i, "review_text"])
    print("Cleaned :", df.loc[i, "review_cleaned"])


# ------------------------------------------------------------
# Part 5. REMOVE EMPTY CLEANED ROWS
# ------------------------------------------------------------

before = df.shape[0]
df = df[df["review_cleaned"].str.strip() != ""]
after = df.shape[0]

print("\n===== EMPTY CLEANED TEXT REMOVAL =====")
print("Before:", before)
print("After:", after)
print("Removed:", before - after)

# ------------------------------------------------------------
# Part 5.1 MINIMAL LANGUAGE DETECTION & FILTERING
# ------------------------------------------------------------

REMOVE_LANGS = {"vi","tr","tl","sw","sq","so","sl","pl","pt","lv","lt","id","hr","es","ca","cy","de","fi","hu"}

def detect_language(text):
    try:
        text_str = str(text).strip()
        if len(text_str) < 10:
            return 'unknown'
        return detect(text_str)
    except LangDetectException:
        return 'unknown'
    except:
        return 'unknown'

# Detect languages
df["detected_lang"] = df["review_cleaned"].apply(detect_language)

print("\n===== LANGUAGE DISTRIBUTION =====")
print(df["detected_lang"].value_counts())

# -----------------------------------------------------
# REMOVE ONLY SPECIFIED LANGUAGES
# keep English + unknown + any other safe languages
# -----------------------------------------------------
before = df.shape[0]

df = df[~df["detected_lang"].isin(REMOVE_LANGS)].copy()
df = df.reset_index(drop=True)

print(f"\nRemoved {before - df.shape[0]} rows from unwanted languages.")
print("\n===== FINAL DATASET SHAPE =====")
print(df.shape)

# -----------------------------------------------------
# Original rows ≈ 34,788
# Final rows = 29,611
# Total removed ≈ 5,177 rows (≈14.9%)
# -----------------------------------------------------

### Apply cleaning for Twitter
# Apply cleaning
twitter_clean = df.copy()

print("\n--- CLEANING CHECK: original vs cleaned ---")
for i in range(3):
    print("\nOriginal:", twitter_clean.loc[i, "review_text"])
    print("Cleaned :", twitter_clean.loc[i, "review_cleaned"])


# -----------------------------------------------------
# [Additional Insights] Major Insights from Top-20 Words: Threads vs Twitter
#
# - Threads shows early-stage excitement + requests:
#   "Threads is promising and feels good to use. But to compete with Twitter and Instagram, it needs more features”
# - Twitter shows dissatisfaction tied to leadership:
#   “Elon/Musk made updates that made things worse
# -----------------------------------------------------
### Threads frequent Words

# Combine all reviews (or you can do per cluster)
all_words = " ".join(threads_clean["review_cleaned"]).split()  # or nltk.word_tokenize(text)
freq_dist = FreqDist(all_words)

# Top 20 words overall
print("Top 20 words overall:")
for word, freq in freq_dist.most_common(20):
    print(f"{word}: {freq}")

# -----------------------------------------------------
# - twitter (5806), instagram (3644), facebook (1135) → heavy cross-platform comparison
# - good, nice, great, better → positive onboarding sentiment
# - follow, people, see, post → emphasis on social interaction and discovering content
# - feature, please, need, want → strong demand for missing functionality
# - use, new → early adoption and first-time user experience
#
# Threads comments show:
# - Users are actively comparing Threads with Twitter, Instagram, and Facebook.
# - The overall tone is positive, with many users praising the app’s feel and design.
# - Feedback is constructive and improvement-focused, with repeated requests for features.
# - Early users want Threads to expand functionality to become competitive with Twitter/Instagram.
# -----------------------------------------------------
### Twitter Frequent words

# Combine all reviews (or you can do per cluster)
all_words = " ".join(twitter_clean["review_cleaned"]).split()  # or nltk.word_tokenize(text)
freq_dist = FreqDist(all_words)

# Top 20 words overall
print("Top 20 words overall:")
for word, freq in freq_dist.most_common(20):
    print(f"{word}: {freq}")

# -----------------------------------------------------
# - elon (3885), musk (1907) → extremely Musk-centric discussion
# - tweet, name, change, update → focused on platform changes under Musk
# - not, no, worse, even → negative sentiment
# - get, back, since → reactions to changes and reversions
# - social → identity as a social platform under debate
#
#Twitter comments show:
# - The platform identity is tightly tied to Elon Musk.
# - Users comment heavily on updates, renaming (Twitter → X), UI changes, etc.
# - Much more complaining, negativity
# -----------------------------------------------------
# ===== Code by Haeyeon Part 1 end ===== #

# ===== Code by Snehitha Tadapaneni Part 1 start ===== #
# =============================================================
# 3-1. SENTIMENT ANALYSIS : USING VADER + KNN FOR THREADS
# =============================================================
# By applying KMeans clustering to the compound sentiment scores, we could segment reviews into three clusters. The centroids of these clusters inform the thresholds for classifying sentiment, allowing for more objective labeling than manual cutoff values.

# ----------- LOAD DATA --------------
df_threads = threads_clean.copy()     
# ------------------------------------

# ---------- VADER SCORES -----------
analyzer = SentimentIntensityAnalyzer()
df_threads["compound"] = df_threads["review_cleaned"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
compound_array = df_threads["compound"].values.reshape(-1,1)
# ------------------------------------

# ----------- KMEANS 3 CLUSTERS -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(compound_array)
df_threads["cluster"] = clusters

# map centroids -> pos/neu/neg
centroids = kmeans.cluster_centers_.flatten()
order = np.argsort(centroids)
label_map = {order[0]: "negative", order[1]: "neutral", order[2]: "positive"}

df_threads["sentiment"] = df_threads["cluster"].map(label_map)
# ------------------------------------

print(df_threads[["review_description","review_cleaned","compound","sentiment"]].head())

# ----------- SENTIMENT DISTRIBUTION PLOT ---------
import matplotlib.pyplot as plt

df_threads["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution (KMeans + VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

# ------ BOXPLOT OF COMPOUND SCORES BY SENTIMENT -------
df_threads.boxplot(column="compound", by="sentiment", grid=False)

plt.title("Compound Score by Sentiment Cluster")
plt.suptitle("")
plt.xlabel("Sentiment")
plt.ylabel("Compound Score")
plt.tight_layout()
plt.show()


# ----------- SCATTER PLOT OF CLUSTERS -----------
# Set style
sns.set(style="whitegrid")

# Plot the clusters along the compound score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=range(len(df_threads)), 
    y="compound", 
    hue="sentiment", 
    palette={"negative":"red", "neutral":"gray", "positive":"green"},
    data=df_threads,
    s=50
)
plt.title("KMeans Clusters of Reviews (VADER Compound Scores)")
plt.xlabel("Review Index")
plt.ylabel("Compound Score")
plt.legend(title="Sentiment")
plt.show()

# The KNN clustering makes a custom threshold for us without us needing to have a hard coded threshold for different sentiments.The reviews seem to be perfectly clustered with respect to their respective scores.

# --------------------------------------------------------------------
# ----------- WORDCLOUDS + TOP WORDS PER SENTIMENT CLUSTER FOR THREADS -----------
# --------------------------------------------------------------------
# Define a function to get top words
def get_top_words(text_series, n=10):
    words = " ".join(text_series).split()
    counter = Counter(words)
    return counter.most_common(n)

# Plot word clouds for each sentiment cluster
sentiments = df_threads["sentiment"].unique()
plt.figure(figsize=(18,6))

for i, sentiment in enumerate(sentiments, 1):
    text = df_threads[df_threads["sentiment"] == sentiment]["review_cleaned"]
    
    # Generate word cloud
    wordcloud = WordCloud(width=400, height=300, background_color="white").generate(" ".join(text))
    
    # Plot
    plt.subplot(1, len(sentiments), i)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment.capitalize()} Reviews")
    
    # Print top 10 words in console
    print(f"Top words in {sentiment} cluster:", get_top_words(text))

plt.tight_layout()
plt.show()

threads_pos = df_threads[df_threads["sentiment"] == "positive"]
threads_neu = df_threads[df_threads["sentiment"] == "neutral"]
threads_neg = df_threads[df_threads["sentiment"] == "negative"]

# - The negative word cloud shows frustrations about posting, missing features, and comparisons where Threads feels worse than Twitter/Instagram.
# - The positive word cloud highlights enthusiasm, with words like good, nice, great, and love showing strong early satisfaction.
# - The neutral word cloud reflects general observations about usage: copying, posting, following, and feature needs without strong sentiment.

# ------------------------------------------------------------
# ----------- STANDARDIZE MANUAL LABELS -----------
# ------------------------------------------------------------
# Clean label mapping
label_mapping = {
    'negative': 'negative',
    'positive': 'positive',
    'neutral': 'neutral',
    'Negative': 'negative',
    'Positive': 'positive',
    'Neutral': 'neutral',
}

# Create cleaned labels column
df_threads["sentiment_true_clean"] = df_threads["sentiment_true"].map(label_mapping)

# Check unmapped labels
unmapped = df_threads[df_threads["sentiment_true_clean"].isna()]["sentiment_true"].unique()
print("Unmapped labels:", unmapped)

print(df_threads["sentiment_true_clean"].value_counts())

# Use only manually labeled rows
df_gold = df_threads[df_threads["sentiment_true_clean"].notna()].copy()

# Train/test split on labeled subset
train_df_threads, test_df_threads = train_test_split(
    df_gold,
    test_size=0.2,
    random_state=42,
    stratify=df_gold["sentiment_true_clean"]
)

print("Train distribution:\n", train_df_threads["sentiment_true_clean"].value_counts())
print("Test distribution:\n", test_df_threads["sentiment_true_clean"].value_counts())


# ------------------------------------------------------------
# ----------- SENTIMENT RELIABILITY
# VADER RELIABILITY CHECK WITH CLEAN LABELS (THREADS)
# ------------------------------------------------------------
print("THREADS — VADER SENTIMENT RELIABILITY CHECK")

# Filter only manually labeled rows
df_labeled = df_threads[df_threads['sentiment_true_clean'].notna()].copy()

print(f"\nTotal manually labeled samples: {len(df_labeled)}")
print("\nDistribution of manual labels:")
print(df_labeled['sentiment_true_clean'].value_counts())

print("\nDistribution of VADER predictions:")
print(df_labeled['sentiment'].value_counts())

# Calculate metrics
y_true = df_labeled['sentiment_true_clean']
y_pred = df_labeled['sentiment']

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\nOVERALL METRICS:")
print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro):    {recall_macro:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")

print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])

# Row % matrix
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# ------------------------------------------------------------
# PLOT CONFUSION MATRICES
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# COUNTS PLOT
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('THREADS — VADER Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)

# PERCENTAGES PLOT
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('THREADS — VADER Confusion Matrix (Row %)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Overall Summary:
# We validated VADER sentiment predictions using 3,047 manually labeled Threads reviews.
# VADER’s overall accuracy was 54.45%, with macro F1 of 0.53, indicating modest performance.
# The confusion matrix shows that VADER tends to overpredict positive sentiment and struggles with neutral or negative statements, largely due to slang and informal writing in user-generated social media data.

# Detailed Analysis:
# To evaluate the reliability of VADER sentiment classification for our Threads dataset, we manually labelled a validation set of 3,047 comments. VADER achieved an accuracy of 54.45% and a macro F1-score of 0.53, indicating moderate but limited performance as a lexicon-based, unsupervised baseline. This is expected because Threads comments contain slang and conversational tone that are difficult for rule-based models to interpret.
# VADER performed well for positive sentiment, achieving a high recall of 92.73% and an F1-score of 0.66, suggesting that it reliably recognises positive language on Threads. However, the model showed substantial challenges with neutral and negative sentiment. Negative comments were identified with reasonable precision (54.02%) but low recall (39.51%), meaning many negative posts were incorrectly classified as neutral or positive. Neutral sentiment was particularly difficult (F1: 0.48), which aligns with known limitations of lexicon-based approaches when dealing with ambiguous, informal, or context-heavy text. Human labelling may also include subjectivity, contributing to inconsistencies in this category.
# Overall, VADER provides a useful initial baseline for sentiment labeling on Threads data, but its limitations—especially around neutral and negative sentiment.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Model 1: Sentiment Classification using TF IDF + Logistic Regression (Threads)
# ------------------------------------------------------------
# Use the full dataset with VADER+KMeans sentiment
df_vader = df_threads.copy()

# Only keep rows where VADER assigned a label (it should be all)
df_vader = df_vader[df_vader["sentiment"].notna()]

print(df_vader["sentiment"].value_counts())

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=3,
    stop_words="english"
)

# Train-test split
X_train = tfidf.fit_transform(train_df_threads["review_cleaned"])
X_test = tfidf.transform(test_df_threads["review_cleaned"])

y_train = train_df_threads["sentiment"]
y_test = test_df_threads["sentiment"]

# Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ------------------------------------------------------------
# Summary of Results:
# We trained a TF-IDF + Logistic Regression classifier on the Threads dataset using VADER-generated sentiment labels.
# The model achieved an accuracy of 74.43% with a macro F1-score of 0.70.

# Performance varies across sentiment classes:

# - Positive sentiment was detected most reliably (F1 = 0.86) with strong precision (0.93).
# - Neutral class achieved moderate performance (F1 = 0.68) and was generally well-recalled (0.74).
# - Negative sentiment remained the most challenging (F1 = 0.55).

# Overall, these results show that a traditional machine-learning model like TF-IDF + Logistic Regression can still provide strong performance for social media sentiment classification. Despite being trained on weak (auto-generated) labels, the model captures meaningful textual patterns and offers more stable, consistent predictions than lexicon-based approaches such as VADER.
# ------------------------------------------------------------
# ===== Code by Snehitha Tadapaneni Part 1 end ===== #

# ===== Code by Haeyeon Part 2 start ===== #
# ------------------------------------------------------------
# Model 2: Sentiment Classification using DistilBERT (Threads)
# ------------------------------------------------------------
# Encode Labels
label_encoder = LabelEncoder()
y_train_bert = label_encoder.fit_transform(train_df_threads["sentiment"])
y_test_bert = label_encoder.transform(test_df_threads["sentiment"])

# Tokenize Text for BERT
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_enc = tokenizer(
    train_df_threads["review_description"].tolist(), 
    # used raw data, not cleaned one
    truncation=True,
    padding=True,
    max_length=128
)

test_enc = tokenizer(
    test_df_threads["review_description"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# Build Dataset Objects
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k,v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BERTDataset(train_enc, y_train_bert)
test_dataset = BERTDataset(test_enc, y_test_bert)

# Load DistilBERT Model+ Define TrainingArguments + Trainer
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# ------------------------------------------------------------
# Hyperparameter before tuning
# ------------------------------------------------------------
# training_args = TrainingArguments(
#      output_dir="./bert_results",
#      eval_strategy="epoch", #  evaluation_strategy="epoch",
#      save_strategy="epoch",
#      num_train_epochs=3,
#      learning_rate=2e-5,
#      per_device_train_batch_size=16,
#      per_device_eval_batch_size=16,
#      weight_decay=0.01,
#      seed=SEED
#  )

# ------------------------------------------------------------
# [Summary-Before tuning]
# The model achieved 84.43% accuracy and a macro F1-score of 0.81, outperforming the TF-IDF + Logistic Regression baseline (macro F1 ≈0.70).
# Performance gains were especially strong for negative and neutral categories—classes that rule-based approaches typically struggle with:
# - Negative: F1 = 0.73
# - Neutral: F1 = 0.77
# - Positive: F1 = 0.93
# ------------------------------------------------------------
# ===== Code by Haeyeon Part 2 end ===== #

# ===== Code by Rachana Part 1 start ===== #
# ------------------------------------------------------------
# HYPERPARAMETER TUNING CODE FOR BERT
# ------------------------------------------------------------
# ----------- HYPERPARAMETER TUNING START --------------
# import optuna
# import numpy as np
# from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, classification_report

# # Create Validation Set from Training Data
# from sklearn.model_selection import train_test_split

# # Split the training data into train and validation sets
# train_texts = train_df_threads["review_description"].tolist()
# train_labels = y_train_bert

# # Split with stratification to maintain class distribution
# train_texts, val_texts, y_train_split, y_val = train_test_split(
#     train_texts, 
#     train_labels, 
#     test_size=0.2, 
#     random_state=SEED,
#     stratify=train_labels
# )

# # Tokenize the new train split and validation set
# train_enc = tokenizer(
#     train_texts, 
#     truncation=True,
#     padding=True,
#     max_length=128
# )

# val_enc = tokenizer(
#     val_texts,
#     truncation=True,
#     padding=True,
#     max_length=128
# )

# # Create dataset objects for train, validation, and test
# train_dataset = BERTDataset(train_enc, y_train_split)
# val_dataset = BERTDataset(val_enc, y_val)
# test_dataset = BERTDataset(test_enc, y_test_bert)

# # three datasets:
# # - train_dataset: for training
# # - val_dataset: for hyperparameter tuning (use this in Optuna)
# # - test_dataset: for final evaluation only

# # ------------------------------------------------------------
# # 2. Objective function for Optuna tuning(This will be commented due to avoid running code for 2 hours)
# # ------------------------------------------------------------
# def objective(trial):

#     # ---- Search space ----
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-5)
#     num_epochs = trial.suggest_int('num_train_epochs', 2, 5)
#     batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
#     weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.3)
#     warmup_steps = trial.suggest_int('warmup_steps', 0, 500)

#     # ---- New model for every trial ----
#     model = DistilBertForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased",
#         num_labels=3
#     )

#     # ---- Training arguments for this trial ----
#     training_args = TrainingArguments(
#         output_dir=f"./bert_trial_{trial.number}",
#         evaluation_strategy="epoch",
#         save_strategy="no",
#         num_train_epochs=num_epochs,
#         learning_rate=learning_rate,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         weight_decay=weight_decay,
#         warmup_steps=warmup_steps,
#         seed=SEED,
#         logging_steps=50,
#         load_best_model_at_end=False,
#     )

#     # ---- Trainer ----
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,  # Use the new train_dataset
#         eval_dataset=val_dataset      # Use val_dataset for tuning, NOT test_dataset
#     )

#     # ---- Train ----
#     trainer.train()

#     # ---- Evaluate ----
#     eval_result = trainer.evaluate()

#     # Optuna tries to minimize this value
#     return eval_result["eval_loss"]


# #
# # 3. Run the Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=20)

# print("Best hyperparameters:", study.best_params)
# ----------- HYPERPARAMETER TUNING END --------------



# We used HuggingFace optuna to check loss for various paramaters and found the best parameters which we applied below
# ------------------------------------------------------------
# HYPERPARAMETER TUNING Applied
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",       # keep evaluating at end of each epoch
    save_strategy="epoch",       # save checkpoint at each epoch
    num_train_epochs=4,          # from Optuna
    learning_rate=3.6744e-5,     # from Optuna
    per_device_train_batch_size=32,  # from Optuna
    per_device_eval_batch_size=32,   
    weight_decay=0.2990,         # from Optuna
    warmup_steps=371,            # from Optuna
    seed=SEED,
    load_best_model_at_end=True,
    logging_steps = 50,
)
# ===== Code by Rachana Part 1 end ===== #

# ===== Code by Haeyeon Part 3 start ===== #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train and Evaluation
trainer.train()

preds = trainer.predict(test_dataset)
y_pred_bert = np.argmax(preds.predictions, axis=1)

print("Threads BERT Accuracy:", accuracy_score(y_test_bert, y_pred_bert))
print(classification_report(
    y_test_bert,
    y_pred_bert,
    target_names=label_encoder.classes_,
    digits=4
))

# ------------------------------------------------------------
#[Summary - after hyperparameter tuning]
# Fine-tuning DistilBERT on the Threads dataset significantly improved sentiment classification performance.
# The model achieved 86.23% accuracy and a macro F1-score of 0.81, outperforming  and the TF-IDF + Logistic Regression baseline (macro F1 ≈0.70).
# Performance gains were especially strong for negative and neutral categories—classes that rule-based approaches typically struggle with:
# - Negative: F1 = 0.74
# - Neutral: F1 = 0.79
# - Positive: F1 = 0.93
#
# These results demonstrate that pretrained transformer models can effectively capture nuanced linguistic cues even when trained on noisy, weakly labeled data.
#Overall, DistilBERT provides a much more robust and reliable method for real-world social media sentiment analysis compared to lexicon-based or classical machine-learning baselines.
# ------------------------------------------------------------
# ===== Code by Haeyeon Part 3 end ===== #

# ===== Code by Rachana Part 2 start ===== #
# =============================================================
# 3-2. SENTIMENT ANALYSIS : USING VADER + KNN FOR TWITTER
# =============================================================

# By applying KMeans clustering to the compound sentiment scores, we could segment reviews into three clusters. 
# The centroids of these clusters inform the thresholds for classifying sentiment, allowing for more objective labeling
# than manual cutoff values.

# ----------- LOAD DATA --------------
df_twitter = twitter_clean.copy()     
# ------------------------------------


# ---------- VADER SCORES -----------
analyzer = SentimentIntensityAnalyzer()
df_twitter["compound"] = df_twitter["review_cleaned"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
compound_array = df_twitter["compound"].values.reshape(-1,1)
# ------------------------------------

# ----------- KMEANS 3 CLUSTERS -----
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(compound_array)
df_twitter["cluster"] = clusters

# map centroids -> pos/neu/neg
centroids = kmeans.cluster_centers_.flatten()
order = np.argsort(centroids)
label_map = {order[0]: "negative", order[1]: "neutral", order[2]: "positive"}

df_twitter["sentiment"] = df_twitter["cluster"].map(label_map)
# ------------------------------------

print(df_twitter[["review_text","review_cleaned","compound","sentiment"]].head())


df_twitter["sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution (KMeans + VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

df_twitter.boxplot(column="compound", by="sentiment", grid=False)

plt.title("Compound Score by Sentiment Cluster")
plt.suptitle("")
plt.xlabel("Sentiment")
plt.ylabel("Compound Score")
plt.tight_layout()
plt.show()

# Set style
sns.set(style="whitegrid")

# Plot the clusters along the compound score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=range(len(df_twitter)), 
    y="compound", 
    hue="sentiment", 
    palette={"negative":"red", "neutral":"gray", "positive":"green"},
    data=df_twitter,
    s=50
)
plt.title("KMeans Clusters of Reviews (VADER Compound Scores)")
plt.xlabel("Review Index")
plt.ylabel("Compound Score")
plt.legend(title="Sentiment")
plt.show()

# --------------------------------------------------------------------
# ----------- WORDCLOUDS + TOP WORDS PER SENTIMENT CLUSTER FOR TWITTER -----------
# --------------------------------------------------------------------


# Define a function to get top words
def get_top_words(text_series, n=10):
    words = " ".join(text_series).split()
    counter = Counter(words)
    return counter.most_common(n)

# Plot word clouds for each sentiment cluster
sentiments = df_twitter["sentiment"].unique()
plt.figure(figsize=(18,6))

for i, sentiment in enumerate(sentiments, 1):
    text = df_twitter[df_twitter["sentiment"] == sentiment]["review_cleaned"]
    
    # Generate word cloud
    wordcloud = WordCloud(width=400, height=300, background_color="white").generate(" ".join(text))
    
    # Plot
    plt.subplot(1, len(sentiments), i)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment.capitalize()} Reviews")
    
    # Print top 10 words in console
    print(f"Top words in {sentiment} cluster:", get_top_words(text))

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Summary:
# The negative word cloud shows frustration with Elon/Musk, worsening updates, and strong complaints about 
# the platform becoming “bad,” “worse,” and “ruined.”
# The positive word cloud highlights appreciation for certain features and updates, with users expressing 
# satisfaction through words like good, great, best, and love.
# The neutral word cloud reflects general discussion around Elon/Musk, tweets, changes, and platform updates
# without strong emotional tone.
# ------------------------------------------------------------

# --------------------------------------------------------------------
# ----------- TRAIN-TEST SPLIT FOR TWITTER -----------
# --------------------------------------------------------------------


train_df_twitter, test_df_twitter = train_test_split(
    df_twitter, 
    test_size=0.2, 
    random_state=42,
    stratify=df_twitter['sentiment'] 
)
twitter_pos = df_twitter[df_twitter["sentiment"] == "positive"]
twitter_neu = df_twitter[df_twitter["sentiment"] == "neutral"]
twitter_neg = df_twitter[df_twitter["sentiment"] == "negative"]


twitter_pos_train = train_df_twitter[train_df_twitter["sentiment"] == "positive"]
twitter_neu_train = train_df_twitter[train_df_twitter["sentiment"] == "neutral"]
twitter_neg_train = train_df_twitter[train_df_twitter["sentiment"] == "negative"]

twitter_pos_test = test_df_twitter[test_df_twitter["sentiment"] == "positive"]
twitter_neu_test = test_df_twitter[test_df_twitter["sentiment"] == "neutral"]
twitter_neg_test = test_df_twitter[test_df_twitter["sentiment"] == "negative"]


# --------------------------------------------------------------------
# ----------- MAP AND STANDARDIZE MANUAL LABELS -----------
# --------------------------------------------------------------------

# Create mapping dictionary for typos and variations
label_mapping = {
    # Negative variations
    'Negative': 'negative',
    'negative': 'negative',
    
    # Positive variations
    'Positive': 'positive',
    
    # Neutral variations
    'Neutral': 'neutral',
}

# Apply mapping
df_twitter['sentiment_true_clean'] = df_twitter['sentiment_true'].map(label_mapping)

# Check for any unmapped values
unmapped = df_twitter[df_twitter['sentiment_true'].notna() & df_twitter['sentiment_true_clean'].isna()]['sentiment_true'].unique()
if len(unmapped) > 0:
    print(f"\nWARNING: Found unmapped values: {unmapped}")
    print("These will be treated as NaN. Please add them to the mapping if needed.\n")

print("\nAfter cleaning - Value counts for 'sentiment_true_clean':")
print(df_twitter['sentiment_true_clean'].value_counts())

print("\nLabels successfully standardized!")
print(f"  - Total labeled: {df_twitter['sentiment_true_clean'].notna().sum()}")
print(f"  - Negative: {(df_twitter['sentiment_true_clean'] == 'negative').sum()}")
print(f"  - Positive: {(df_twitter['sentiment_true_clean'] == 'positive').sum()}")
print(f"  - Neutral: {(df_twitter['sentiment_true_clean'] == 'neutral').sum()}")


# ----------------------------------------------------------------------------
# ----------- VADER RELIABILITY CHECK WITH CLEAN LABELS FOR TWITTER -----------
# ----------------------------------------------------------------------------

print("VADER SENTIMENT RELIABILITY CHECK")

# Filter non-null sentiment_true values
df_labeled = df_twitter[df_twitter['sentiment_true_clean'].notna()].copy()
print(f"\nTotal manually labeled samples: {len(df_labeled)}")
print(f"\nDistribution of manual labels:")
print(df_labeled['sentiment_true_clean'].value_counts())
print(f"\nDistribution of VADER predictions:")
print(df_labeled['sentiment'].value_counts())

# Calculate metrics
y_true = df_labeled['sentiment_true_clean']
y_pred = df_labeled['sentiment']

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("OVERALL METRICS:")
print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro):    {recall_macro:.4f}")
print(f"F1-Score (Macro):  {f1_macro:.4f}")

print("DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])

# Calculate percentages for each row
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('VADER Sentiment: Confusion Matrix (Counts)\n(True Labels vs VADER Predicted)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('VADER Predicted', fontsize=11)

# Percentages
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('VADER Sentiment: Confusion Matrix (Row %)\n(True Labels vs VADER Predicted)', 
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('VADER Predicted', fontsize=11)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
#Summary:  To evaluate the reliability of VADER sentiment classification for our Twitter dataset, 
# we manually labelled a subset of around 3k reviews as a validation set. VADER achieved an overall accuracy of 60.85% 
# with a macro F1-score of 0.54, showing reasonable performance as an unsupervised, lexicon-based baseline for social 
# media sentiment analysis.
#The model worked very well for for negative sentiment with precision (97.25%), indicating high confidence when 
# classifying negative reviews, though with moderate recall (48.95%). Positive sentiment detection was really accurate as well ,
#  with 86.70% recall and an F1-score of 0.77, suggesting VADER effectively captures positive expressions on Twitter data. 
# As expected with rule-based approaches, neutral sentiment proved challenging (F1: 0.19), a common issue when analysing 
# context-dependent langauge in social media. The manual labelling could also be dependent on an individual hence,
#  making it ambiguous for humans as well

#These results validate VADER as a suitable tool for initial sentiment labeling and could be used for sentiment classification. 
# ------------------------------------------------------------

# ------------------------------------------------------------
# Sentiment Classification using TF IDF + Logistic Regression (Twitter)
# ------------------------------------------------------------
# Use the full dataset with VADER+KMeans sentiment
df_vader_twi = df_twitter.copy()

# Only keep rows where VADER assigned a label (it should be all)
df_vader_twi = df_vader_twi[df_vader_twi["sentiment"].notna()]

print(df_vader_twi["sentiment"].value_counts())

tfidf_t = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=3,
    stop_words="english"
)

X_train_t = tfidf_t.fit_transform(train_df_twitter["review_cleaned"])
X_test_t = tfidf_t.transform(test_df_twitter["review_cleaned"])

y_train_t = train_df_twitter["sentiment"]
y_test_t = test_df_twitter["sentiment"]

model_t = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

model_t.fit(X_train_t, y_train_t)


y_pred_t = model_t.predict(X_test_t)

print("Accuracy:", accuracy_score(y_test_t, y_pred_t))
print("\nClassification Report:")
print(classification_report(y_test_t, y_pred_t, digits=4))

# ------------------------------------------------------------
# We trained a TF-IDF + Logistic Regression classifier on the Twitter dataset using VADER-assigned sentiment labels.
#  The model achieved 83.61% accuracy with a macro F1-score of 0.84, indicating strong overall performance and a clear 
# improvement over VADER’s direct predictions.

# Performance varies across sentiment categories:

# Positive sentiment was the most accurately detected (F1 = 0.88) and achieved the highest precision (0.91).
# Neutral sentiment showed solid performance (F1 = 0.80) with strong recall (0.83), indicating the model captures subtle,
#  mixed-tone expressions well.
# Negative sentiment also performed strongly (F1 = 0.82), showing that the classifier handles critical or complaint-oriented 
# language more effectively than simpler lexicon-based methods.
# Overall, these results demonstrate that TF-IDF + Logistic Regression is a highly competitive baseline for large-scale social
#  media sentiment analysis. Despite relying on weak, auto-generated labels, the model successfully captures meaningful linguistic
#  patterns in real-world Twitter text and significantly outperforms rule-based tools like VADER across all sentiment categories.
# ------------------------------------------------------------
# ===== Code by Rachana Part 2 end ===== #

# ===== Code by Haeyeon Part 4 start ===== #
# ------------------------------------------------------------
# Model 2: Sentiment Classification using DistilBERT (Twitter)
# ------------------------------------------------------------
# Encode sentiment labels for Twitter dataset
label_encoder_tw = LabelEncoder()

y_train_bert_tw = label_encoder_tw.fit_transform(train_df_twitter["sentiment"])
y_test_bert_tw  = label_encoder_tw.transform(test_df_twitter["sentiment"])

# Tokenize Text for BERT
# Convert to str
train_df_twitter["review_text"] = train_df_twitter["review_text"].astype(str)
test_df_twitter["review_text"]  = test_df_twitter["review_text"].astype(str)

# Remove broken variables
train_df_twitter = train_df_twitter[train_df_twitter["review_text"].notna()].copy()
test_df_twitter  = test_df_twitter[test_df_twitter["review_text"].notna()].copy()

tokenizer_tw = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_enc_tw = tokenizer_tw(
    train_df_twitter["review_text"].tolist(),    # raw text, no cleaning needed
    truncation=True,
    padding=True,
    max_length=128
)

test_enc_tw = tokenizer_tw(
    test_df_twitter["review_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# Build Dataset Objects, use previous class BERTDataset

train_dataset_tw = BERTDataset(train_enc_tw, y_train_bert_tw)
test_dataset_tw  = BERTDataset(test_enc_tw,  y_test_bert_tw)

# Load DistilBERT Model+ Define TrainingArguments + Trainer
model_tw = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

# ------------------------------------------------------------
# Hyperparameter tuning:
# - Initial learning_rate=1e-5 by Haeyeon
# - Tuned learning_rate=2e-5 by Rachna
# ------------------------------------------------------------
training_args_tw = TrainingArguments(
    output_dir="./bert_results_twitter",
    eval_strategy="epoch", # evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5, # Initial : 1e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    seed=SEED
)

trainer_tw = Trainer(
    model=model_tw,
    args=training_args_tw,
    train_dataset=train_dataset_tw,
    eval_dataset=test_dataset_tw
)

# Train and Evaluation
trainer_tw.train()

preds_tw = trainer_tw.predict(test_dataset_tw)
y_pred_bert_tw = np.argmax(preds_tw.predictions, axis=1)

print("Twitter BERT Accuracy:", accuracy_score(y_test_bert_tw, y_pred_bert_tw))

print(classification_report(
    y_test_bert_tw,
    y_pred_bert_tw,
    target_names=label_encoder_tw.classes_,
    digits=4
))

# ------------------------------------------------------------
#[Summary]
# Fine-tuning DistilBERT on the Twitter dataset led to a substantial improvement in sentiment classification performance.
# The model achieved 93.06%(Previously 91.8%) accuracy and a macro F1-score of 0.93,surpassing the TF-IDF + Logistic Regression baseline (macro F1 ≈0.84).
# Performance was consistently strong across all three sentiment classes:
# - Negative: F1 = 0.93
# - Neutral: F1 = 0.90
# - Positive: F1 = 0.96
# 
# These results show that DistilBERT captures the linguistic nuances of Twitter text extremely well, even when trained on weakly labeled sentiment data. Compared to traditional machine-learning models and lexicon-based methods, the transformer model demonstrates far more robust generalization, stronger class balance, and improved handling of subtle or ambiguous sentiment expressions.
# Overall, DistilBERT provides a highly reliable and accurate approach for large-scale Twitter sentiment analysis.
# ------------------------------------------------------------
# ===== Code by Haeyeon Part 4 end ===== #

# ===== Code by Snehitha Tadapaneni Part 2 start ===== #
# --------------------------------------------------------------------
# --------- DATA CLEANING PIPELINE 2: TOPIC MODELLING PREPARATION -----------
# --------------------------------------------------------------------
negations = {"no", "not", "nor", "dont", "can't", "cannot", "never"}
stop_words = stop_words - negations

# App-specific words to remove
custom_stopwords_tm = set([
    "app", "apps", "application", "applications",
    "experience", "account", 
])

# ============================
#  Cleaning Function
# ============================
def clean_text_for_tm(text):
    tokens = simple_preprocess(text, deacc=True)
    return tokens

# ============================
#  Bigram Builder
# ============================
def create_bigrams(texts):
    bigram = Phrases(texts, min_count=10, threshold=10)
    bigram_mod = Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

# ============================
# Prepare Dataset for Topic Modeling
# ============================
def prepare_tm_texts(df):
    docs = df["review_cleaned"].apply(clean_text_for_tm).tolist()
    docs_bigrams = create_bigrams(docs)
    return docs_bigrams

# ------- TOPIC MODELLING 1: LDA -----------
def train_lda_model(docs_bigrams, num_topics=5):

    dictionary = Dictionary(docs_bigrams)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in docs_bigrams]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=2000,
        passes=10,
        alpha='auto'
    )

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=docs_bigrams,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_score = coherence_model.get_coherence()

    return lda_model, corpus, dictionary, coherence_score



def run_lda(docs_df):
    docs_bigrams = prepare_tm_texts(docs_df)

    topic_range = range(3, 7)
    coherence_scores = {}

    for k in topic_range:
        lda_model, corpus, dictionary, coherence = train_lda_model(docs_bigrams, num_topics=k)
        coherence_scores[k] = coherence
        print(f"k={k} Coherence={coherence:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    print("\nBest number of topics =", best_k)

    final_lda, corpus, dictionary, coherence = train_lda_model(docs_bigrams, num_topics=best_k)

    print("Final Coherence Score:", coherence)

    topics = final_lda.show_topics(num_topics=-1, num_words=15, formatted=False)

    for topic_id, words in topics:
        print(f"TOPIC {topic_id+1}:")
        top_terms = [w for w, weight in words]
        print(", ".join(top_terms))
        print()

    return final_lda    # 


print("\n===== POSITIVE TOPICS =====")
def run_lda_block(df_input):
    print("\n===== RUNNING LDA BLOCK =====")
    return run_lda(df_input)


# Visualization function for LDA topics
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# 4×4 Grid Visualization for One Sentiment
# ---------------------------------------------------------
def visualize_lda_grid(lda_model, sentiment_label, num_words=10):

    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    num_topics = len(topics)

    print(f"\n--- VISUALIZING {num_topics} TOPICS FOR {sentiment_label.upper()} ---")

    # Grid size (max 4x4)
    rows = 4
    cols = 4
    total_plots = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()

    for i, ax in enumerate(axes):

        if i < num_topics:
            topic_id, topic_data = topics[i]
            words = [w for w, wt in topic_data]
            weights = [wt for w, wt in topic_data]

            ax.barh(words[::-1], weights[::-1])
            ax.set_title(f"Topic {topic_id+1}")
            ax.tick_params(labelsize=9)
        else:
            ax.axis("off")   # hide empty grid cells

    plt.suptitle(f"{sentiment_label.upper()} — LDA Topics", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Wrapper for convenience
def visualize_lda_block(lda_model, sentiment_label):
    visualize_lda_grid(lda_model, sentiment_label)


# ---------------------------------------------------------
# --------- THREADS LDA TOPIC MODELING ---------
# ---------------------------------------------------------
print("\n===== THREADS POSITIVE TOPICS =====")
lda_pos_threads = run_lda_block(threads_pos)
print("\n===== THREADS NEUTRAL TOPICS =====")
lda_neu_threads = run_lda_block(threads_neu)
print("\n===== THREADS NEGATIVE TOPICS =====")
lda_neg_threads = run_lda_block(threads_neg)

visualize_lda_block(lda_pos_threads, "Threads Positive")
visualize_lda_block(lda_neu_threads, "Threads Neutral")
visualize_lda_block(lda_neg_threads, "Threads Negative")

# Sumarry of Findings for Threads LDA
# - Positive reviews: Users express generally positive impressions and note that the app works well, while also offering polite suggestions for additional features.
# - Neutral reviews: Comments focus on platform comparisons, technical observations, and practical usage notes without strong emotional tone.
# - Negative reviews: Users report dissatisfaction related to crashes, missing features, feed issues, and comparisons where Threads feels weaker than alternatives.

# ---------------------------------------------------------
# --------- TWITTER LDA TOPIC MODELING ---------
# ---------------------------------------------------------
print("\n===== TWITTER POSITIVE TOPICS =====")
lda_pos_twitter = run_lda_block(twitter_pos)

print("\n===== TWITTER NEUTRAL TOPICS =====")
lda_neu_twitter = run_lda_block(twitter_neu)

print("\n===== TWITTER NEGATIVE TOPICS =====")
lda_neg_twitter = run_lda_block(twitter_neg)

visualize_lda_block(lda_pos_twitter, "Twitter Positive")
visualize_lda_block(lda_neu_twitter, "Twitter Neutral")
visualize_lda_block(lda_neg_twitter, "Twitter Negative")

# Summary of Findings for Twitter LDA
# - Positive reviews: Users express favorable impressions, highlight improvements, and appreciate specific features or platform behavior.
# - Neutral reviews: Comments focus on platform changes, updates, and general functionality, often mentioning rebranding or feature adjustments without strong emotion.
# - Negative reviews: Users express dissatisfaction with updates, rebranding decisions, and limits, frequently criticizing how recent changes have affected the platform experience.

# ===== Code by Snehitha Tadapaneni Part 2 end ===== #

# ===== Code by Rachana Part 3 start ===== #

# --------------------------------------------------------------------
# --------- TOPIC MODELLING PREPARATION - NMF -----------
# --------------------------------------------------------------------


def run_nmf(docs_df):
    docs_bigrams = prepare_tm_texts(docs_df)
    docs_text = [" ".join(doc) for doc in docs_bigrams]

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=10,
        ngram_range=(1, 1),
    )

    tfidf = tfidf_vectorizer.fit_transform(docs_text)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    dictionary = Dictionary(docs_bigrams)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    topic_range = range(3, 10)
    coherence_scores = {}

    for k in topic_range:
        nmf_model = NMF(
            n_components=k,
            random_state=42,
            init="nndsvda",
            max_iter=2000
        )
        
        W = nmf_model.fit_transform(tfidf)
        H = nmf_model.components_

        top_words = []
        for topic in H:
            idxs = topic.argsort()[-20:]
            top_words.append([feature_names[i] for i in idxs])

        coherence_model = CoherenceModel(
            topics=top_words,
            texts=docs_bigrams,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence = coherence_model.get_coherence()
        coherence_scores[k] = coherence
        print(f"k={k} Coherence={coherence:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    print("\nBest number of NMF topics =", best_k)

    final_nmf = NMF(
        n_components=best_k,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )

    W_final = final_nmf.fit_transform(tfidf)
    H_final = final_nmf.components_

    print("\n--- FINAL NMF TOPIC WORDS ---\n")
    for idx, topic in enumerate(H_final):
        indices = topic.argsort()[-15:]
        words = [feature_names[i] for i in indices]
        print(f"TOPIC {idx+1}: {', '.join(words)}")

    return final_nmf, feature_names



def run_nmf_block(df_input):
    print("\n===== RUNNING NMF BLOCK =====")
    return run_nmf(df_input)


#Visulaizing the Topics in pos, neu, neg (NMF)

# --------------------------------------------------------------------
# VISUALIZATION - NMF
# --------------------------------------------------------------------


def plot_topic(words, weights, title):
    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], weights[::-1], color='royalblue')
    plt.title(title)
    plt.xlabel("Topic Weight")
    plt.tight_layout()
    plt.show()

def visualize_nmf_topics(nmf_model, feature_names, sentiment_label):
    print(f"\n--- VISUALIZING NMF TOPICS: {sentiment_label.upper()} ---")
    
    H = nmf_model.components_

    for idx, topic in enumerate(H):
        top_idx = topic.argsort()[-15:][::-1]
        words = [feature_names[i] for i in top_idx]
        weights = [topic[i] for i in top_idx]

        title = f"{sentiment_label.upper()} - NMF Topic {idx+1}"
        plot_topic(words, weights, title)

def visualize_nmf_block(nmf_model, feature_names, sentiment_label):
    visualize_nmf_topics(nmf_model, feature_names, sentiment_label)

# --------------------------------------------------------------------
# NMF THREADS ANALYSIS 
# --------------------------------------------------------------------

print("\n===== NMF THREADS POSITIVE =====")
nmf_pos, pos_feats = run_nmf_block(threads_pos)

print("\n===== NMF THREADS NEUTRAL =====")
nmf_neu, neu_feats = run_nmf_block(threads_neu)

print("\n===== NMF THREADS NEGATIVE =====")
nmf_neg, neg_feats = run_nmf_block(threads_neg)

visualize_nmf_block(nmf_pos, pos_feats, "Threads Positive")
visualize_nmf_block(nmf_neu, neu_feats, "Threads Neutral")
visualize_nmf_block(nmf_neg, neg_feats, "Threads Negative")

#Observations:
# Positive Sentiment (Best k = 6) :NMF produced six meaningful topics reflecting various aspects 
# of positive user experience.
# General Impressions - new, use, feature, need
# Interface & Platform Notes - see, ui, platform
# Competitor Comparisons - facebook, instagram, alternative
# Meta Ecosystem Context - social, platform, world
# Functionality Mentions - work, job, start, feature
# Light Positive Reactions - cool, wow, amazing, love

# Neutral Sentiment (Best k = 6): Neutral topics were clearer and more structured than LDA,
#  with well-defined clusters.
# Copying/Clone Commentary - cheap, clone, copied, twitter
# Account & Login Activities - login, delete, create, sign
# Competitor Mentions - mark, zuck, tweeter
# Feed & Usage Observations - feed, see, post, work
# Minor Technical Issues - glitching, bug, not_working
# Routine App Interactions - write, review, comment, time

# Negative Sentiment (Best k = 9): Because negative reviews mention diverse frustrations, 
# NMF discovered nine separate issue clusters, reflecting greater complexity than LDA.
# App Not Working - not_working, glitch, install, ui
# UI & Privacy Concerns - screen, privacy, content
# Missing Functions - post, see, need, delete
# Strong Dissatisfaction - rubbish, pathetic, useless
# Copying/Clone Complaints - copy, copying, cheap, copy_twitter
# Design Issues - boring, nothing_new, poor
# Feed/Discovery Problems - feed, trending, hashtags
# Quality & Competitor Comparison - clone, fake, twitter
# Technical Failures - crash, upload, picture, try


# --------------------------------------------------------------------
# NMF TWITTER ANALYSIS
# --------------------------------------------------------------------


print("\n===== NMF TWITTER POSITIVE =====")
nmf_pos_tw, pos_feats_tw = run_nmf_block(twitter_pos)

print("\n===== NMF TWITTER NEUTRAL =====")
nmf_neu_tw, neu_feats_tw = run_nmf_block(twitter_neu)
print("\n===== NMF TWITTER NEGATIVE =====")
nmf_neg_tw, neg_feats_tw = run_nmf_block(twitter_neg)

visualize_nmf_block(nmf_pos_tw, pos_feats_tw, "Twitter Positive")
visualize_nmf_block(nmf_neu_tw, neu_feats_tw, "Twitter Neutral")
visualize_nmf_block(nmf_neg_tw, neg_feats_tw, "Twitter Negative")

#Observations:
# Positive Sentiment (k ≈ 3) :
# General Enjoyment — love, excellent, best_social
# Feature Appreciation / Improvements — better, awesome, free_speech
# Satisfaction with Updates — good, great, name, change, logo


# Neutral Sentiment (k ≈ 6) :
# Leadership & Feature Mentions — elon_musk, feature
# Platform Behavior — ruined, platform, anymore
# Tweet/Video/Limit Notes — tweet, limit, video
# Rebranding Discussions — bird, back, rebranding
# Update Notes — update, phone
# Logo/Name Change — change, name, logo

# Negative Sentiment (k ≈ 3) :
# Update-Related Frustrations — no, worse, limit, post
# Rebranding Disapproval — suck, name, logo, bird
# Leadership Criticism — bad, elon, ruined, musk

# NMF seems to have decent coherence for positive neutral and negative around 0.40 but 
# it is significantly low than LDA and the insights found in LDA seem to be much better than NMF. 
# Hence, LDA is the best choice for now.

# ===== Code by Rachana Part 3 end ===== #

# ===== Code by Haeyeon Part 5 start ===== #
# --------------------------------------------------------------------
# --------- TOPIC MODELLING PREPARATION - BERTopic -----------
# --------------------------------------------------------------------

# Prevent text truncation in DataFrame printing
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 80)

# Shared embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Helper: Horizontal bar plot for single topic
# -----------------------------
def plot_topic(words, weights, title):
    plt.figure(figsize=(10,5))
    plt.barh(words[::-1], weights[::-1], color="royalblue")
    plt.title(title)
    plt.xlabel("Topic Weight")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Helper: BERTopic summary
# -----------------------------
def inspect_topics(model, texts, group_name):
    print(f"\n===== BERTOPIC SUMMARY ({group_name.upper()}) — 5 TOPICS =====")
    #topic_info = model.get_topic_info()
    topic_info = model.get_topic_info().copy()
    topic_info["Representative_Docs"] = topic_info["Representative_Docs"].apply(
        lambda docs: "\n".join(docs)
    )
    print(topic_info)

    # Topic distribution barplot
    plt.figure(figsize=(8,4))
    sns.barplot(x="Topic", y="Count", data=topic_info[topic_info["Topic"] != -1])
    plt.title(f"Topic Distribution ({group_name.capitalize()} Reviews)")
    plt.tight_layout()
    plt.show()

    # Plot top words per topic
    for tid in topic_info["Topic"]:
        if tid == -1:
            continue
        words_scores = model.get_topic(tid)
        words = [w for w, score in words_scores]
        weights = [score for w, score in words_scores]
        plot_topic(words, weights, title=f"{group_name.upper()} - BERTopic {tid}")

    # Representative examples
    print(f"\n--- REPRESENTATIVE EXAMPLES ({group_name}) ---")
    repr_docs = model.get_representative_docs()
    for tid, docs in repr_docs.items():
        if tid == -1:
            continue
        print(f"\nTopic {tid} Examples:")
        for d in docs[:3]:
            print("-", d[:200], "...")

# -----------------------------
# Helper: compute BERTopic coherence
# -----------------------------
def get_bertopic_top_words(model, top_n=15):
    topic_words = []
    for t in model.get_topics().keys():
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:top_n]]
        topic_words.append(words)
    return topic_words

def tokenize_texts(texts):
    return [t.split() for t in texts]

def compute_bertopic_coherence(model, texts, top_n=15):
    tokenized = tokenize_texts(texts)
    word_lists = get_bertopic_top_words(model, top_n)
    dictionary = Dictionary(tokenized)
    cm = CoherenceModel(
        topics=word_lists,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v"
    )
    return cm.get_coherence()

# -----------------------------
# BERTopic Pipeline
#  - Run BERTopic for a given sentiment subset of a DataFrame.
#  - Returns the fitted model, topics, probabilities, embeddings, and coherence score.
# -----------------------------
def run_bertopic_pipeline(df, sentiment_col="sentiment", review_col="review_description",
                          sentiment_value=None, min_topic_size=20, reduce_to=5):

    # Filter data
    subset = df[df[sentiment_col] == sentiment_value].copy()

    # Remove NaN values and convert to string
    subset = subset[subset[review_col].notna()].copy()
    subset[review_col] = subset[review_col].astype(str)
    texts = subset[review_col].tolist()

    texts = [t for t in texts if t.strip() != ""]

    # Compute embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Fit BERTopic
    model = BERTopic(
        language="english",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size,
        top_n_words=15,
        calculate_probabilities=True,
        verbose=False
    )

    topics, probs = model.fit_transform(texts, embeddings)

    # Reduce topics
    if reduce_to is not None:
        model = model.reduce_topics(texts, nr_topics=reduce_to)
        topics, probs = model.transform(texts, embeddings)

    # Inspect topics
    inspect_topics(model, texts, sentiment_value.capitalize())

    # Compute coherence
    coherence = compute_bertopic_coherence(model, texts)

    print(f"{sentiment_value.capitalize()} BERTopic Coherence:", coherence)

    return model, topics, probs, embeddings, coherence

# --------------------------------------------------------------------
# Helper: grouped plot
# --------------------------------------------------------------------
def plot_bertopic_topics_grouped(model, group_name="BERTopic Topics", top_n=15):
    topics = model.get_topics()
    topic_ids = [tid for tid in topics.keys() if tid != -1]

    # Only 3 topics per row
    cols = 3
    rows = math.ceil(len(topic_ids) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, tid in enumerate(topic_ids):
        words_scores = topics[tid][:top_n]
        words = [w for w, _ in words_scores]
        weights = [s for _, s in words_scores]

        sns.barplot(x=weights, y=words, ax=axes[i], color="royalblue")
        axes[i].set_title(f"{group_name} — Topic {tid}")
        axes[i].set_xlabel("Weight")
        axes[i].set_ylabel("Word")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------
# BERTopic Threads analysis
# --------------------------------------------------------------------
threads_pos_model, threads_pos_topics, threads_pos_probs, threads_pos_emb, threads_pos_coh = \
    run_bertopic_pipeline(df_threads, sentiment_value="positive")

threads_neu_model, threads_neu_topics, threads_neu_probs, threads_neu_emb, threads_neu_coh = \
    run_bertopic_pipeline(df_threads, sentiment_value="neutral")

threads_neg_model, threads_neg_topics, threads_neg_probs, threads_neg_emb, threads_neg_coh = \
    run_bertopic_pipeline(df_threads, sentiment_value="negative")
'''
# For grouped plot
plot_bertopic_topics_grouped(threads_pos_model, "Threads Positive")
plot_bertopic_topics_grouped(threads_neu_model, "Threads Neutral")
plot_bertopic_topics_grouped(threads_neg_model, "Threads Negative")
'''
# --------------------------------------------------------------------
# Observations from Threads:
# Threads Positive
# - Users like the app’s vibe and easy interface.
# - Many say it feels similar to Twitter/Instagram.
# - Some ask for missing features like follow-only feed and messaging.
#
# Threads Neutral
# - Comments mention missing features, bugs, and crashes.
# - Some users say the app is a copy of Twitter.
# - Mixed opinions without strong emotion.
#
# Threads Negative
# - Complaints about crashes, bugs, and feed problems.
# - Users feel the app is unstable or hard to use.
# -Some are unhappy about links to Instagram accounts.
# --------------------------------------------------------------------
#
# --------------------------------------------------------------------
# BERTopic Twitter analysis
# --------------------------------------------------------------------
twitter_pos_model, twitter_pos_topics, twitter_pos_probs, twitter_pos_emb, twitter_pos_coh = \
    run_bertopic_pipeline(twitter_pos, sentiment_value="positive", review_col= "review_text")

twitter_neu_model, twitter_neu_topics, twitter_neu_probs, twitter_neu_emb, twitter_neu_coh = \
    run_bertopic_pipeline(twitter_neu, sentiment_value="neutral", review_col = "review_text")

twitter_neg_model, twitter_neg_topics, twitter_neg_probs, twitter_neg_emb, twitter_neg_coh = \
    run_bertopic_pipeline(twitter_neg, sentiment_value="negative",review_col="review_text")
'''
# For grouped plot
plot_bertopic_topics_grouped(twitter_pos_model, "Twitter Positive")
plot_bertopic_topics_grouped(twitter_neu_model, "Twitter Neutral")
plot_bertopic_topics_grouped(twitter_neg_model, "Twitter Negative")
'''
# --------------------------------------------------------------------
# Observations from Twitter:
# Twitter Positive
# - Users praise the app and say it is fun or useful.
# - Some support recent changes and mention free speech.
# - A few reviews talk about account help or recovery.
#
# Twitter Neutral
# - Comments about the name change to “X.”
# - Mentions of policy changes like rate limits or paid features.
# - General observations, not strongly positive or negative.
#
# Twitter Negative
# - Complaints about rate limits and new restrictions.
# - Users dislike some updates and the rebranding.
# - Short negative reactions like “bad,” “dumpster fire,” etc.
# --------------------------------------------------------------------
#===== Code by Haeyeon Part 5 end ===== #

