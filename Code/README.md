# FinalProject-GroupCorpuscrew
# FinalProject-GroupCorpuscrew

# Comparative Analysis of Threads and Twitter Reviews

This project implements a complete NLP pipeline to analyze user reviews from **Threads** and **Twitter**.  
The workflow includes data preprocessing, sentiment analysis (VADER, Logistic Regression, DistilBERT),  
and topic modeling (LDA, NMF, BERTopic).

All code is contained in a single script/notebook and should be executed **top to bottom**.

---

## 1. Project Structure

project_root/
- Data/
  - threads_reviews_labelled.csv
  - twitter_reviews_labelled.csv

- comparative_analysis_threads_twitter.py
- README.md

---

## 2. Installation & Requirements

Install required packages:
    pip install pandas numpy nltk emoji langdetect vaderSentiment scikit-learn gensim wordcloud seaborn matplotlib torch transformers sentence-transformers umap-learn hdbscan bertopic

### Requirements

This project was developed and tested with the following versions:

- **Python:** 3.11  
- **Transformers:** 4.33.3  
- **Datasets:** 2.14.6  
- **HuggingFace Hub:** 0.19.4  
- **Accelerate:** 0.23.0  
- **BERTopic:** 0.15.0  
- **Sentence-Transformers:** 2.2.2  
- **NumPy:** 1.26.4  
- **PyArrow:** 12.0.1  
- **Scikit-Learn:** 1.3.2  


Download NLTK resources:
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


---

## 3. Execution Order (Run in This Exact Sequence)

### 3.1 Global Setup
- Import all libraries  
- Set global seeds  
- Configure UMAP + HDBSCAN  
- Disable tokenizer parallelism  
- Load required NLTK packages  

**Must be executed first.**

---

### 3.2 Load the Datasets
- Load `threads_reviews_labelled.csv`  
- Load `twitter_reviews_labelled.csv`  
- Print `.head()`, `.info()`, and missing value statistics  

**Output:** Raw Threads and Twitter dataframes.

---

### 3.3 Data Cleaning — Threads
Performs:
- Raw duplicate removal  
- Long-review duplicate removal  
- Brand normalization (e.g., twitter/twt/x → twitter)  
- Text cleaning (URLs, emojis, punctuation, tokenizing, lemmatization)  
- Stopword filtering (negations preserved)  
- Removal of empty cleaned rows  
- Language detection + filtering  

**Outputs:**  
- `threads_clean`  
- `threads_pos`, `threads_neu`, `threads_neg` (after VADER)

---

### 3.4 Data Cleaning — Twitter
Same pipeline adapted for Twitter:
- Duplicate removal  
- Brand normalization  
- Text preprocessing  
- Language filtering  

**Outputs:**  
- `twitter_clean`  
- `twitter_pos`, `twitter_neu`, `twitter_neg` (after VADER)

---

## 4. Sentiment Analysis Pipeline

### 4.1 VADER + KMeans Clustering (Threads & Twitter)
- Computes VADER compound score  
- Applies KMeans (k=3) to auto-define sentiment thresholds  
- Maps clusters to: **negative**, **neutral**, **positive**  
- Generates:
  - Sentiment distribution plot  
  - Boxplots  
  - Scatter plot of clusters  
  - Word clouds  

**Outputs:**  
`df_threads["sentiment"]`  
`df_twitter["sentiment"]`

---

### 4.2 Manual Label Cleaning & Reliability Check
- Standardizes `sentiment_true` → `sentiment_true_clean`  
- Compares VADER predictions to human labels  
- Generates:
  - Accuracy  
  - Precision / Recall / F1  
  - Confusion matrices (counts + %)  

**Purpose:** Evaluate the reliability of VADER as a weak labeler.

---

### 4.3 TF–IDF + Logistic Regression
- Vectorizes `review_cleaned` using TF–IDF  
- Trains Logistic Regression on VADER labels  
- Evaluates:
  - Accuracy  
  - Classification report  

**Purpose:** Classical ML baseline sentiment classifier.

---

### 4.4 DistilBERT Fine-Tuning
- Tokenizes text using DistilBERT tokenizer  
- Encodes labels  
- Loads `DistilBertForSequenceClassification`  
- Trains model using HuggingFace Trainer  
- Uses best hyperparameters tuned via Optuna  
- Evaluates accuracy & class-wise F1  

**Purpose:** Best-performing sentiment model in this pipeline.

---

## 5. Topic Modeling Pipeline

### 5.1 Topic Modeling Preprocessing
Shared steps for LDA and NMF:
- Clean text for topic modeling  
- Tokenize with `simple_preprocess`  
- Create bigram phrases  
- Produce token lists per document  

---

### 5.2 LDA Topic Modeling (Threads & Twitter)
For each sentiment subset:
- Try multiple topic numbers (k = 3–6)  
- Compute coherence score (c_v)  
- Choose best-k  
- Train final LDA model  
- Print top words per topic  
- Visualize topics using bar charts  

**Outputs:**  
- `lda_pos_*`, `lda_neu_*`, `lda_neg_*`

---

### 5.3 NMF Topic Modeling (Threads & Twitter)
Steps:
- Create TF–IDF matrix  
- Train NMF for k = 3–10  
- Compute coherence for each k  
- Pick best-k  
- Display top topic words  
- Visualize with bar charts  

**Outputs:**  
- `nmf_pos_*`, `nmf_neu_*`, `nmf_neg_*`

---

### 5.4 BERTopic Modeling (Threads & Twitter)
- Generates sentence embeddings using MiniLM  
- Applies UMAP dimensionality reduction  
- Clusters using HDBSCAN  
- Reduces to ~5 final topics  
- Displays:
  - Topic distribution  
  - Top words  
  - Representative example documents  
  - Coherence score  

**Outputs:**  
- `threads_pos_model`, `threads_neu_model`, `threads_neg_model`  
- `twitter_pos_model`, `twitter_neu_model`, `twitter_neg_model`  
- Coherence values for each model

---

## 6. Final Outputs Produced

Running the entire workflow yields:
- Cleaned and language-filtered datasets  
- Sentiment predictions (VADER, TF–IDF, DistilBERT)  
- Confusion matrices  
- Topic modeling results (LDA, NMF, BERTopic)  
- Topic word lists & visualizations  
- Platform-level sentiment and theme comparison  

---

## 7. How to Run Everything

### If using a Python script:
    python/python3 comparative_analysis_threads_twitter.py







