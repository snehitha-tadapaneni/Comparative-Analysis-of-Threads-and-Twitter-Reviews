# FinalProject-GroupCorpuscrew

# Comparative Analysis of Threads and Twitter Reviews

This project implements a complete NLP pipeline to analyze user reviews from **Threads** and **Twitter**.  
The workflow includes data preprocessing, sentiment analysis (VADER, Logistic Regression, DistilBERT),  
and topic modeling (LDA, NMF, BERTopic).

All code is contained in a single script/notebook and should be executed **top to bottom**.

---

## 1. Installation & Requirements

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

