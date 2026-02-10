# 🧠 Predicting Depression from Social Media Posts  
**San Diego State University**

**Author:**  
- Anh Huy Nguyen  

---

## 📌 Overview
This project explores whether **depression can be detected from social media language** using Natural Language Processing (NLP) and Machine Learning. We analyze **Reddit posts and comments** from a large, labeled mental-health dataset and compare **traditional ML baselines** against a **transformer-based deep learning model (DistilBERT)**.

The goal is **early risk detection**: identifying posts that may indicate depressive tendencies, which could support scalable **digital mental-health screening tools**.  
This work is **research-oriented** and intended strictly for **screening and analysis**, not diagnosis.

---

## 🎯 Research Questions
- Can language alone reliably indicate depressive tendencies?
- How do **classical ML models** compare with **transformer-based models**?
- Which linguistic patterns most strongly correlate with depression?
- What trade-offs exist between **precision vs. recall** in mental-health screening?

---

## 📊 Dataset
**Reddit Mental Health Dataset (Kaggle)**  
- ~1.3 million posts & comments  
- Multiple mental-health labels (depression, control, anxiety, etc.)

**Key columns:**
| Column | Description |
|------|------------|
| `post_id` | Unique Reddit post ID |
| `subreddit` | Subreddit name |
| `title` | Post title |
| `body` | Full post text |
| `author` | Reddit username |
| `created_utc` | Timestamp |
| `label` | Mental-health category |

Dataset source:  
https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset

---

## 🧩 Project Pipeline
Our end-to-end workflow follows a **six-stage NLP pipeline**:

### 1️⃣ Data Collection
- Download labeled Reddit posts from Kaggle
- Filter relevant mental-health categories

### 2️⃣ Text Preprocessing
- Lowercasing  
- Punctuation & number removal  
- Tokenization  
- Stopword removal  
- Lemmatization  

Libraries used: `NLTK`, `spaCy`

### 3️⃣ Feature Extraction
We experiment with multiple text representations:
- **TF-IDF** (baseline)
- **Bag of Words**
- **Transformer embeddings (DistilBERT)**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['body'])
```

### 4️⃣ Model Training
Models evaluated:
- Logistic Regression
- Random Forest
- Naive Bayes
- **DistilBERT (fine-tuned)**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5️⃣ Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- PR-AUC (important for class imbalance)

### 6️⃣ Interpretation & Reporting
- Error analysis on misclassified posts  
- Threshold tuning for high-recall screening  
- Visualization of results  

---

## 📈 Key Results
- **Best overall model:** Fine-tuned **DistilBERT**
- **Major gain:** Improved **recall** on depression posts compared to TF-IDF baselines
- **Strong baseline:** TF-IDF + Logistic Regression performed surprisingly well
- **Most errors:** Short, sarcastic, or context-poor posts

### 🔍 Takeaway
> Transformer models offer **consistent but modest gains** over strong classical baselines, especially for **borderline depression cases**.

---

## ⚠️ Ethical Considerations
- This model is **not a diagnostic tool**
- False positives are preferable to false negatives in screening
- Outputs should **never replace professional clinical judgment**
- Privacy and consent are critical when working with mental-health data

---

## 🛠️ Tech Stack
- **Python**
- **scikit-learn**
- **NLTK / spaCy**
- **PyTorch**
- **Hugging Face Transformers**
- **Pandas / NumPy / Matplotlib**

---

## 📚 References
1. Coppersmith et al. (2015). *Analyzing the Language of Mental Health on Twitter*. ACL  
2. De Choudhury et al. (2013). *Predicting Depression via Social Media*. AAAI  
3. Devlin et al. (2019). *BERT*. NAACL-HLT  
4. Sanh et al. (2019). *DistilBERT*. arXiv  
5. Kaggle (2022). *Reddit Mental Health Dataset*

---

## 📌 Disclaimer
This project is for **academic research purposes only**.  
It does **not** provide medical advice, diagnosis, or treatment.
