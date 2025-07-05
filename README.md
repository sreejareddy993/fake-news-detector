# 📰 Fake News Detection Web App

A machine learning-powered web application that detects whether a given news article is **Fake** or **Real** using NLP techniques.

---

## 🚀 Live Demo

👉 (https://sreejareddy993-fake-news-detector.streamlit.app)

---

## 📌 Features

- 🔍 Input any news content and check its authenticity
- ⚙️ Trained on thousands of real and fake news articles
- 🧠 Uses TF-IDF vectorization and Logistic Regression
- 🌐 Built with Streamlit — accessible via web browser
- 📊 Achieves high accuracy on test data

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Programming language |
| **Pandas** | Data cleaning and manipulation |
| **Scikit-learn** | Machine Learning (TF-IDF, Logistic Regression) |
| **Streamlit** | Web App framework |
| **GitHub** | Version control & public hosting |
| **Streamlit Cloud** | Free web deployment |

---

## 📁 Dataset Used

- Fake and real news articles from [Kaggle Fake News Dataset (ISOT)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Combined `Fake.csv` and `True.csv`, labeled as `0` and `1` respectively

---

## 📊 Model Training

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
