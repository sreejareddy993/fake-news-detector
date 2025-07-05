import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
fake_df = pd.read_csv("Fake.csv", encoding='ISO-8859-1')
real_df = pd.read_csv("True.csv", encoding='ISO-8859-1')
fake_df["label"] = 0
real_df["label"] = 1
data = pd.concat([fake_df, real_df])
data = data.sample(frac=1).reset_index(drop=True)

# Train model
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Streamlit App
st.title("📰 Fake News Detection App")
st.write("Paste any news content below and the model will tell you whether it's Fake or Real.")

# User input
user_input = st.text_area("Enter News Article Content Here", height=250)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        if prediction == 0:
            st.error("🛑 This news is **Fake**!")
        else:
            st.success("✅ This news is **Real**!")
