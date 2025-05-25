# 📚 Sentiment Analysis on Amazon Kindle Reviews

## 🚀 Project Overview

This project focuses on **Sentiment Analysis** of **Amazon Kindle product reviews**. The goal is to classify user reviews as **Positive** & **Negative**,helping businesses and researchers understand customer satisfaction, trends, and feedback.

By leveraging **Natural Language Processing (NLP)** techniques and **Machine Learning (ML)** models, we extract meaningful insights from unstructured text data and build a robust system that predicts the sentiment of unseen Kindle reviews.

---

## 📊 Dataset

- **Source**: [Amazon Product Review Dataset (Kindle Store)
- **Size**: ~1 million reviews (subset used for this project)
- **Fields**:
  - `reviewText`: The text of the review
  - `overall`: Rating (1-5 stars)
  - `summary`: Short summary of the review
  - `reviewerID`, `asin`, `helpful`, etc.

**Sentiment Labeling**:
- **1-2-3 stars** → Negative

- **4-5 stars** → Positive

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Lowercasing text
- Removing punctuation, special characters, numbers
- Stopword removal
- Lemmatization/Stemming

### 2️⃣ Feature Engineering
- **TF-IDF (Term Frequency - Inverse Document Frequency)**
- **Word2Vec embeddings** (optional for advanced modeling)

### 3️⃣ Modeling
- Logistic Regression
- Naive Bayes Classifier
- Random Forest
- (Optional) LSTM/GRU with Word2Vec embeddings
- (Optional) BERT fine-tuning for advanced modeling

### 4️⃣ Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🧰 Tools & Libraries

| Tool/Library        | Purpose                        |
|---------------------|--------------------------------|
| Python              | Programming Language           |
| Pandas, NumPy       | Data Manipulation              |
| NLTK, spaCy         | Text Preprocessing             |
| Scikit-learn        | Machine Learning Models        |
| Matplotlib, Seaborn | Visualization                  |
| Gensim (optional)   | Word2Vec Embeddings            |
| PyTorch/Keras (optional) | Deep Learning Models       |
| Jupyter Notebook    | Prototyping and Analysis       |

---

## 📈 Results

| Model                      | Accuracy (%) | F1-Score (Macro) |
|---------------------------|--------------|------------------|

| LSTM + Word2Vec           | ~88%         | ~87%             |

- Confusion Matrix and classification reports available in project notebooks.

---

## 📊 Visualizations

- Word Clouds for **Positive** and **Negative** reviews
- Bar plots of **rating distribution**
- Confusion Matrix for model evaluation

---

## 🌟 Key Learnings

- Text data preprocessing for NLP tasks
- Vectorizing text using **TF-IDF** and **Word2Vec**
- Building and evaluating classification models
- Understanding model evaluation metrics in NLP
- Introduction to deep learning models like LSTM and BERT

---

## 🚀 Future Enhancements

- Fine-tune **BERT** for improved accuracy
- Deploy a **web app** (using Streamlit or Flask) to predict sentiment
- Perform **aspect-based sentiment analysis** (e.g., content, delivery, price)
- Explore **multilingual sentiment analysis**

---


