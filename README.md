# 🧠 Quora Question Pairs - Duplicate Detection (BERT + BoW)

This project explores two different approaches to identify whether two questions on Quora are duplicates:
1. **Deep Learning-based** using BERT (transformers)
2. **Traditional NLP** using Bag-of-Words with Logistic Regression

---

## 📌 Project Overview

- **Dataset**: [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/data)
- **Task**: Binary classification — Are two questions semantically identical?

---

## 🔍 Approach 1: BERT + Hugging Face Transformers

### ✅ Features
- `bert-base-uncased` pretrained model
- Fine-tuning on ~1k sample pairs (for fast demo)
- Custom training pipeline with Hugging Face's `Trainer`
- Evaluation using **Accuracy** and **F1 Score**
- Interactive UI with **Gradio**

### 🚀 How to Run (Colab Recommended)
1. Upload `train.csv` to Colab
2. Install required libraries:
   ```bash
   pip install transformers datasets gradio scikit-learn
Run the notebook cells

Use the Gradio app to test live question pairs

🧪 Sample Output

Input:
  Q1: What is the step by step guide to invest in share market in india?
  Q2: What is the step by step guide to invest in share market?

Output:
  Duplicate (Confidence: 0.92)
  
📊 Metrics (on sample)
Accuracy: ~88–90%



🔍 Approach 2: Bag-of-Words 

✅ Features
Preprocessing: lowercasing, stopword removal, punctuation

CountVectorizer or TF-IDF vectorization

Handcrafted features: token overlap, length difference

Lightweight Logistic Regression model

Fast training and interpretability

🚀 How to Run
Install requirements:

pip install pandas scikit-learn nltk
Run the notebook bow-with-preprocessing-and-advanced-features.ipynb

📊 Metrics
Accuracy: ~82–85%

🧠 Comparison

| Feature           | BERT                        | BoW  |
| ----------------- | --------------------------- | ------------------------- |
| Type              | Deep Learning (Transformer) | Traditional ML            |
| Accuracy (sample) | \~88–90%                    | \~82–85%                  |
| Speed             | Slower                      | Very fast                 |


