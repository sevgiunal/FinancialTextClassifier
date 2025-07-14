# 📊 Finance Text Classification with GravityAI

This project classifies financial news articles into predefined categories using a trained machine learning model. The application is packaged for deployment on the GravityAI platform.

---

## 🚀 How It Works

The `classify_articles.py` script:
1. Loads pretrained components (`.pkl` files):
   - Text vectorizer (e.g., TF-IDF)
   - Classification model (e.g., Logistic Regression, SVM)
   - Label encoder for human-readable category output
2. Accepts text input via GravityAI-compatible JSON format.
3. Returns a predicted finance-related category for each input article.
