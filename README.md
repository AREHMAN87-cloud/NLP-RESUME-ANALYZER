# 📄 Resume Screening App using NLP

This is a simple web app built with **Flask** and **Natural Language Processing (NLP)** to classify resumes into different categories (e.g., Data Scientist, Web Developer, etc.) based on their content.

## 🚀 Features
- Upload **PDF** or **TXT** resumes.
- Automatically extracts and processes text.
- Classifies the resume using a pre-trained ML model (SVM + TF-IDF).
- Simple and clean HTML interface with CSS styling.

## 🧠 How It Works
The model uses:
- `TF-IDF` vectorizer for converting text to numerical data.
- `SVM` classifier for prediction.
- `LabelEncoder` for converting category labels.

All trained components are saved using `pickle`:
- `tfidf.pkl` (text vectorizer)
- `clf.pkl` (classifier)
- `encoder.pkl` (label encoder)

These files are saved **locally** in the same directory where your training notebook or script is executed.

## 📂 File Structure
```
Resume screening APP/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── tfidf.pkl
├── clf.pkl
├── encoder.pkl
```

> ⚠️ Make sure all `.pkl` files are in the **same directory** as `app.py`. If you organize files into folders, update the file paths accordingly in your `app.py`.

## 🛠 How to Run
1. Install requirements:
   ```
   pip install flask PyPDF2 scikit-learn
   ```

2. Run the Flask app:
   ```
   python app.py
   ```

3. Visit `http://127.0.0.1:5000` in your browser.
