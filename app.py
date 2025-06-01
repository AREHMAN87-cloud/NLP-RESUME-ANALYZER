from flask import Flask, request, render_template
import pickle
import PyPDF2
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Max 2MB upload size

# Load models
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    resume_text = ''
    
    # Check if file uploaded
    if 'resume_file' in request.files:
        file = request.files['resume_file']
        if file and file.filename != '':
            filename = file.filename.lower()
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif filename.endswith('.txt'):
                resume_text = file.read().decode('utf-8')
            else:
                return render_template('index.html', prediction='Unsupported file format! Please upload PDF or TXT.')
    
    # If no file, get text from textarea
    if not resume_text:
        resume_text = request.form.get('resume_text')
        if not resume_text.strip():
            return render_template('index.html', prediction='Please provide resume text or upload a file.')
    
    # Transform and predict
    vector = tfidf.transform([resume_text]).toarray()
    prediction = clf.predict(vector)[0]
    label = le.inverse_transform([prediction])[0]

    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
