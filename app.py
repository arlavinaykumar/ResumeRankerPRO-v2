from flask import Flask, render_template, request, send_file
import os
import PyPDF2
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

app = Flask(__name__)

# ✅ Load the SpaCy model
print("Loading SpaCy model...")
nlp = spacy.load('en_core_web_sm')
print("SpaCy model loaded ✅")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ✅ Text Extraction Function
def extract_text_from_pdf(pdf_path):
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print("❌ Error extracting text:", e)
    return text

# ✅ Preprocessing Function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# ✅ Ranking Logic
def rank_resumes(resume_texts, job_desc):
    print("Ranking resumes now...")
    all_texts = resume_texts + [job_desc]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
    scores = similarity.flatten()
    return scores

@app.route('/')
def home():
    print("Rendering home page...")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        print("Received upload request...")
        job_desc = request.form['job_desc']
        uploaded_files = request.files.getlist('resumes')
        resume_texts = []
        file_names = []

        for file in uploaded_files:
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            text = extract_text_from_pdf(filepath)
            preprocessed = preprocess_text(text)
            resume_texts.append(preprocessed)
            file_names.append(file.filename)

        job_desc_processed = preprocess_text(job_desc)
        scores = rank_resumes(resume_texts, job_desc_processed)

        df = pd.DataFrame({
            'Resume': file_names,
            'Score': scores
        }).sort_values(by='Score', ascending=False)

        result_path = os.path.join(UPLOAD_FOLDER, 'ranked_resumes.xlsx')
        df.to_excel(result_path, index=False)

        print("Ranking complete ✅")
        return render_template('result.html', tables=[df.to_html(classes='data')], download_link='ranked_resumes.xlsx')

    except Exception as e:
        print("❌ Upload error:", str(e))
        return "Error occurred during upload!"

@app.route('/download')
def download():
    print("Download requested...")
    path = os.path.join(UPLOAD_FOLDER, 'ranked_resumes.xlsx')
    return send_file(path, as_attachment=True)

# ✅ Run the app
if __name__ == '__main__':
    print("🚀 Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True)




