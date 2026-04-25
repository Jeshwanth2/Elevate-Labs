from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ranker import ResumeRanker

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ranker = ResumeRanker()

@app.route('/api/rank', methods=['POST'])
def rank_resumes():
    """Rank uploaded resumes against a job description."""
    try:
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400

        files = request.files.getlist('resumes')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'At least one resume is required'}), 400

        resume_texts = []
        resume_names = []

        for file in files:
            if file and file.filename:
                text = ranker.extract_text(file)
                if text:
                    resume_texts.append(text)
                    resume_names.append(file.filename)

        if not resume_texts:
            return jsonify({'error': 'Could not extract text from any resume'}), 400

        results = ranker.rank_resumes(job_description, resume_texts, resume_names)
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def serve_index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Resume Ranker API is running'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)