import re
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Try importing PDF/DOCX parsers
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


# ─── Keyword Taxonomy ──────────────────────────────────────────────────────────

SKILL_KEYWORDS = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
        'go', 'rust', 'swift', 'kotlin', 'php', 'scala', 'r', 'matlab',
    ],
    'web': [
        'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'fastapi',
        'html', 'css', 'rest', 'graphql', 'api', 'frontend', 'backend',
    ],
    'data': [
        'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'sql', 'spark', 'hadoop',
        'data analysis', 'statistics', 'visualization', 'tableau', 'power bi',
    ],
    'cloud': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'ci/cd', 'devops', 'linux', 'git', 'microservices',
    ],
    'soft_skills': [
        'leadership', 'communication', 'teamwork', 'agile', 'scrum',
        'problem solving', 'project management', 'collaboration',
    ],
}

EDUCATION_WEIGHTS = {
    'phd': 1.0, 'doctorate': 1.0,
    'master': 0.85, 'mba': 0.85, 'ms': 0.85,
    'bachelor': 0.7, 'bs': 0.7, 'be': 0.7, 'btech': 0.7,
    'associate': 0.5, 'diploma': 0.4, 'certification': 0.3,
}

EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
    r'experience\s+(?:of\s+)?(\d+)\+?\s*years?',
    r'(\d+)\+?\s*(?:yrs?|yr)\s+(?:of\s+)?(?:experience|exp)',
]


class ResumeRanker:
    """
    ML-powered resume ranker using:
    - TF-IDF vectorization + cosine similarity  (content match)
    - Keyword density scoring                   (skill coverage)
    - Education level detection                 (qualification boost)
    - Experience year extraction                (seniority signal)
    - Composite weighted scoring                (final rank)
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
        )

    # ─── Text Extraction ───────────────────────────────────────────────────────

    def extract_text(self, file_obj) -> str:
        """Extract text from PDF, DOCX, or plain TXT file objects."""
        filename = file_obj.filename.lower()
        content = file_obj.read()

        if filename.endswith('.pdf') and PDF_SUPPORT:
            return self._extract_pdf(io.BytesIO(content))
        elif filename.endswith('.docx') and DOCX_SUPPORT:
            return self._extract_docx(io.BytesIO(content))
        else:
            # Fallback: treat as plain text
            try:
                return content.decode('utf-8', errors='ignore')
            except Exception:
                return ''

    def _extract_pdf(self, buf: io.BytesIO) -> str:
        try:
            reader = PyPDF2.PdfReader(buf)
            return '\n'.join(
                page.extract_text() or '' for page in reader.pages
            )
        except Exception:
            return ''

    def _extract_docx(self, buf: io.BytesIO) -> str:
        try:
            doc = Document(buf)
            return '\n'.join(p.text for p in doc.paragraphs)
        except Exception:
            return ''

    # ─── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s+#./]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]
        return ' '.join(tokens)

    # ─── Feature Scorers ───────────────────────────────────────────────────────

    def tfidf_similarity(self, job_desc: str, resumes: list[str]) -> np.ndarray:
        """Cosine similarity between job description and each resume."""
        docs = [job_desc] + resumes
        preprocessed = [self.preprocess(d) for d in docs]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(preprocessed)
        except ValueError:
            return np.zeros(len(resumes))
        jd_vec = tfidf_matrix[0]
        res_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(jd_vec, res_vecs).flatten()
        return sims

    def keyword_score(self, text: str, job_desc: str) -> float:
        """Fraction of job-description keywords present in the resume."""
        jd_lower = job_desc.lower()
        resume_lower = text.lower()

        # Collect all known skill keywords mentioned in the JD
        jd_skills = []
        for category, keywords in SKILL_KEYWORDS.items():
            for kw in keywords:
                if kw in jd_lower:
                    jd_skills.append(kw)

        # Also tokenize raw JD words as additional signals
        jd_tokens = set(re.findall(r'\b\w{4,}\b', jd_lower))

        if not jd_skills and not jd_tokens:
            return 0.5

        skill_hits = sum(1 for kw in jd_skills if kw in resume_lower)
        skill_score = skill_hits / max(len(jd_skills), 1)

        token_hits = sum(1 for t in jd_tokens if t in resume_lower)
        token_score = min(token_hits / max(len(jd_tokens), 1), 1.0)

        return 0.6 * skill_score + 0.4 * token_score

    def education_score(self, text: str) -> float:
        """Detect highest education level in the resume."""
        lower = text.lower()
        best = 0.0
        for keyword, weight in EDUCATION_WEIGHTS.items():
            if keyword in lower:
                best = max(best, weight)
        return best if best > 0 else 0.3   # default if undetected

    def experience_score(self, text: str) -> float:
        """Extract years of experience and normalise to [0, 1]."""
        years = []
        for pattern in EXPERIENCE_PATTERNS:
            for match in re.finditer(pattern, text.lower()):
                try:
                    years.append(int(match.group(1)))
                except ValueError:
                    pass
        if not years:
            return 0.3
        max_years = max(years)
        # Clamp: 15+ years → 1.0
        return min(max_years / 15.0, 1.0)

    def section_completeness(self, text: str) -> float:
        """Reward resumes that have standard sections."""
        sections = [
            'experience', 'education', 'skills', 'projects',
            'summary', 'objective', 'certifications', 'achievements',
        ]
        lower = text.lower()
        hits = sum(1 for s in sections if s in lower)
        return hits / len(sections)

    # ─── Main Ranking ──────────────────────────────────────────────────────────

    def rank_resumes(
        self,
        job_description: str,
        resume_texts: list[str],
        resume_names: list[str],
    ) -> list[dict]:
        """
        Compute a composite score for each resume and return ranked results.

        Weights:
          TF-IDF similarity   40 %
          Keyword coverage    30 %
          Experience level    15 %
          Education level     10 %
          Section completeness 5 %
        """
        n = len(resume_texts)

        # Raw feature arrays
        tfidf_scores   = self.tfidf_similarity(job_description, resume_texts)
        keyword_scores = np.array([self.keyword_score(t, job_description) for t in resume_texts])
        exp_scores     = np.array([self.experience_score(t)      for t in resume_texts])
        edu_scores     = np.array([self.education_score(t)       for t in resume_texts])
        sec_scores     = np.array([self.section_completeness(t)  for t in resume_texts])

        # Normalise each feature to [0, 1]
        def safe_normalise(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.ones(len(arr)) * 0.5
            return (arr - mn) / (mx - mn)

        tfidf_n   = safe_normalise(tfidf_scores)
        keyword_n = safe_normalise(keyword_scores)
        exp_n     = safe_normalise(exp_scores)
        edu_n     = safe_normalise(edu_scores)
        sec_n     = safe_normalise(sec_scores)

        # Weighted composite
        composite = (
            0.40 * tfidf_n +
            0.30 * keyword_n +
            0.15 * exp_n +
            0.10 * edu_n +
            0.05 * sec_n
        )

        results = []
        for i in range(n):
            matched_skills = self._matched_skills(resume_texts[i], job_description)
            missing_skills = self._missing_skills(resume_texts[i], job_description)

            results.append({
                'name':              resume_names[i],
                'score':             round(float(composite[i]) * 100, 1),
                'tfidf_score':       round(float(tfidf_scores[i])   * 100, 1),
                'keyword_score':     round(float(keyword_scores[i]) * 100, 1),
                'experience_score':  round(float(exp_scores[i])     * 100, 1),
                'education_score':   round(float(edu_scores[i])     * 100, 1),
                'section_score':     round(float(sec_scores[i])     * 100, 1),
                'matched_skills':    matched_skills,
                'missing_skills':    missing_skills,
                'word_count':        len(resume_texts[i].split()),
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        # Attach rank and grade
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
            r['grade'] = self._grade(r['score'])

        return results

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _matched_skills(self, text: str, job_desc: str) -> list[str]:
        jd_lower = job_desc.lower()
        res_lower = text.lower()
        matched = []
        for keywords in SKILL_KEYWORDS.values():
            for kw in keywords:
                if kw in jd_lower and kw in res_lower:
                    matched.append(kw)
        return matched[:12]

    def _missing_skills(self, text: str, job_desc: str) -> list[str]:
        jd_lower = job_desc.lower()
        res_lower = text.lower()
        missing = []
        for keywords in SKILL_KEYWORDS.values():
            for kw in keywords:
                if kw in jd_lower and kw not in res_lower:
                    missing.append(kw)
        return missing[:8]

    def _grade(self, score: float) -> str:
        if score >= 80: return 'Excellent'
        if score >= 65: return 'Good'
        if score >= 50: return 'Average'
        if score >= 35: return 'Below Average'
        return 'Poor'