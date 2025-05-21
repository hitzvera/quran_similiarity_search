from flask import Flask, request, render_template
import whisper
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import re



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
whisper_model = whisper.load_model("small")
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")



# Load Quran data
with open('quran_data.json', 'r', encoding='utf-8') as f:
    quran_ayahs = json.load(f)

def remove_diacritics(text):
    import re
    return re.sub(r'[ًٌٍَُِّْٰ]', '', text)

def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / norm



# Normalize and index dataset embeddings
texts = [remove_diacritics(ayah['text']) for ayah in quran_ayahs]
raw_embeddings = sbert_model.encode(texts)
normalized_embeddings = np.array([safe_normalize(vec) for vec in raw_embeddings])

dim = normalized_embeddings[0].shape[0]
index = faiss.IndexFlatIP(dim)
index.add(normalized_embeddings)


# Normalize function
def normalize_arabic(text):
    return re.sub(r'[ًٌٍَُِّْٰ]', '', text)

def label_score(score):
    if score > 0.90:
        return "🔵 Exact Match"
    elif score > 0.80:
        return "🟢 High Match"
    elif score > 0.70:
        return "🟡 Medium Match"
    else:
        return "🔴 Low Match"

# Highlight match function
def highlight_multiple(transcription, ayah_text):
    norm_trans = normalize_arabic(transcription)
    norm_ayah = normalize_arabic(ayah_text)
    tokens = norm_trans.split()

    norm_to_orig = {}
    for match in re.finditer(r'\S+', ayah_text):
        norm_token = normalize_arabic(match.group())
        norm_to_orig.setdefault(norm_token, []).append((match.start(), match.end()))

    highlights = []
    for token in tokens:
        for key in norm_to_orig:
            if token == key or token in key:
                highlights.extend(norm_to_orig[key])

    highlights = sorted(set(highlights), key=lambda x: x[0])
    
    result = ""
    last = 0
    for start, end in highlights:
        result += ayah_text[last:start] + "<mark>" + ayah_text[start:end] + "</mark>"
        last = end
    result += ayah_text[last:]
    return result



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        audio_file = request.files['audio']
        if audio_file:
            file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            audio_file.save(file_path)
            print("Saved audio to:", file_path)

            result = whisper_model.transcribe(file_path, language='ar')
            transcription = result['text'].strip()

            print(transcription, 'kucing')

            query_embedding = sbert_model.encode(remove_diacritics(transcription))
            query_embedding = safe_normalize(query_embedding)

            D, I = index.search(np.array([query_embedding]), k=5)

            results = []
            for idx, score in zip(I[0], D[0]):
                ayah = quran_ayahs[idx]
                highlighted = highlight_multiple(transcription, ayah['text'])
                score = round(score, 2)

                results.append({
                    'surah': ayah['surah'],
                    'ayah': ayah['ayah'],
                    'text': highlighted,
                    'score': label_score(score) # e.g., 0.88
                })

            return render_template('index.html', transcription=transcription, results=results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
