
import json
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

JSONL_PATH = ""  
EMBED_DIM = 300
SENTENCE_MODEL = "all-MiniLM-L6-v2"

def load_questions_from_jsonl(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question", "").strip()
            if q:
                questions.append(q)
    return questions

questions = load_questions_from_jsonl(JSONL_PATH)

model = SentenceTransformer(SENTENCE_MODEL)

embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)


pca = PCA(n_components=EMBED_DIM)
pca.fit(embeddings)


with open("pca_sbert_qa.pkl", "wb") as f:
    pickle.dump(pca, f)

def embed_query(text: str) -> list[float]:
    vec = model.encode([text], convert_to_numpy=True)
    return pca.transform(vec)[0].tolist()

example_question = questions[0]
example_embedding = embed_query(example_question)
