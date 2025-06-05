import pickle
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("pca_sbert_qa.pkl", "rb") as f:
    pca = pickle.load(f)

def encode_sbert_qa(text: str) -> list[float]:
    emb = sbert_model.encode([text], convert_to_numpy=True)
    reduced = pca.transform(emb)
    return reduced[0].tolist()