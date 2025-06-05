import os
import csv
import numpy as np
from neo4j import GraphDatabase
from groq import Groq
from sbert.encode_pca import encode_sbert_300d
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASS     = "12345678"
VECTOR_INDEXES = ["idx_concept_embedding", "idx_atom_embedding"]
EMBED_DIM      = 300
TOP_K          = 10
INPUT_CSV      = ""
RESULTS_CSV    = ""
GROQ_KEY      = os.getenv("GROQ_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
groq = Groq(api_key=GROQ_KEY)

def embed_query(text: str) -> list[float]:
    return encode_sbert_300d(text)

def knn_search(vec: list[float], k=TOP_K):
    if np.linalg.norm(vec) == 0:
        print(" Empty embedding vector")
        return []
    cypher = """
    WITH $v AS q
    CALL db.index.vector.queryNodes($index, $k, q)
    YIELD node, score
    RETURN node.name AS name, score
    """
    candidates = []
    with driver.session() as s:
        for idx in VECTOR_INDEXES:
            rows = s.run(cypher, v=vec, k=k, index=idx).data()
            candidates.extend(rows)
    candidates.sort(key=lambda r: r["score"])
    return candidates[:k]

def build_context(candidates) -> str:
    return "\n".join(f"- {c['name']}" for c in candidates)

def ask_llama_open(question, context):
    prompt = f"""You are a medical assistant.

# Task
You will be given a medical QUESTION along with a CONTEXT that contains knowledge extracted from a medical knowledge graph.

# Rules
- When answering, read the CONTEXT first. If it fully answers the QUESTION, rely only on it.
- If the CONTEXT is insufficient, answer from your own medical knowledge.
- Don't mention about insufficiency of context, just answer.
- Respond in 3-4 concise sentences, strictly focused on the QUESTION.
- Output only the answer text—no preamble, no extra words.
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:"""
    res = groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

if __name__ == "__main__":
    seen_questions = set()

    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            seen_questions.update(row["question"].strip() for row in csv.DictReader(f))

    with open(INPUT_CSV, "r", encoding="utf-8") as infile, open(RESULTS_CSV, "a", newline='', encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        if os.stat(RESULTS_CSV).st_size == 0:
            writer.writerow(["question", "predicted", "ground_truth"])

        for row in reader:
            try:
                question = row["question"].strip()
                if question in seen_questions:
                    continue

                vec = embed_query(question)
                context = build_context(knn_search(vec))
                predicted = ask_llama_open(question, context)
                ground_truth = row["answer"].strip()

                writer.writerow([question, predicted, ground_truth])
                seen_questions.add(question)

                print(f" Saved: {question[:80]}...\n")

            except Exception as e:
                print(f" Error: {e}")

