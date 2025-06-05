import os
import csv
from keybert import KeyBERT
from groq import Groq
from neo4j import GraphDatabase

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

INPUT_CSV = ""
RESULTS_CSV = ""


kw_model = KeyBERT(model="all-MiniLM-L6-v2")

def extract_keywords(text: str, top_k=10) -> list[str]:
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=top_k
    )
    return [kw for kw, score in keywords]

def query_neo4j_with_links(keywords):
    all_links = []
    per_keyword_limit = 5 if len(keywords) > 10 else 10

    for kw in keywords:
        cypher_query = f"""
        MATCH (n)
        WHERE toLower(n.name) CONTAINS '{kw}'
        WITH n LIMIT 1
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS main, r.RELA AS rel, m.name AS neighbor
        LIMIT {per_keyword_limit}
        """
        with driver.session() as session:
            result = session.run(cypher_query)
            for record in result:
                main = record.get("main")
                rel = record.get("rel")
                neighbor = record.get("neighbor")
                if main and rel and neighbor:
                    all_links.append(f"{main} → {rel} → {neighbor}")

    return all_links


def ask_llama3_with_context(question: str, context_links: list):
    context_text = "\n".join(context_links) if context_links else "(no relevant knowledge was found in the graph — use your own medical knowledge to answer)"
    prompt = f"""You are a medical assistant.

# Task
You will be given a medical QUESTION along with a CONTEXT that contains knowledge extracted from a medical knowledge graph.

# Rules
- When answering, read the CONTEXT first. If it fully answers the QUESTION, rely only on it.
- If the CONTEXT is insufficient, answer from your own medical knowledge.
- Don't mention about insufficiency of context, just answer.
- Respond in 3-4 concise sentences, strictly focused on the QUESTION.
- Output only the answer text—no preamble, no extra words.
CONTEXT:
{context_text}

QUESTION:
{question}
"""

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content.strip()

if __name__ == "__main__":
    seen_questions = set()

    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_questions.add(row["question"].strip())

    with open(INPUT_CSV, "r", encoding="utf-8") as infile, \
         open(RESULTS_CSV, "a", newline='', encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        if os.stat(RESULTS_CSV).st_size == 0:
            writer.writerow(["question", "predicted", "ground_truth"])

        for row in reader:
            try:
                question = row["question"].strip()
                if question in seen_questions:
                    continue

                ground_truth = row["answer"].strip()
                keywords = extract_keywords(question)
                context_links = query_neo4j_with_links(keywords)
                print(" Keywords:", keywords)
                print(" Context links:", context_links)
                predicted = ask_llama3_with_context(question, context_links)

                writer.writerow([question, predicted, ground_truth])
                seen_questions.add(question)
                print(f" Saved result for: {question[:80]}...\n")

            except Exception as e:
                print(f" Error processing question: {e}")
