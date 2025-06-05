
from groq import Groq
from neo4j import GraphDatabase
import json
import os
import csv
from keybert import KeyBERT
client = Groq(api_key=GROQ_API_KEY)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

RESULTS_PATH = ""
PROGRESS_PATH = ""
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
    per_keyword_limit = 5 if len(keywords) > 7 else 7

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
def ask_llama3_text_answer(question, context_docs, options_list):
    options_block = "\n".join(f"- {opt}" for opt in options_list)
    prompt = f"""You are a helpful medical assistant.

The context below contains relevant concepts and their closest relationships.
Based only on the CONTEXT and OPTIONS, select the best answer to the QUESTION.
Your answer must be **exactly one of the options below**. Do not include letters like 'A.', 'B.' etc. Just return the full answer text. Do not explain.

CONTEXT:
{chr(10).join(context_docs)}

QUESTION:
{question}

OPTIONS:
{options_block}

ANSWER:
"""
    print(f"\n📨 Prompt sent to LLaMA 3:\n{'='*40}\n{prompt}\n{'='*40}")
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content.strip()

def rag_pipeline_medmcqa(sample):
    if sample.get("choice_type") != "single":
        return None, None, sample.get("id")

    question = sample["question"]
    options_dict = {
        "A": sample.get("opa", ""),
        "B": sample.get("opb", ""),
        "C": sample.get("opc", ""),
        "D": sample.get("opd", "")
    }
    idx_to_key = {1: "opa", 2: "opb", 3: "opc", 4: "opd"}
    correct_idx = sample.get("cop", None)
    answer_gt = sample.get(idx_to_key.get(correct_idx, ""), "")

    print(f"\n Question: {question}")
    print(f" Options: {list(options_dict.values())}")
    print(f" Ground Truth Option: {answer_gt}")

    question_keywords = extract_keywords(question)
    print(" Keywords:", question_keywords)
    
    options_keywords = []
    for opt in options_dict.values():
        options_keywords.extend(extract_keywords(opt))

    context = query_neo4j_with_links(question_keywords)
    if not context:
        print(" No context found in graph. Passing fallback note to model.")
        context = ["(no relevant knowledge was found in the graph — use your own medical knowledge to answer)"]

    predicted = ask_llama3_text_answer(question, context, list(options_dict.values()))
    return predicted, answer_gt, sample.get("id")

def save_progress(results):
    with open(RESULTS_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted", "gold"])
        for row in results:
            writer.writerow(row)

def load_progress():
    if not os.path.exists(RESULTS_PATH):
        return set(), []
    seen_ids = set()
    rows = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen_ids.add(row["id"])
            rows.append((row["id"], row["predicted"], row["gold"]))
    return seen_ids, rows

if __name__ == "__main__":
    y_true = []
    y_pred = []
    results = []

    seen_ids, previous_rows = load_progress()
    results.extend([(r[0], r[1], r[2]) for r in previous_rows])

    with open("", "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            buffer += line.strip()
            if buffer.endswith("}"):
                try:
                    sample = json.loads(buffer)
                    buffer = ""
                    if sample.get("id") in seen_ids:
                        continue

                    pred, gold, sid = rag_pipeline_medmcqa(sample)
                    if pred is None:
                        continue
                    print(f"\n Example → Predicted: {pred}")
                    print(f" Ground Truth: {gold}\n")

                    y_true.append(gold.lower().strip())
                    y_pred.append(pred.lower().strip())
                    results.append((sid, pred, gold))
                    seen_ids.add(sid)
                    save_progress(results)
                except Exception as e:
                    print(f" Error: {e}")
                    continue

    total = len(y_true)
    correct = sum([1 for p, g in zip(y_pred, y_true) if p == g])
    accuracy = (correct / total) * 100 if total > 0 else 0