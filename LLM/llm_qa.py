from groq import Groq
import json
import os
import csv

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

RESULTS_PATH = ""
INPUT_PATH = ""

def ask_llama3_text_answer(question, options_list):
    options_block = "\n".join(f"- {opt}" for opt in options_list)
    prompt = f"""You are a helpful medical assistant.

The context below contains relevant concepts and their closest relationships.
Based only on the OPTIONS, select the best answer to the QUESTION.
Your answer must be **exactly one of the options below**. Do not include letters like 'A.', 'B.' etc. Just return the full answer text. Do not explain.

QUESTION:
{question}

OPTIONS:
{options_block}

ANSWER:
"""
    print(f"\nPrompt sent to LLaMA 3:\n{'='*40}\n{prompt}\n{'='*40}")
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content.strip()

def rag_pipeline(sample):
    question = sample["question"]
    options_dict = {
        "A": sample.get("opa", ""),
        "B": sample.get("opb", ""),
        "C": sample.get("opc", ""),
        "D": sample.get("opd", "")
    }
    idx_to_key = {1: "opa", 2: "opb", 3: "opc", 4: "opd"}
    correct_idx = sample.get("cop", None)
    gold = sample.get(idx_to_key.get(correct_idx, ""), "")
    predicted = ask_llama3_text_answer(question, list(options_dict.values()))
    
    return {
        "id": sample.get("id"),
        "question": question,
        "predicted": predicted,
        "gold": gold,
        "is_correct": int(predicted.lower().strip() == gold.lower().strip()),
        "subject": sample.get("subject_name", ""),
        "topic": sample.get("topic_name", ""),
        "options": "; ".join(options_dict.values())
    }

def save_results(rows):
    with open(RESULTS_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def load_seen_ids():
    if not os.path.exists(RESULTS_PATH):
        return set()
    with open(RESULTS_PATH, newline='', encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}

if __name__ == "__main__":
    seen_ids = load_seen_ids()
    results = []
    y_true, y_pred = [], []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            buffer += line.strip()
            if buffer.endswith("}"):
                try:
                    sample = json.loads(buffer)
                    buffer = ""
                    if sample.get("id") in seen_ids:
                        continue

                    row = rag_pipeline(sample)
                    print(f"\nPredicted: {row['predicted']}")
                    print(f"Gold: {row['gold']}\n")

                    y_pred.append(row["predicted"].lower().strip())
                    y_true.append(row["gold"].lower().strip())
                    results.append(row)
                    save_results(results)
                    seen_ids.add(row["id"])
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    total = len(y_true)
    correct = sum(1 for p, g in zip(y_pred, y_true) if p == g)
    accuracy = (correct / total) * 100 if total else 0
    print(accuracy)