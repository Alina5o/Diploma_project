
import os
import csv
from groq import Groq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


INPUT_CSV = ""
RESULTS_CSV = ""



def ask_llama3_open_answer(question: str):
    prompt = f"""You are a medical assistant. Answer the following medical question clearly, accurately, and briefly.

Respond in no more than 2–3 sentences. Avoid unnecessary details or examples. Do not list items. If you are unsure, say you don't know.


QUESTION:
{question}

ANSWER:"""
    print(f"\n Prompt to LLaMA 3:\n{'='*40}\n{prompt}\n{'='*40}")
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content.strip()

if __name__ == "__main__":
    seen_questions = set()

    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            for row in reader:
                seen_questions.add(row["question"].strip())

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

                ground_truth = row["answer"].strip()
                predicted = ask_llama3_open_answer(question)

                writer.writerow([question, predicted, ground_truth])
                seen_questions.add(question)

                print(f" Saved result for: {question[:80]}...\n")

            except Exception as e:
                print(f" Error processing row: {e}")
