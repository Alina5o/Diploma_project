
import os, csv, json, numpy as np
from neo4j import GraphDatabase
from groq   import Groq
from sbert.pca import encode_sbert_qa
NEO4J_URI     = "bolt://localhost:7687"
NEO4J_USER    = "neo4j"
NEO4J_PASS    = "12345678"
VECTOR_INDEXES = ["idx_concept_embedding", "idx_atom_embedding"]  
EMBED_DIM     = 300
TOP_K         = 10
JSONL_PATH    = ""
RESULTS_CSV   = ""
GROQ_KEY      = os.getenv("GROQ_API_KEY") 

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

groq    = Groq(api_key=GROQ_KEY)

def embed_query(text: str) -> list[float]:
    return encode_sbert_qa(text)
def knn_search(vec: list[float], k=TOP_K):
    
    if np.linalg.norm(vec) == 0:
        print(" Skipping vector search due to empty embedding.")
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

def ask_llama(question, options, context):
    options_block = "\n".join(f"- {o}" for o in options)
    prompt = f"""You are a helpful medical assistant.

The context below contains relevant concepts and their closest relationships.
Based only on the OPTIONS, select the best answer to the QUESTION.
Your answer must be exactly one of the options below. Just return the full answer text. No explanations. No marks, symbols, etc.

QUESTION:
{question}

OPTIONS:
{options_block}

CONTEXT:
{context}

ANSWER:
"""
    res = groq.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

def save_results_row(row, write_header=False):
    mode = "a" if os.path.exists(RESULTS_CSV) else "w"
    with open(RESULTS_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header and mode == "w":
            writer.writerow(["id", "predicted", "gold"])
        writer.writerow(row)
def load_seen():
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}

def main():
    seen_ids = load_seen()
    y_true, y_pred = [], []

    print("work")
    first_row = not os.path.exists(RESULTS_CSV)

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                sid = sample["id"]
                if  sid in seen_ids:
                    continue

                question = sample["question"]
                options  = [sample.get(k, "") for k in ("opa", "opb", "opc", "opd")]
                gold     = sample.get(f"op{chr(96 + sample.get('cop', 0))}", "")

                q_vec    = embed_query(question)
                nn       = knn_search(q_vec, TOP_K)
                context  = ", ".join(r["name"] for r in nn) if nn else "None"
                pred     = ask_llama(question, options, context)

                print(f" {sid} → {pred} | GT: {gold}")
                save_results_row([sid, pred, gold], write_header=first_row)
                first_row = False

                y_true.append(gold.lower().strip())
                y_pred.append(pred.lower().strip())

            except Exception as e:
                print(f" Error processing {sample.get('id', '???')}: {e}")

    if y_true:
        acc = 100 * sum(p == g for p, g in zip(y_pred, y_true)) / len(y_true)
        print(f"\n Accuracy: {acc:.2f}%  ({sum(p == g for p, g in zip(y_pred, y_true))}/{len(y_true)})")


if __name__ == "__main__":
    main()

