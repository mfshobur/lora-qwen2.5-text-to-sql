"""
Evaluate the base and LoRA fine-tuned Qwen2.5-0.5B models on text-to-SQL.

Two metrics per model:
  - execution accuracy: the predicted SQL runs without error
  - execution match:    the predicted SQL returns the same result set as the
                        gold SQL

Model responses are passed through extract_sql() first, since the base model
tends to wrap SQL in markdown or prose. Results are written to
eval_results.json for the report and examples notebook to use.
"""
import os
# silence HF progress bars / advisory warnings before importing transformers
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import warnings
import torch
import psycopg2
from dotenv import load_dotenv
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

load_dotenv()

MODEL_PATH     = "./models/qwen2.5-0.5b-instruct"
LORA_PATH      = "./lora_output"
TEST_DATA_PATH = "dataset_output/test.json"
OUT_PATH       = "eval_results.json"

with open("schema_prompt.txt") as f:
    SCHEMA = f.read()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}


# SQL extraction
def extract_sql(text: str) -> str:
    """Pull a bare SQL statement out of a model response that may contain
    markdown fences or prose."""
    if text is None:
        return ""
    t = text.strip()

    # ```sql ... ```  or  ``` ... ```
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()

    # take from the first SELECT / WITH keyword onward
    m = re.search(r"\b(SELECT|WITH)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start():]

    # cut at a trailing fence if one survived
    t = t.split("```")[0].strip()

    # keep only the first statement
    if ";" in t:
        t = t.split(";")[0].strip() + ";"
    return t.strip()


# database helpers
def make_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 5000;")
    conn.commit()
    cur.close()
    return conn


def run_query(sql: str, conn):
    """Return (ok, rows). rows is a sorted list of stringified tuples or None."""
    if not sql:
        return False, None
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall() if cur.description else []
        conn.rollback()
        cur.close()
        norm = sorted(tuple(str(c) for c in r) for r in rows)
        return True, norm
    except Exception:
        conn.rollback()
        return False, None


# model loading and generation
def load_model(lora=False):
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH if lora else MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, device_map="cpu",
    )
    if lora:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_sql(model, tokenizer, question):
    messages = [
        {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query."},
        {"role": "user",   "content": f"Schema:{SCHEMA}\nQuestion: {question}"},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = tokenized if isinstance(tokenized, torch.Tensor) else tokenized["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=160, do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# evaluation loop
def evaluate(model, tokenizer, samples, conn, label):
    results = []
    executes = exec_match = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        raw  = generate_sql(model, tokenizer, s["natural_language"])
        pred = extract_sql(raw)

        ok, pred_rows = run_query(pred, conn)
        gold_ok, gold_rows = run_query(s["sql_query"], conn)

        match = bool(ok and gold_ok and pred_rows == gold_rows)
        if ok:    executes += 1
        if match: exec_match += 1

        results.append({
            "question":   s["natural_language"],
            "complexity": s.get("complexity", "unknown"),
            "gold_sql":   s["sql_query"],
            "raw_output": raw,
            "pred_sql":   pred,
            "executes":   ok,
            "exec_match": match,
            "gold_ok":    gold_ok,
        })
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{label}] {i+1}/{len(samples)}  "
                  f"exec={executes/(i+1):.1%} match={exec_match/(i+1):.1%}  "
                  f"({el:.0f}s)", flush=True)

    n = len(samples)
    return {
        "exec_accuracy": executes / n,
        "exec_match":    exec_match / n,
        "n":             n,
        "results":       results,
    }


def main():
    with open(TEST_DATA_PATH) as f:
        samples = json.load(f)
    print(f"Evaluating on {len(samples)} test samples", flush=True)

    conn = make_conn()

    print("\nBase model", flush=True)
    base_model, base_tok = load_model(lora=False)
    base = evaluate(base_model, base_tok, samples, conn, "base")
    print(f"Base: exec={base['exec_accuracy']:.1%}  match={base['exec_match']:.1%}", flush=True)
    del base_model

    print("\nFine-tuned (LoRA) model", flush=True)
    ft_model, ft_tok = load_model(lora=True)
    ft = evaluate(ft_model, ft_tok, samples, conn, "ft")
    print(f"FT: exec={ft['exec_accuracy']:.1%}  match={ft['exec_match']:.1%}", flush=True)
    del ft_model

    conn.close()

    with open(OUT_PATH, "w") as f:
        json.dump({"base": base, "fine_tuned": ft}, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
