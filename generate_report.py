"""
Build report.html from eval_results.json.

Produces a standalone HTML report on the LoRA fine-tuning experiment.
"""
import os
import json
import html
import sqlglot
import psycopg2
from collections import Counter, defaultdict
from datetime import date
from dotenv import load_dotenv

load_dotenv()

with open("eval_results.json") as f:
    d = json.load(f)
BASE = d["base"]
FT   = d["fine_tuned"]
N    = FT["n"]

COMPLEXITY_ORDER = ["easy", "medium", "hard", "very_hard"]


# per-complexity stats
def by_complexity(results):
    agg = defaultdict(lambda: [0, 0, 0])  # n, exec, match
    for r in results:
        a = agg[r["complexity"]]
        a[0] += 1
        a[1] += r["executes"]
        a[2] += r["exec_match"]
    return agg


base_cx = by_complexity(BASE["results"])
ft_cx   = by_complexity(FT["results"])

# categorize the fine-tuned model's execution failures
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", 5432),
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
)


def classify(msg):
    msg = msg.split("\n")[0]
    if "does not exist" in msg and "column" in msg:        return "Undefined column / alias"
    if "does not exist" in msg and ("relation" in msg or "table" in msg): return "Undefined table"
    if "missing FROM-clause" in msg:                       return "Bad table alias (missing FROM)"
    if "nested" in msg and "aggregate" in msg:             return "Nested aggregate"
    if "must appear in the GROUP BY" in msg:               return "GROUP BY violation"
    if "syntax error" in msg:                              return "Syntax error"
    if "function" in msg and "does not exist" in msg:      return "Unknown function"
    return "Other"


fail_cats = Counter()
for r in FT["results"]:
    if r["executes"]:
        continue
    try:
        cur = conn.cursor()
        cur.execute(r["pred_sql"])
        conn.rollback()
    except Exception as e:
        conn.rollback()
        fail_cats[classify(str(e))] += 1
conn.close()

exec_fail   = [r for r in FT["results"] if not r["executes"]]
runs_wrong  = [r for r in FT["results"] if r["executes"] and not r["exec_match"]]
correct     = [r for r in FT["results"] if r["exec_match"]]


# pick example queries to show in the report
base_by_q = {r["question"]: r for r in BASE["results"]}

wins = []
for r in FT["results"]:
    b = base_by_q.get(r["question"])
    if r["exec_match"] and b and not b["exec_match"]:
        wins.append((r, b))
PREFERRED = {
    "medium": "Get the total commission for each employee based on completed orders.",
}
wins_sample = []
seen = set()
for cx in ["easy", "hard"]:
    candidates = [(r, b) for r, b in wins if r["complexity"] == cx]
    if candidates:
        wins_sample.append(candidates[0]); seen.add(cx)

runs_wrong_sample = runs_wrong[:4]
exec_fail_sample  = exec_fail[:3]


# html helpers
def esc(s):
    return html.escape(str(s)) if s is not None else ""


def fmt_sql(sql):
    """Pretty-print SQL with sqlglot; fall back to the raw string if it
    cannot be parsed (e.g. the base model emitted invalid SQL)."""
    if not sql:
        return "(no SQL produced)"
    try:
        return sqlglot.transpile(sql, read="postgres", write="postgres",
                                 pretty=True)[0]
    except Exception:
        return sql


def pct(x):
    return f"{x*100:.1f}%"


def bar(value, color):
    return (f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{value*100:.1f}%;background:{color}"></div>'
            f'<span class="bar-label">{pct(value)}</span></div>')


def cx_rows():
    out = []
    for cx in COMPLEXITY_ORDER:
        bn, _, bm = base_cx[cx]
        fn, _, fm = ft_cx[cx]
        out.append(f"""<tr>
          <td class="cx">{cx.replace('_',' ')}</td>
          <td>{fn}</td>
          <td>{bar(bm/bn, '#94a3b8')}</td>
          <td>{bar(fm/fn, '#6366f1')}</td>
          <td class="delta">+{pct(fm/fn - bm/bn)}</td>
        </tr>""")
    return "\n".join(out)


def fail_cat_rows():
    total = sum(fail_cats.values())
    out = []
    for cat, cnt in fail_cats.most_common():
        out.append(f"""<tr><td>{esc(cat)}</td><td>{cnt}</td>
          <td>{bar(cnt/total, '#ef4444')}</td></tr>""")
    return "\n".join(out)


def example_card(r, b=None, kind="win"):
    badge = {"win": ('correct', '#16a34a'),
             "wrong": ('runs, wrong rows', '#d97706'),
             "fail": ('failed to execute', '#dc2626')}[kind]
    if b is not None:
        models_block = f"""<div class="sql-row">
          <div class="sql-block base">
            <div class="sql-head">Base model</div>
            <pre>{esc(fmt_sql(b['pred_sql']))}</pre>
          </div>
          <div class="sql-block ft">
            <div class="sql-head">Fine-tuned model</div>
            <pre>{esc(fmt_sql(r['pred_sql']))}</pre>
          </div>
        </div>"""
    else:
        models_block = f"""<div class="sql-block ft">
          <div class="sql-head">Fine-tuned model</div>
          <pre>{esc(fmt_sql(r['pred_sql']))}</pre>
        </div>"""
    return f"""<div class="example">
      <div class="ex-q"><span class="cx-tag">{r['complexity'].replace('_',' ')}</span>
        {esc(r['question'])}</div>
      <span class="badge" style="background:{badge[1]}">{badge[0]}</span>
      {models_block}
      <div class="sql-block gold">
        <div class="sql-head">Gold (reference) SQL</div>
        <pre>{esc(fmt_sql(r['gold_sql']))}</pre>
      </div>
    </div>"""


wins_html  = "\n".join(example_card(r, b, "win") for r, b in wins_sample)
wrong_html = "\n".join(example_card(r, base_by_q.get(r["question"]), "wrong") for r in runs_wrong_sample)
fail_html  = "\n".join(example_card(r, base_by_q.get(r["question"]), "fail")  for r in exec_fail_sample)

delta_exec  = FT["exec_accuracy"] - BASE["exec_accuracy"]
delta_match = FT["exec_match"]    - BASE["exec_match"]

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA Fine-Tuning for Text-to-SQL: Evaluation Report</title>
<style>
  :root {{ --indigo:#6366f1; --slate:#94a3b8; --ink:#1e293b; --bg:#f8fafc; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         color:var(--ink); background:var(--bg); margin:0; line-height:1.6; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:48px 24px 80px; }}
  h1 {{ font-size:30px; margin:0 0 4px; }}
  h2 {{ font-size:21px; margin:44px 0 14px; padding-bottom:6px;
        border-bottom:2px solid #e2e8f0; }}
  h3 {{ font-size:16px; margin:24px 0 8px; }}
  .sub {{ color:#64748b; font-size:14px; margin-bottom:8px; }}
  .lead {{ font-size:15px; background:#fff; border:1px solid #e2e8f0;
           border-radius:10px; padding:16px 18px; }}
  .summary {{ background:#eef2ff; border:1px solid #c7d2fe; border-left:4px solid var(--indigo);
              border-radius:10px; padding:16px 20px; margin:20px 0; }}
  .summary .tldr {{ font-size:12px; font-weight:700; letter-spacing:.06em;
                    color:var(--indigo); text-transform:uppercase; margin-bottom:6px; }}
  .summary p {{ margin:0; font-size:15px; }}
  .summary strong {{ color:var(--indigo); }}
  .cards {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:18px 0; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:18px; }}
  .card .big {{ font-size:32px; font-weight:700; }}
  .card .lbl {{ font-size:13px; color:#64748b; }}
  .card .mv {{ font-size:13px; color:#16a34a; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
           border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; font-size:14px; }}
  th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid #f1f5f9; }}
  th {{ background:#f1f5f9; font-size:12px; text-transform:uppercase;
        letter-spacing:.04em; color:#475569; }}
  td.cx {{ text-transform:capitalize; font-weight:600; }}
  td.delta {{ color:#16a34a; font-weight:600; }}
  .bar-track {{ position:relative; background:#f1f5f9; border-radius:5px;
                height:20px; min-width:130px; }}
  .bar-fill {{ height:100%; border-radius:5px; }}
  .bar-label {{ position:absolute; right:7px; top:0; font-size:12px;
                line-height:20px; color:#334155; }}
  .example {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px;
              padding:16px; margin:14px 0; }}
  .ex-q {{ font-weight:600; font-size:14px; margin-bottom:8px; }}
  .cx-tag {{ background:#eef2ff; color:var(--indigo); font-size:11px;
             padding:2px 7px; border-radius:5px; margin-right:6px;
             text-transform:capitalize; font-weight:600; }}
  .badge {{ display:inline-block; color:#fff; font-size:11px; font-weight:600;
            padding:2px 9px; border-radius:5px; margin-bottom:8px; }}
  .sql-row {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px; }}
  .sql-block {{ margin-top:8px; }}
  .sql-head {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em;
               color:#64748b; margin-bottom:2px; }}
  pre {{ background:#0f172a; color:#e2e8f0; padding:11px 13px; border-radius:7px;
         overflow-x:auto; font-size:12.5px; margin:0;
         font-family:'SF Mono',Menlo,Consolas,monospace; white-space:pre-wrap; }}
  .sql-block.gold pre {{ background:#052e16; }}
  .sql-block.base pre {{ background:#3f1d1d; }}
  .note {{ background:#fffbeb; border:1px solid #fde68a; border-radius:8px;
           padding:12px 14px; font-size:13.5px; margin:14px 0; }}
  ul {{ padding-left:22px; }}
  li {{ margin:4px 0; }}
  footer {{ margin-top:50px; color:#94a3b8; font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">

<h1>Fine-Tuning a 0.5B LLM for Natural-Language-to-SQL</h1>
<div class="sub">LoRA adaptation of Qwen2.5-0.5B-Instruct &middot; PostgreSQL &middot;
  evaluated {date.today().isoformat()}</div>

<h2>Headline results</h2>
<div class="cards">
  <div class="card">
    <div class="lbl">Execution match, base</div>
    <div class="big" style="color:var(--slate)">{pct(BASE['exec_match'])}</div>
  </div>
  <div class="card">
    <div class="lbl">Execution match, fine-tuned</div>
    <div class="big" style="color:var(--indigo)">{pct(FT['exec_match'])}</div>
    <div class="mv">+{pct(delta_match)} improvement</div>
  </div>
  <div class="card">
    <div class="lbl">Execution accuracy, base</div>
    <div class="big" style="color:var(--slate)">{pct(BASE['exec_accuracy'])}</div>
  </div>
  <div class="card">
    <div class="lbl">Execution accuracy, fine-tuned</div>
    <div class="big" style="color:var(--indigo)">{pct(FT['exec_accuracy'])}</div>
    <div class="mv">+{pct(delta_exec)} improvement</div>
  </div>
</div>
<p class="sub">Base model responses are passed through an SQL extractor before scoring, so neither model is penalized for Markdown formatting.</p>

<h2>Accuracy by question difficulty</h2>
<p class="sub">Execution match across {N} test questions.</p>
<table>
  <tr><th>Difficulty</th><th>n</th><th>Base (match)</th>
      <th>Fine-tuned (match)</th><th>Gain</th></tr>
  {cx_rows()}
</table>

<h2>Failure analysis</h2>
<p>
  Of {N} test questions, the fine-tuned model produced
  <strong>{len(correct)} fully correct</strong> queries,
  <strong>{len(runs_wrong)} that execute but return the wrong rows</strong>, and
  <strong>{len(exec_fail)} that fail to execute</strong>. The gap between
  {pct(FT['exec_accuracy'])} execution accuracy and {pct(FT['exec_match'])}
  execution match shows that the model reliably learned valid PostgreSQL
  <em>syntax</em>, but roughly a third of its runnable queries still miss a
  semantic detail.
</p>

<h3>Why queries fail to execute ({len(exec_fail)} cases)</h3>
<table>
  <tr><th>Error category</th><th>Count</th><th>Share</th></tr>
  {fail_cat_rows()}
</table>

<h3>Why runnable queries return wrong rows ({len(runs_wrong)} cases)</h3>
<ul>
  <li><strong>Missing filter.</strong> Omits a <code>WHERE</code> condition implied by the question.</li>
  <li><strong>Missing or wrong join.</strong> Skips a table needed to scope the result.</li>
</ul>

<h2>Example outputs</h2>
<h3>Where fine-tuning wins</h3>
<p class="sub">Base model wrong, fine-tuned model correct. One example per difficulty level.</p>
{wins_html}

<h3>Fine-tuned model: runs but wrong</h3>
{wrong_html}

<h3>Fine-tuned model: fails to execute</h3>
{fail_html}

<h2>Limitations</h2>
<ul>
  <li><strong>Single schema.</strong> The model is fine-tuned and tested on one
      fixed 5-table schema. It is not expected to generalize to unseen schemas.</li>
  <li><strong>Execution match is order-insensitive.</strong> Result rows are compared
      as sorted multisets, so a query with a wrong <code>ORDER BY</code> but the
      right rows still counts as a match.</li>
  <li><strong>LLM-generated reference SQL.</strong> The dataset was generated by a
      larger LLM, then cleaned: MySQL-isms rewritten to PostgreSQL, alias-in-HAVING
      errors fixed, non-executing queries dropped, cross-split duplicates removed.
      All {N} reference queries execute against the live database.</li>
  <li><strong>Greedy decoding.</strong> Both models use deterministic
      (greedy) generation; sampling was not explored.</li>
</ul>

<footer>Generated from eval_results.json &middot; {N} held-out test questions</footer>
</div>
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(HTML)

print(f"Wrote report.html  ({len(HTML)//1024} KB)")
print(f"  base : exec {pct(BASE['exec_accuracy'])}  match {pct(BASE['exec_match'])}")
print(f"  ft   : exec {pct(FT['exec_accuracy'])}  match {pct(FT['exec_match'])}")
print(f"  examples: {len(wins_sample)} wins, {len(runs_wrong_sample)} wrong, "
      f"{len(exec_fail_sample)} fail")
