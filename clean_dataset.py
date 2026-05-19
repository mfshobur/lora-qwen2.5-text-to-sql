import json
import re
import os
import shutil
import psycopg2
import sqlglot
import sqlglot.expressions as exp
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

SPLITS = {
    "train":      "dataset_output/train.json",
    "validation": "dataset_output/validation.json",
    "test":       "dataset_output/test.json",
}

RAW_DIR = "dataset_output/raw"


def backup_originals():
    """Copy the original split files into dataset_output/raw/ before cleaning,
    so the overwrite is reversible. Skipped if a backup already exists."""
    if os.path.isdir(RAW_DIR):
        print(f"Backup already exists at {RAW_DIR}, keeping it.")
        return
    os.makedirs(RAW_DIR)
    for path in SPLITS.values():
        shutil.copy(path, os.path.join(RAW_DIR, os.path.basename(path)))
    print(f"Backed up original splits to {RAW_DIR}")


# 1a. rewrite MySQL syntax to PostgreSQL

def fix_mysql_syntax(sql: str) -> str:
    # YEAR(x) -> EXTRACT(YEAR FROM x)
    sql = re.sub(r'\bYEAR\s*\(([^)]+)\)', r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)
    # MONTH(x) -> EXTRACT(MONTH FROM x)
    sql = re.sub(r'\bMONTH\s*\(([^)]+)\)', r'EXTRACT(MONTH FROM \1)', sql, flags=re.IGNORECASE)
    # DAY(x) -> EXTRACT(DAY FROM x)
    sql = re.sub(r'\bDAY\s*\(([^)]+)\)', r'EXTRACT(DAY FROM \1)', sql, flags=re.IGNORECASE)
    # CURDATE() -> CURRENT_DATE
    sql = re.sub(r'\bCURDATE\s*\(\s*\)', 'CURRENT_DATE', sql, flags=re.IGNORECASE)
    # DATE_SUB(x, INTERVAL n UNIT) -> x - INTERVAL 'n unit'
    sql = re.sub(
        r'\bDATE_SUB\s*\(([^,]+),\s*INTERVAL\s+(\d+)\s+(\w+)\s*\)',
        lambda m: f"{m.group(1).strip()} - INTERVAL '{m.group(2)} {m.group(3).lower()}'",
        sql, flags=re.IGNORECASE
    )
    # DATE_ADD(x, INTERVAL n UNIT) -> x + INTERVAL 'n unit'
    sql = re.sub(
        r'\bDATE_ADD\s*\(([^,]+),\s*INTERVAL\s+(\d+)\s+(\w+)\s*\)',
        lambda m: f"{m.group(1).strip()} + INTERVAL '{m.group(2)} {m.group(3).lower()}'",
        sql, flags=re.IGNORECASE
    )
    # bare INTERVAL n UNIT (without quotes) -> INTERVAL 'n unit'
    sql = re.sub(
        r'\bINTERVAL\s+(\d+)\s+(DAY|MONTH|YEAR|HOUR|MINUTE|SECOND)S?\b',
        lambda m: f"INTERVAL '{m.group(1)} {m.group(2).lower()}'",
        sql, flags=re.IGNORECASE
    )
    # IFNULL(a, b) -> COALESCE(a, b)
    sql = re.sub(r'\bIFNULL\s*\(', 'COALESCE(', sql, flags=re.IGNORECASE)
    # ISNULL(x) -> x IS NULL
    sql = re.sub(r'\bISNULL\s*\(([^)]+)\)', r'\1 IS NULL', sql, flags=re.IGNORECASE)
    # GROUP_CONCAT(x) -> STRING_AGG(x, ',')
    sql = re.sub(r'\bGROUP_CONCAT\s*\(([^)]+)\)', r"STRING_AGG(\1, ',')", sql, flags=re.IGNORECASE)
    # LIMIT n,m -> LIMIT m OFFSET n
    sql = re.sub(
        r'\bLIMIT\s+(\d+)\s*,\s*(\d+)',
        lambda m: f'LIMIT {m.group(2)} OFFSET {m.group(1)}',
        sql, flags=re.IGNORECASE
    )
    # NOW() - INTERVAL ... is valid in PostgreSQL, leave it
    return sql


# 1b. fix alias-in-HAVING using sqlglot

def fix_alias_in_having(sql: str) -> str:
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return sql

    select = tree.find(exp.Select)
    having = tree.find(exp.Having)

    if not select or not having:
        return sql

    # build a map of alias to expression from the SELECT clause
    alias_map = {}
    for expr in select.expressions:
        if isinstance(expr, exp.Alias):
            alias_map[expr.alias.lower()] = expr.this

    if not alias_map:
        return sql

    # replace column references in HAVING that match aliases
    def replace_alias(node):
        if isinstance(node, exp.Column) and not node.table:
            name = node.name.lower()
            if name in alias_map:
                return alias_map[name].copy()
        return node

    new_having = having.transform(replace_alias)
    having.replace(new_having)

    return tree.sql(dialect="postgres")


# 1c. validate by running the query against the database

def check_sql_executes(sql: str, conn) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(sql)
        conn.rollback()
        cur.close()
        return True
    except Exception:
        conn.rollback()
        return False


# main cleaning logic

def clean_split(data: list, conn, split_name: str) -> list:
    total = len(data)
    fixed_mysql = 0
    fixed_having = 0
    removed = 0
    cleaned = []

    for d in data:
        sql = d["sql_query"]
        original = sql

        # fix MySQL syntax
        sql = fix_mysql_syntax(sql)
        if sql != original:
            fixed_mysql += 1

        # fix alias-in-HAVING
        sql_after_having = fix_alias_in_having(sql)
        if sql_after_having != sql:
            fixed_having += 1
            sql = sql_after_having

        d["sql_query"] = sql

        # validate against the database
        if check_sql_executes(sql, conn):
            cleaned.append(d)
        else:
            removed += 1

    print(f"  {split_name}: kept {len(cleaned)} of {total} "
          f"(mysql fixes: {fixed_mysql}, having fixes: {fixed_having}, removed: {removed})")
    return cleaned


def remove_cross_split_duplicates(train, val, test):
    protected = set(d["natural_language"] for d in val) | set(d["natural_language"] for d in test)
    before = len(train)
    train = [d for d in train if d["natural_language"] not in protected]
    removed = before - len(train)
    if removed:
        print(f"  Removed {removed} duplicate question(s) from train that appear in val/test")
    return train


def load(path):
    with open(path) as f:
        return json.load(f)


def save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    conn = psycopg2.connect(**DB_CONFIG)

    backup_originals()

    print("Loading splits...")
    train = load(SPLITS["train"])
    val   = load(SPLITS["validation"])
    test  = load(SPLITS["test"])

    print("\nCleaning splits...")
    train = clean_split(train, conn, "train")
    val   = clean_split(val,   conn, "validation")
    test  = clean_split(test,  conn, "test")

    print("\nRemoving cross-split duplicates...")
    train = remove_cross_split_duplicates(train, val, test)

    print("\nSaving cleaned splits...")
    save(train, SPLITS["train"])
    save(val,   SPLITS["validation"])
    save(test,  SPLITS["test"])

    conn.close()

    print(f"\nFinal counts. train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print("Done.")


if __name__ == "__main__":
    main()
