import random
import psycopg2
from faker import Faker
from dotenv import load_dotenv
import os

load_dotenv()

# fixed seeds so the generated database is reproducible
random.seed(42)
Faker.seed(42)
fake = Faker()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

NUM_CUSTOMERS   = 1000
NUM_EMPLOYEES   = 50
NUM_PRODUCTS    = 200
NUM_ORDERS      = 5000
NUM_ORDER_ITEMS = 15000

def seed():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("Seeding employees...")
    cur.executemany(
        "INSERT INTO employees (full_name, region, commission_rate) VALUES (%s, %s, %s)",
        [
            (
                fake.name(),
                random.choice(["west", "central", "east"]),
                round(random.uniform(0.03, 0.15), 4),
            )
            for _ in range(NUM_EMPLOYEES)
        ],
    )

    print("Seeding customers...")
    cur.executemany(
        "INSERT INTO customers (full_name, email, city, segment, signup_date) VALUES (%s, %s, %s, %s, %s)",
        [
            (
                fake.name(),
                fake.unique.email(),
                fake.city(),
                random.choice(["retail", "corporate", "vip"]),
                fake.date_between(start_date="-5y", end_date="today"),
            )
            for _ in range(NUM_CUSTOMERS)
        ],
    )

    print("Seeding products...")
    categories = ["Electronics", "Clothing", "Food", "Office", "Tools", "Sports", "Beauty"]
    cur.executemany(
        "INSERT INTO products (product_name, category, price, cost) VALUES (%s, %s, %s, %s)",
        [
            (
                fake.bs().title()[:150],
                random.choice(categories),
                round(random.uniform(5, 500), 2),
                round(random.uniform(2, 300), 2),
            )
            for _ in range(NUM_PRODUCTS)
        ],
    )

    conn.commit()

    # Fetch IDs for FK use
    cur.execute("SELECT customer_id FROM customers")
    customer_ids = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT employee_id FROM employees")
    employee_ids = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT product_id FROM products")
    product_ids = [r[0] for r in cur.fetchall()]

    print("Seeding orders...")
    cur.executemany(
        "INSERT INTO orders (customer_id, employee_id, order_date, status, total_amount) VALUES (%s, %s, %s, %s, %s)",
        [
            (
                random.choice(customer_ids),
                random.choice(employee_ids),
                fake.date_between(start_date="-3y", end_date="today"),
                random.choice(["completed", "cancelled", "pending"]),
                0,  # will update after order_items are inserted
            )
            for _ in range(NUM_ORDERS)
        ],
    )
    conn.commit()

    cur.execute("SELECT order_id FROM orders")
    order_ids = [r[0] for r in cur.fetchall()]

    print("Seeding order_items...")
    cur.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (%s, %s, %s, %s)",
        [
            (
                random.choice(order_ids),
                random.choice(product_ids),
                random.randint(1, 10),
                round(random.uniform(5, 500), 2),
            )
            for _ in range(NUM_ORDER_ITEMS)
        ],
    )
    conn.commit()

    print("Updating order totals...")
    cur.execute("""
        UPDATE orders o
        SET total_amount = sub.total
        FROM (
            SELECT order_id, SUM(quantity * unit_price) AS total
            FROM order_items
            GROUP BY order_id
        ) sub
        WHERE o.order_id = sub.order_id
    """)
    conn.commit()

    cur.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    seed()
