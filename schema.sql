-- Reset: drop existing tables so this script can be re-run from scratch.
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders      CASCADE;
DROP TABLE IF EXISTS products    CASCADE;
DROP TABLE IF EXISTS employees   CASCADE;
DROP TABLE IF EXISTS customers   CASCADE;

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    full_name    VARCHAR(100) NOT NULL,
    email        VARCHAR(100) UNIQUE NOT NULL,
    city         VARCHAR(100),
    segment      VARCHAR(20) CHECK (segment IN ('retail', 'corporate', 'vip')),
    signup_date  DATE
);

CREATE TABLE employees (
    employee_id      SERIAL PRIMARY KEY,
    full_name        VARCHAR(100) NOT NULL,
    region           VARCHAR(10) CHECK (region IN ('west', 'central', 'east')),
    commission_rate  NUMERIC(5, 4)
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    product_name  VARCHAR(150) NOT NULL,
    category      VARCHAR(100),
    price         NUMERIC(12, 2),
    cost          NUMERIC(12, 2)
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT NOT NULL REFERENCES customers(customer_id),
    employee_id   INT REFERENCES employees(employee_id),
    order_date    DATE,
    status        VARCHAR(20) CHECK (status IN ('completed', 'cancelled', 'pending')),
    total_amount  NUMERIC(12, 2)
);

CREATE TABLE order_items (
    order_item_id  SERIAL PRIMARY KEY,
    order_id       INT NOT NULL REFERENCES orders(order_id),
    product_id     INT NOT NULL REFERENCES products(product_id),
    quantity       INT,
    unit_price     NUMERIC(12, 2)
);
