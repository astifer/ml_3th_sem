CREATE DATABASE main_database;

\c main_database;

CREATE TABLE main_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

INSERT INTO main_table (name) VALUES ('First entry'), ('Second entry');


CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    salary DECIMAL(10, 2)
);


INSERT INTO employees (name, salary) VALUES ('John Doe', 50000.00), ('Jane Smith', 60000.00);