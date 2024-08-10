# Comprehensive SQL for Data Science

## Introduction

SQL (Structured Query Language) is an essential tool for data scientists, used for managing, manipulating, and analyzing relational databases. This covers key SQL commands and concepts, with a focus on data science applications.

## Table of Contents

1. [Basic Queries](#basic-queries)
2. [Filtering and Sorting](#filtering-and-sorting)
3. [Aggregate Functions](#aggregate-functions)
4. [Joins](#joins)
5. [Subqueries](#subqueries)
6. [Data Manipulation](#data-manipulation)
7. [Table Operations](#table-operations)
8. [Window Functions](#window-functions)
9. [Advanced Techniques](#advanced-techniques)
10. [Best Practices for Data Scientists](#best-practices-for-data-scientists)

## Basic Queries

### SELECT Statement

```sql
-- Select all columns from a table
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Select distinct values
SELECT DISTINCT department FROM employees;
```

### Aliasing

```sql
-- Alias for columns
SELECT first_name AS "First Name", last_name AS "Last Name" FROM employees;

-- Alias for tables
SELECT e.first_name, e.last_name FROM employees e;
```

## Filtering and Sorting

### WHERE Clause

```sql
-- Basic filtering
SELECT * FROM employees WHERE department = 'Sales';

-- Multiple conditions
SELECT * FROM employees WHERE department = 'Sales' AND salary > 50000;

-- IN operator
SELECT * FROM employees WHERE department IN ('Sales', 'Marketing');

-- LIKE operator for pattern matching
SELECT * FROM employees WHERE last_name LIKE 'S%';
```

### ORDER BY Clause

```sql
-- Sort in ascending order
SELECT * FROM employees ORDER BY salary;

-- Sort in descending order
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple sorting criteria
SELECT * FROM employees ORDER BY department, salary DESC;
```

## Aggregate Functions

```sql
-- Count
SELECT COUNT(*) FROM employees;

-- Sum
SELECT SUM(salary) FROM employees;

-- Average
SELECT AVG(salary) FROM employees;

-- Min and Max
SELECT MIN(salary), MAX(salary) FROM employees;

-- Group By
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department;

-- Having (for filtering grouped data)
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 60000;
```

## Joins

```sql
-- Inner Join
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Left Join
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Right Join
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;

-- Full Outer Join
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.department_id;
```

## Subqueries

```sql
-- Subquery in WHERE clause
SELECT * FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'New York');

-- Subquery in SELECT
SELECT 
    first_name, 
    last_name, 
    salary,
    (SELECT AVG(salary) FROM employees) AS avg_company_salary
FROM employees;

-- Correlated subquery
SELECT e.first_name, e.last_name, e.salary
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id);
```

## Data Manipulation

```sql
-- Insert data
INSERT INTO employees (first_name, last_name, salary, department_id)
VALUES ('John', 'Doe', 50000, 1);

-- Update data
UPDATE employees
SET salary = salary * 1.1
WHERE department_id = 1;

-- Delete data
DELETE FROM employees
WHERE employee_id = 100;
```

## Table Operations

```sql
-- Create table
CREATE TABLE projects (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(100),
    start_date DATE,
    end_date DATE
);

-- Alter table
ALTER TABLE projects
ADD COLUMN budget DECIMAL(10,2);

-- Drop table
DROP TABLE projects;
```

## Window Functions

```sql
-- Rank employees by salary within each department
SELECT 
    first_name,
    last_name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank
FROM employees;

-- Running total of salary
SELECT 
    first_name,
    last_name,
    salary,
    SUM(salary) OVER (ORDER BY employee_id) AS running_total
FROM employees;

-- Moving average of salary (last 3 employees)
SELECT 
    first_name,
    last_name,
    salary,
    AVG(salary) OVER (ORDER BY employee_id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM employees;
```

## Advanced Techniques

### Common Table Expressions (CTE)

```sql
WITH high_salary_employees AS (
    SELECT * FROM employees WHERE salary > 70000
)
SELECT department, COUNT(*) AS high_earners
FROM high_salary_employees
GROUP BY department;
```

### CASE Statement

```sql
SELECT 
    first_name,
    last_name,
    salary,
    CASE 
        WHEN salary < 50000 THEN 'Low'
        WHEN salary BETWEEN 50000 AND 100000 THEN 'Medium'
        ELSE 'High'
    END AS salary_category
FROM employees;
```

### Pivoting Data

```sql
-- Assuming we have a sales table with columns: product, year, and amount

SELECT 
    product,
    SUM(CASE WHEN year = 2021 THEN amount ELSE 0 END) AS "2021",
    SUM(CASE WHEN year = 2022 THEN amount ELSE 0 END) AS "2022",
    SUM(CASE WHEN year = 2023 THEN amount ELSE 0 END) AS "2023"
FROM sales
GROUP BY product;
```

## Best Practices for Data Science

1. **Use appropriate indexing**: Properly indexed tables can significantly speed up query performance.

2. **Write efficient queries**: Avoid SELECT * when you only need specific columns. Use JOINs judiciously.

3. **Use CTEs for complex queries**: Common Table Expressions can make your queries more readable and maintainable.

4. **Understand the execution plan**: Use EXPLAIN or similar tools to understand how your queries are executed and optimize them.

5. **Be cautious with large-scale data manipulation**: When working with large datasets, consider the impact of your queries on database performance.

6. **Use transactions for data integrity**: When performing multiple related operations, use transactions to ensure data consistency.

7. **Regularly maintain and optimize your database**: This includes updating statistics, rebuilding indexes, and purging unnecessary data.

8. **Handle NULL values appropriately**: Be aware of how NULL values affect your queries and results.

9. **Use parameterized queries**: When working with application code, use parameterized queries to prevent SQL injection and improve performance.

10. **Document your complex queries**: Add comments to explain the purpose and logic of complex SQL queries for future reference.

Remember, the specific syntax and available features may vary slightly depending on the database management system you're using (e.g., MySQL, PostgreSQL, SQL Server). Always refer to the documentation of your specific DBMS for the most accurate information.