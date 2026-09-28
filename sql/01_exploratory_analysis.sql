SELECT COUNT(*)
FROM sales;

SELECT *
FROM sales
LIMIT 10;

SELECT
    MIN(date) AS start_date,
    MAX(date) AS end_date,
    COUNT(DISTINCT store) AS stores,
    COUNT(DISTINCT item) AS items,
    COUNT(*) AS observations
FROM sales;

SELECT
    store,
    SUM(sales) AS total_sales,
    ROUND(AVG(sales), 2) AS avg_daily_sales
FROM sales
GROUP BY store
ORDER BY total_sales DESC;

SELECT
    item,
    SUM(sales) AS total_sales,
    ROUND(AVG(sales), 2) AS avg_sales
FROM sales
GROUP BY item
ORDER BY total_sales DESC
LIMIT 10;

SELECT
    EXTRACT(YEAR FROM date) AS year,
    SUM(sales) AS total_sales
FROM sales
GROUP BY EXTRACT(YEAR FROM date)
ORDER BY year;

SELECT
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    SUM(sales) AS total_sales
FROM sales
GROUP BY
    EXTRACT(YEAR FROM date),
    EXTRACT(MONTH FROM date)
ORDER BY year, month;

WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', date) AS month,
        SUM(sales) AS total_sales
    FROM sales
    GROUP BY DATE_TRUNC('month', date)
)

SELECT *
FROM monthly_sales
ORDER BY month;

WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', date) AS month,
        SUM(sales) AS total_sales
    FROM sales
    GROUP BY DATE_TRUNC('month', date)
)

SELECT
    month,
    total_sales,

    LAG(total_sales) OVER (
        ORDER BY month
    ) AS previous_month_sales

FROM monthly_sales
ORDER BY month;

WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', date) AS month,
        SUM(sales) AS total_sales
    FROM sales
    GROUP BY DATE_TRUNC('month', date)
),

sales_with_lag AS (
    SELECT
        month,
        total_sales,
        LAG(total_sales) OVER (
            ORDER BY month
        ) AS previous_month_sales
    FROM monthly_sales
)

SELECT
    month,
    total_sales,
    previous_month_sales,

    ROUND(
        100.0 *
        (total_sales - previous_month_sales)
        / NULLIF(previous_month_sales, 0),
        2
    ) AS growth_pct

FROM sales_with_lag
ORDER BY month;

WITH item_sales AS (
    SELECT
        store,
        item,
        SUM(sales) AS total_sales
    FROM sales
    GROUP BY store, item
),

ranked_items AS (
    SELECT
        store,
        item,
        total_sales,

        RANK() OVER (
            PARTITION BY store
            ORDER BY total_sales DESC
        ) AS item_rank

    FROM item_sales
)

SELECT *
FROM ranked_items
WHERE item_rank <= 5
ORDER BY store, item_rank;

-- =============================================
-- Retail Demand Forecasting
-- Exploratory SQL Analysis
-- =============================================


-- Dataset validation

SELECT
    MIN(date) AS start_date,
    MAX(date) AS end_date,
    COUNT(DISTINCT store) AS stores,
    COUNT(DISTINCT item) AS items,
    COUNT(*) AS observations
FROM sales;


-- Total sales by store

SELECT
    store,
    SUM(sales) AS total_sales,
    ROUND(AVG(sales), 2) AS avg_daily_sales
FROM sales
GROUP BY store
ORDER BY total_sales DESC;