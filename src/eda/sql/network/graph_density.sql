WITH nodes AS (
    SELECT COUNT(DISTINCT buyer_id) AS buyers, COUNT(DISTINCT supplier_id) AS suppliers
    FROM rq1_features
),
edges AS (
    SELECT COUNT(*) AS edges
    FROM (SELECT DISTINCT buyer_id, supplier_id FROM rq1_features)
)
SELECT
    buyers,
    suppliers,
    edges,
    edges / NULLIF(CAST(buyers * suppliers AS DOUBLE), 0) AS graph_density
FROM nodes
CROSS JOIN edges;
