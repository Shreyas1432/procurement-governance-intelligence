SELECT
    node_type,
    degree,
    COUNT(*) AS node_count
FROM rq1_network_metrics
GROUP BY node_type, degree
ORDER BY node_type, degree;
