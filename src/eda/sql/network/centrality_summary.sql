SELECT
    node_type,
    ROUND(centrality, 3) AS centrality_band,
    COUNT(*) AS nodes
FROM rq1_network_metrics
GROUP BY node_type, centrality_band
ORDER BY node_type, centrality_band;
