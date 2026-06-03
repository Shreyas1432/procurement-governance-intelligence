SELECT
    community_id,
    COUNT(*) AS nodes,
    SUM(CASE WHEN node_type = 'buyer' THEN 1 ELSE 0 END) AS buyers,
    SUM(CASE WHEN node_type = 'supplier' THEN 1 ELSE 0 END) AS suppliers,
    AVG(modularity) AS modularity
FROM rq1_communities
GROUP BY community_id
ORDER BY nodes DESC
LIMIT 30;
