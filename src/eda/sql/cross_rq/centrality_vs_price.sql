WITH price_by_buyer AS (
    SELECT
        buyer_id,
        AVG(anomaly_score) AS mean_anomaly_score,
        SUM(CASE WHEN consensus_flag THEN 1 ELSE 0 END) AS consensus_anomalies
    FROM rq3_anomalies
    GROUP BY buyer_id
)
SELECT
    ROUND(b.buyer_dependency_normalized, 2) AS centrality_band,
    AVG(p.mean_anomaly_score) AS mean_price_anomaly,
    SUM(p.consensus_anomalies) AS consensus_anomalies
FROM price_by_buyer p
JOIN rq1_buyer_metrics b USING (buyer_id)
GROUP BY centrality_band
ORDER BY centrality_band;
