SELECT
    cpv_division,
    COUNT(*) AS priced_contracts,
    SUM(CASE WHEN consensus_flag THEN 1 ELSE 0 END) AS consensus_anomalies,
    AVG(anomaly_score) AS mean_anomaly_score
FROM rq3_anomalies
GROUP BY cpv_division
ORDER BY consensus_anomalies DESC
LIMIT 25;
