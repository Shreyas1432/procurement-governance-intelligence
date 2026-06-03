SELECT
    ROUND(b.buyer_dependency_normalized, 2) AS centrality_band,
    AVG(p.ensemble_risk_prob) AS mean_governance_risk,
    COUNT(*) AS contracts
FROM rq2_predictions p
JOIN rq1_buyer_metrics b USING (buyer_id)
GROUP BY centrality_band
ORDER BY centrality_band;
