WITH buyer_communities AS (
    SELECT entity_id AS buyer_id, community_id
    FROM rq1_communities
    WHERE node_type = 'buyer'
),
risk_by_buyer AS (
    SELECT buyer_id, AVG(ensemble_risk_prob) AS mean_governance_risk
    FROM rq2_predictions
    GROUP BY buyer_id
)
SELECT
    c.community_id,
    COUNT(*) AS buyers,
    AVG(r.mean_governance_risk) AS mean_governance_risk
FROM buyer_communities c
JOIN risk_by_buyer r USING (buyer_id)
GROUP BY c.community_id
ORDER BY buyers DESC
LIMIT 30;
