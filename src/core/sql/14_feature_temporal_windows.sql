SELECT
    s.contract_id,
    s.supplier_id,
    s.award_date AS focal_award_date,
    COUNT(h.contract_id) AS supplier_contracts_prior_365d,
    AVG(h.final_price) AS supplier_avg_price_prior_365d,
    AVG(CASE WHEN h.bid_count <= 1 THEN 1.0 ELSE 0.0 END) AS supplier_single_bid_rate_prior_365d,
    MAX(h.award_date) AS supplier_feature_latest_date
FROM contracts_base s
LEFT JOIN contracts_base h
  ON s.supplier_id = h.supplier_id
 AND h.award_date < s.award_date
 AND h.award_date >= s.award_date - INTERVAL 365 DAY
WHERE s.supplier_id IS NOT NULL
GROUP BY s.contract_id, s.supplier_id, s.award_date;
