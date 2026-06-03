SELECT 'buyer_concentration_hhi' AS feature, COUNT(*) AS rows, AVG(buyer_concentration_hhi) AS mean_value, COUNT(buyer_concentration_hhi) / COUNT(*) AS completeness FROM rq2_features
UNION ALL
SELECT 'single_bid_rate', COUNT(*), AVG(single_bid_rate), COUNT(single_bid_rate) / COUNT(*) FROM rq2_features
UNION ALL
SELECT 'cpv_risk_score', COUNT(*), AVG(cpv_risk_score), COUNT(cpv_risk_score) / COUNT(*) FROM rq2_features
UNION ALL
SELECT 'award_year', COUNT(*), AVG(award_year), COUNT(award_year) / COUNT(*) FROM rq2_features;
