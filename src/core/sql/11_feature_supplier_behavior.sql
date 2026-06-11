WITH supplier_history AS (
    SELECT
        s.supplier_id,
        s.contract_id AS focal_contract,
        s.award_date AS focal_award_date,
        COUNT(h.contract_id) AS prior_contracts,
        AVG(h.bid_count) AS avg_prior_bids,
        SUM(CASE WHEN h.bid_count = 1 THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(h.contract_id), 0) AS single_bid_rate,
        COUNT(DISTINCT h.cpv_division) AS cpv_diversity,
        STDDEV(h.final_price) / NULLIF(AVG(h.final_price), 0) AS price_cv,
        AVG(h.final_price) AS avg_prior_price,
        MAX(h.award_date) AS supplier_feature_latest_date
    FROM contracts_base s
    LEFT JOIN contracts_base h
      ON s.supplier_id = h.supplier_id
     AND h.award_date < s.award_date
    WHERE s.supplier_id IS NOT NULL
    GROUP BY s.supplier_id, s.contract_id, s.award_date
)
SELECT *
FROM supplier_history
WHERE prior_contracts >= 3;
