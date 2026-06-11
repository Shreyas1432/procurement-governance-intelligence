WITH cpv_counts AS (
    SELECT cpv_division, COUNT(*) AS n
    FROM contracts_base
    WHERE cpv_division IS NOT NULL AND cpv_division != 'UNKNOWN' AND final_price > 0
    GROUP BY cpv_division
    -- Gate lowered 1000 -> 300 to admit 12 CPV sectors under the year-stratified
    -- sample (cap=100k); 300+ priced contracts per sector keeps the per-CPV median
    -- baseline and the OneHotEncoder(min_frequency=25) stable. See cap/gate sweep.
    HAVING COUNT(*) >= 300
)
SELECT
    c.contract_id,
    c.buyer_id,
    c.supplier_id,
    c.cpv_division,
    c.final_price,
    LN(c.final_price + 1) AS log_price,
    c.estimated_price,
    LN(c.estimated_price + 1) AS log_estimated_price,
    c.bid_count,
    c.procedure_type,
    c.award_year,
    c.contract_amendments
FROM contracts_base c
JOIN cpv_counts cc ON c.cpv_division = cc.cpv_division
WHERE c.final_price > 0 AND c.estimated_price > 0;
