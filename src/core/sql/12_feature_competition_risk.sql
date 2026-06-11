SELECT
    contract_id,
    procedure_type,
    bid_count,
    contract_amendments,
    CASE
        WHEN procedure_type IN ('NEGOTIATED_WITHOUT_PUBLICATION', 'OUTRIGHT_AWARD') THEN 1.0
        WHEN procedure_type ILIKE '%NEGOTIATED%' THEN 0.75
        WHEN procedure_type IN ('RESTRICTED', 'COMPETITIVE_DIALOG') THEN 0.35
        WHEN procedure_type = 'OPEN' THEN 0.05
        ELSE 0.50
    END AS procedure_risk,
    CASE
        WHEN bid_count IS NULL THEN NULL
        WHEN bid_count <= 1 THEN 1.0
        WHEN bid_count = 2 THEN 0.65
        WHEN bid_count BETWEEN 3 AND 5 THEN 0.30
        ELSE 0.05
    END AS competition_risk
FROM contracts_base;
