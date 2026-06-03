SELECT 'buyers' AS entity, COUNT(DISTINCT buyer_id) AS distinct_count FROM contracts_clean
UNION ALL
SELECT 'suppliers', COUNT(DISTINCT supplier_id) FROM contracts_clean
UNION ALL
SELECT 'cpv_divisions', COUNT(DISTINCT cpv_division) FROM contracts_clean
UNION ALL
SELECT 'procedures', COUNT(DISTINCT procedure_type) FROM contracts_clean
UNION ALL
SELECT 'award_years', COUNT(DISTINCT award_year) FROM contracts_clean
ORDER BY distinct_count DESC;
