SELECT 'total_rows' AS metric, COUNT(*)::DOUBLE AS value FROM contracts_base
UNION ALL SELECT 'missing_buyer_id', COUNT(*)::DOUBLE FROM contracts_base WHERE buyer_id IS NULL
UNION ALL SELECT 'missing_supplier_id', COUNT(*)::DOUBLE FROM contracts_base WHERE supplier_id IS NULL
UNION ALL SELECT 'duplicate_contract_ids', (COUNT(*) - COUNT(DISTINCT contract_id))::DOUBLE FROM contracts_base
UNION ALL SELECT 'non_positive_prices', COUNT(*)::DOUBLE FROM contracts_base WHERE final_price <= 0
UNION ALL SELECT 'invalid_award_year', COUNT(*)::DOUBLE FROM contracts_base WHERE award_year < 2006 OR award_year > 2026;
