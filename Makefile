.PHONY: features rq1 rq2 rq3 pipeline audit test dashboard figures strip clean

PY = .venv/bin/python

features:
	$(PY) -m src.core.feature_engineering

rq1:
	$(PY) -m src.rq1_supplier_network.rq1_network

rq2:
	$(PY) -m src.rq2_governance_risk.rq2_governance

rq3:
	$(PY) -m src.rq3_price_intelligence.rq3_pricing

pipeline:
	$(PY) run_full_pipeline.py

audit:
	$(PY) tests/test_leakage_audit.py

test:
	.venv/bin/pytest tests/ -v

dashboard:
	.venv/bin/streamlit run src/dashboard/app.py

figures:
	$(PY) src/figures/build_figures.py

strip:
	$(PY) scripts/strip_comments.py

clean:
	rm -rf dist
