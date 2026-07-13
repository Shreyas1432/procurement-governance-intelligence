import hashlib
import json
import re
from pathlib import Path

import networkx as nx
import polars as pl
import pytest

from src.core.config import DATA_RESULTS, ROOT
from src.rq4_resilience.resilience import build_exposure_table, load_sample, simulate_removal

LOCKED_ARTIFACTS = [
    ROOT / "refactor" / "golden_master.json",
    DATA_RESULTS / "rq1_success_metrics.json",
    DATA_RESULTS / "rq2_success_metrics.json",
    DATA_RESULTS / "rq3_success_metrics.json",
    DATA_RESULTS / "integration_metrics.json",
    DATA_RESULTS / "cross_effect_sizes.json",
    DATA_RESULTS / "model_performance_ci.json",
]

BANNED_WORDS = [
    "damage", "loss", "lost", "cost_of_failure", "impact_eur",
    "revenue", "market_share", "vulnerability", "bankruptcy", "default_risk",
]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_existing(paths):
    return {p: _hash_file(p) for p in paths if p.exists()}


@pytest.fixture(scope="module")
def sample():
    graph, coverage = load_sample()
    return graph, coverage


class TestDeterminism:
    def test_two_samples_are_identical(self):
        g1, c1 = load_sample()
        g2, c2 = load_sample()
        assert set(g1.nodes()) == set(g2.nodes())
        assert set(g1.edges()) == set(g2.edges())
        assert c1["coverage"] == c2["coverage"]

    def test_two_metrics_runs_are_identical(self):
        from src.rq4_resilience.resilience import run_resilience
        run_resilience()
        first = json.loads((DATA_RESULTS / "resilience_metrics.json").read_text())
        first_table = pl.read_parquet(DATA_RESULTS / "resilience_exposure_table.parquet")
        run_resilience()
        second = json.loads((DATA_RESULTS / "resilience_metrics.json").read_text())
        second_table = pl.read_parquet(DATA_RESULTS / "resilience_exposure_table.parquet")
        assert first == second
        assert first_table.equals(second_table)


class TestConservation:
    def test_single_node_exposed_value_matches_independent_sum(self, sample):
        graph, _ = sample
        node = max(graph.nodes(), key=lambda n: graph.degree(n, weight="weight"))
        result = simulate_removal(graph, [node])
        independent_sum = sum(
            d.get("weight", 0.0) for _, _, d in graph.edges([node], data=True)
        )
        assert result["exposed_contract_value"] == pytest.approx(independent_sum)


class TestSanity:
    def test_degree1_supplier_orphans_its_degree1_buyer(self, sample):
        graph, _ = sample
        target = None
        for n, d in graph.nodes(data=True):
            if d.get("bipartite") == 1 and graph.degree(n) == 1:
                buyer = next(iter(graph.neighbors(n)))
                if graph.degree(buyer) == 1:
                    target = (n, buyer)
                    break
        if target is None:
            pytest.skip("No degree-1 supplier with a degree-1 buyer counterpart in this sample")
        supplier, buyer = target
        result = simulate_removal(graph, [supplier])
        assert buyer in result["orphaned_counterparties"]
        assert buyer in result["stranded_buyers"]

    def test_removing_unknown_node_raises(self, sample):
        graph, _ = sample
        with pytest.raises(ValueError):
            simulate_removal(graph, ["NOT_A_REAL_NODE"])


class TestNoMutation:
    def test_input_graph_unchanged_after_simulate_removal(self, sample):
        graph, _ = sample
        nodes_before = set(graph.nodes())
        edges_before = set(graph.edges())
        node = next(iter(graph.nodes()))
        simulate_removal(graph, [node])
        assert set(graph.nodes()) == nodes_before
        assert set(graph.edges()) == edges_before

    def test_input_graph_unchanged_after_build_exposure_table(self, sample):
        graph, _ = sample
        nodes_before = set(graph.nodes())
        edges_before = set(graph.edges())
        build_exposure_table(graph)
        assert set(graph.nodes()) == nodes_before
        assert set(graph.edges()) == edges_before


class TestLockedArtifactsUntouched:
    def test_locked_artifacts_byte_identical(self):
        before = _hash_existing(LOCKED_ARTIFACTS)
        from src.rq4_resilience.resilience import run_resilience
        run_resilience()
        after = _hash_existing(LOCKED_ARTIFACTS)
        assert before == after, "Resilience run touched a locked artifact"


def _strip_mandated_caveat(text: str) -> str:
    # The FRAMING RULES mandate a caveat string that must literally contain
    # "financial loss" as a negation ("not an estimate of financial loss").
    # The vocabulary guard checks that banned words are never used to NAME or
    # FRAME the metric elsewhere; it must not flag the mandated phrase itself,
    # however it is wrapped/split as a Python string literal in source.
    return re.sub(r"financial\s+loss", "", text, flags=re.IGNORECASE)


class TestVocabularyGuard:
    def test_no_banned_words_in_module_source(self):
        source = _strip_mandated_caveat(
            (ROOT / "src" / "rq4_resilience" / "resilience.py").read_text().lower()
        )
        hits = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", source)]
        assert not hits, f"Banned vocabulary found in resilience.py: {hits}"

    def test_no_banned_words_in_metrics_json(self):
        path = DATA_RESULTS / "resilience_metrics.json"
        assert path.exists()
        text = _strip_mandated_caveat(path.read_text().lower())
        hits = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text)]
        assert not hits, f"Banned vocabulary found in resilience_metrics.json: {hits}"

    def test_no_banned_words_in_exposure_table_columns(self):
        path = DATA_RESULTS / "resilience_exposure_table.parquet"
        assert path.exists()
        df = pl.read_parquet(path)
        cols = " ".join(df.columns).lower()
        hits = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", cols)]
        assert not hits, f"Banned vocabulary found in exposure table columns: {hits}"
