# Explanation: Supplier Network Intelligence (`rq1_network.py`)

This document explains the **Supplier Network Intelligence** script (`src/rq1_supplier_network/rq1_network.py`). It answers **why** we use it, **how** it works, and what **business outcomes** it delivers.

---

## 1. Why Are We Using It?

In public procurement, buyers (government agencies) and suppliers (companies) do not work in isolation. They form a complex network of contracts and financial relationships. Traditional analysis tools look at individual companies or contracts one by one. This approach misses the "big picture" systemic risks.

We use `rq1_network.py` to:
* **Model Connections as a Network**: By using graph theory, we represent buyers and suppliers as "nodes" (points) and contracts as "edges" (lines connecting the points). Because buyers only contract with suppliers (and not with other buyers), this is modeled as a **bipartite graph**.
* **Identify Systemic Vulnerabilities**: We want to understand which suppliers are "too big to fail," which buyers are overly dependent on a single vendor, and whether there are hidden groups or cliques that might suggest anticompetitive behavior.

---

## 2. How Are We Using It?

The script is executed as part of the data pipeline (via `make rq1` or `run_full_pipeline.py`). It performs three main algorithms on the contract data:

### A. Network Centrality (Importance Scoring)
* **What it does**: Measures how critical a buyer or supplier is in the network.
* **How it works**: It calculates two types of centrality:
  * **Degree Centrality**: The number of different partners a node is connected to (i.e. how many suppliers a buyer uses, or how many buyers a supplier serves).
  * **Betweenness Centrality**: How often a node acts as a bridge or connector between different parts of the network.
* **Result**: Compiles a normalized importance score between `0` (low importance) and `1` (systemically critical).

### B. Louvain Community Detection (Market Grouping)
* **What it does**: Automatically groups buyers and suppliers into "communities" or clusters based on who does business with whom.
* **How it works**: It groups nodes together to maximize a metric called "modularity" (nodes inside a community are highly connected to each other, but have few connections to other communities).
* **Result**: Assigns every buyer and supplier a community ID, showing which market cluster they belong to.

### C. Supply Chain Resilience (Risk Analysis)
* **What it does**: Measures how vulnerable a buyer is to their largest supplier exiting the market.
* **How it works**: For each buyer, it calculates:
  * How many suppliers they have.
  * The share of total contract value won by their top supplier (**value concentration**).
* **Result**: Assigns a **resilience score** to each buyer. A high score means a buyer is highly dependent on a single supplier (low resilience), whereas a low score indicates a diversified supplier base (high resilience).

---

## 3. What Are the Business Outcomes?

Using network intelligence provides several critical advantages and business outcomes:

### 1. Proactive Risk Management
* **Outcome**: Public auditors can identify government agencies that rely on a single contractor for the majority of their services.
* **Value**: If that contractor experiences financial trouble or defaults, the government can intervene early or prepare backup suppliers, avoiding disruptions to public services (like public transport or waste management).

### 2. Collusion and Cartel Detection
* **Outcome**: Louvain community detection flags clusters where a small group of suppliers consistently wins contracts from the same small group of buyers.
* **Value**: These isolated clusters can be automatically flagged for investigation by antitrust authorities to check for bid-rigging or favoritism.

### 3. Focused Audit Allocation
* **Outcome**: Instead of auditing thousands of suppliers randomly, authorities can sort nodes by their **Betweenness Centrality**.
* **Value**: Auditing the most central hubs in the network ensures that oversight resources are focused where they have the maximum systemic impact.

### 4. Improving Market Competition
* **Outcome**: The script reports the buyer-dependency vs single-bidder-rate correlation as a diagnostic. At present this is **uninformative**: `single_bidder_rate` is observable for only ~0.9% of buyers (lot_bidscount sparsity), so the headline value is `null` (`degenerate_low_coverage`) and only an under-powered observed-only r (−0.038) is recorded; see `docs/THRESHOLD_JUSTIFICATION.md` Sections 3, 9.
* **Value**: With adequate bid-count coverage this diagnostic could test whether reducing supplier concentration is associated with fewer single-bidder contracts; it is framed as a coverage-limited indicator, not as causal proof.
