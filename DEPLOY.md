# Deploying the PGI Dashboard

This document covers both deploy paths for `src/dashboard/`. Read the
**deploy gate** section first: it explains why there are two builds and
why they must never be mixed.

Nothing in this document has been run. No git operations, no history
rewrites, and no actual deployment were performed while preparing it.

## The deploy gate

The dashboard's RQ2 screen labels real named suppliers as "high governance
risk." That label is a procedural-recording-irregularity proxy (see
`docs/THRESHOLD_JUSTIFICATION.md` and the caveat banner on every
risk-bearing screen), not a corruption finding. Showing a named entity next
to that label to the public internet is a legal and ethical hazard even
though the underlying analysis is honest and well-documented.

So there are two builds of the same codebase, switched by one environment
variable, `PGI_PUBLIC_BUILD`:

| | PUBLIC build (`PGI_PUBLIC_BUILD=1`) | FULL build (default) |
|---|---|---|
| Audience | Public internet | Access-controlled only (private network, VPN, or an authenticating reverse proxy) |
| Roles | Viewer only. Admin sign-in is not rendered and is refused server-side even if attempted. | Viewer and admin |
| Named IDs | Never shown. Every identifier is a salted-hash pseudonym. | Admin sees real IDs behind an explicit, logged Reveal action |
| Requires | `anon_salt` at **build time only** (`scripts/build_public_data.py`, run locally before committing `data/public/`) -- the deployed app never calls `_anon_salt()` at runtime, since `anonymize_id()` is a passthrough under `PGI_PUBLIC_BUILD=1` on data that's already pseudonymized at rest. Streamlit Cloud's secrets manager does not need `anon_salt` configured. | Works with the local dev fallback salt, but should still set a real secret for anything beyond a laptop demo |

The two builds are **one codebase**, not a fork: `src/dashboard/_auth.py`
reads `PGI_PUBLIC_BUILD` once at import time and gates every admin/reveal
code path from that single flag, so there is only one file to audit for
this guarantee (see "How the gate is enforced" below).

## Prerequisite: `src/dashboard/` is currently not tracked by git

`.gitignore` has `src/dashboard/` as a blanket "WIP scaffolding not yet
locked" rule. That means a `git push` today ships **no dashboard code at
all**. Before a git-based hosted platform (path 1 below) can deploy
anything, you need to decide whether to narrow or remove that line and
commit the dashboard code. This decision was intentionally left to you: it
was not made as part of this deploy-prep task. The Docker path (path 2)
does not need this, since a local `docker build` reads the filesystem
directly and is unaffected by `.gitignore`.

## Path 1: public, viewer-only, pseudonymised (hosted platform)

For a platform that deploys from a git repo (Streamlit Community Cloud or
similar):

1. Resolve the prerequisite above (commit `src/dashboard/`, `requirements-app.txt`,
   `.streamlit/config.toml`).
2. Generate a real secret: `python -c "import secrets; print(secrets.token_hex(32))"`.
3. In the platform's secrets manager (not in the repo), set:
   ```toml
   anon_salt = "<the generated value>"
   ```
4. Set the environment variable `PGI_PUBLIC_BUILD=1` in the platform's
   environment configuration.
5. Point the platform at `src/dashboard/app.py` as the entry point and
   `requirements-app.txt` as the dependency file.
6. Deploy. Verify (see Part 4 verification below) before sharing the URL:
   admin sign-in is not offered, every ID on every screen is a pseudonym,
   and the Audit Log page does not appear in navigation.

`data/features/` and `data/results/` must be present in the deploy
(read-only artifacts, no raw data). `data/raw/` and `data/curated/` must
never be included; the dashboard never reads them.

## Path 2: full build, named IDs, access-controlled private host

For the admin build with named suppliers and buyers visible:

1. Build the image: `docker build -t pgi-dashboard .` (not run as part of
   this task; you run this).
2. Do **not** set `PGI_PUBLIC_BUILD` (or explicitly leave it unset) so the
   admin role and reveal path are available.
3. Provide `anon_salt` via a mounted secrets file or your host's secret
   manager, not baked into the image:
   ```
   docker run -p 8501:8501 \
     -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
     pgi-dashboard
   ```
4. Put the container behind access control before anyone else reaches it:
   a VPN, an internal network with no public ingress, or an authenticating
   reverse proxy (OIDC, SSO, or at minimum HTTP basic auth) in front of it.
   The dashboard's own auth (`_auth.py`) is documented in-product as an
   interim control, not a substitute for network-level access control on a
   named-data deployment.
5. Rotate or set the seeded default admin password
   (`admin` / `CHANGE_ME_ON_FIRST_LOGIN`) before anyone else can reach the
   instance -- it self-seeds on first run if `_auth_store.json` does not
   already exist.

## How the gate is enforced (for the audit trail)

All of the following live in `src/dashboard/_auth.py`, gated by the single
`PUBLIC_BUILD` module constant:

- `authenticate()` refuses to return the `ADMIN` role when `PUBLIC_BUILD`
  is set, even for a correct admin password (logged as
  `denied_public_build`).
- `_render_login_and_register()` never builds the admin sign-in tab in a
  public build -- it is not rendered, not just styled away.
- `reveal_or_anonymize()` always returns the anonymized pseudonym in a
  public build, independent of whatever role happens to be in session
  state. This is the second, independent gate: even if the first one were
  ever bypassed, this one still holds.
- `anonymize_id()` is a passthrough under `PGI_PUBLIC_BUILD=1`: every
  ID-shaped value the public build ever handles already came from
  `data/public/`, pre-anonymized at rest by `scripts/build_public_data.py`
  (which calls this same function with `PUBLIC_BUILD` unset, so the real
  `st.secrets["anon_salt"]`-keyed hash runs exactly once, at build time,
  against the production salt). The deployed app therefore never reads
  `anon_salt` at runtime; re-hashing an already-anonymized value would
  double-hash it and break cross-screen pseudonym consistency. Outside
  `PUBLIC_BUILD` (the FULL build's live VIEWER-role hashing), the original
  behavior is unchanged: it raises rather than falling back to the hardcoded
  development salt if `PUBLIC_BUILD` were ever set without a configured
  secret.
- `src/dashboard/app.py` adds one more explicit guard
  (`role == ADMIN and not PUBLIC_BUILD`) on the Admin Log navigation entry,
  defense in depth on top of the fact that `role` can never actually equal
  `ADMIN` under the flag.
- `entity_label_for_list()` (`_auth.py`) and the search-matching branch of
  `entity_picker()` (`_shell.py`) both also check `not PUBLIC_BUILD`
  independently of `role`, closing two paths that displayed or matched
  against a real ID directly without going through `reveal_or_anonymize()`.
  Verified by forcing `st.session_state["role"] = "ADMIN"` directly (bypassing
  `authenticate()` entirely) and confirming no real ID surfaces from any of
  these four functions -- see Part 4 verification output.

Every other `role == "ADMIN"` check across the dashboard's screens
(`00_overview.py`, `01b_network_resilience.py`, `02_rq2_governance_risk.py`,
`03_rq3_price_variance.py`) reads the same `st.session_state["role"]` value
set only by `authenticate()`, so none of them needed individual changes:
they are unreachable by construction once the flag is set.

`.dockerignore` keeps the FULL-build image itself honest too: `docker COPY`
does not respect `.gitignore`, so without it, `COPY src/ src/` would bake in
`src/dashboard/_auth_store.json` (password hashes from whoever last ran the
dashboard locally) and `src/dashboard/logs/access.log`. Neither belongs in
any image; a deploy should start from a fresh auth store (self-seeds the
default admin account on first run, see "Rotate ... admin password" above)
or one provided at runtime.

## What a public deploy cannot avoid loading into server memory

The pseudonym is computed live (`hash(anon_salt + real_id)`), not read from
a stored mapping file -- there is no mapping file anywhere in this
codebase to accidentally ship. But the underlying parquet/JSON artifacts
the app reads (`data/results/*`, `data/features/*`) do contain real named
IDs, because that is the source data every screen (public or full build)
reads from. The public-build guarantee is that a real ID is **never
rendered to the client or reachable through any code path**, not that the
server process never touches one in memory. Do not represent this to
anyone as "the public build's data has no real names in it" -- the correct
statement is "the public build never displays or serves a real name."

## Known gap carried forward, not fixed by this task

`data/public/features/rq1_network_features.parquet` retains `buyer_region`/
`supplier_region` unpseudonymized (they're coarse geographic aggregates, not
direct identifiers). A rare region combined with a specific CPV division and
award year could in principle narrow a row down to a small enough set of
real entities to be a residual quasi-identifier risk -- known, and accepted
for the portfolio build; not addressed by this task.

`data/models/*.pkl` and some `data/results/*.parquet` / `*.json` files are
already tracked in git from before the `*.parquet` gitignore rule existed.
They are model/metric artifacts, not raw named contract data, so they are
not the same risk class as the pseudonym-mapping concern above. Left
tracked as-is per your explicit call during this task; if you want them
untracked later, that is a `git rm --cached` pass with its own review, not
bundled into this one.

Also found, not created by this task: a stray `src/dashboard/.streamlit/`
directory (dated before this session) containing its own `config.toml` and
a `secrets.toml` with an unrelated `app_password` key -- leftover from an
earlier session that ran Streamlit with a different working directory.
It's harmless (already covered by the blanket `src/dashboard/` gitignore
rule, and not the path Streamlit actually reads when launched via `make
dashboard` from the repo root, which uses the root-level `.streamlit/` this
task created). Left in place; delete it if you confirm it's unused.
