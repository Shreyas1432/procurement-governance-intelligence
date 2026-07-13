# Audit Reports

Multi-dimensional audits of Project_Main (security, edge cases, test integrity, dependencies).

## Versioning

Each audit run is stored in its own `vN/` folder so changes can be diffed over time:

- `v1/` - **baseline** (2026-06-02): 21 findings (0 CRITICAL, 6 MAJOR, 14 MINOR, 1 INFO)
- `v2/` - _next run, after edits_

### How to re-audit after edits
1. Apply fixes for `v1` findings.
2. Re-run the audit with the same protocol/scope.
3. Save results as `Audit_report/v2/AUDIT_REPORT_v2.{json,md}`.
4. Compare `v2` vs `v1` to confirm findings are resolved and no regressions appeared.

Finding IDs are namespaced per version (`PM-V1-001`, `PM-V2-001`, etc.). When a `v1`
finding is fixed, reference its ID in the `v2` report (e.g. "PM-V1-004 - RESOLVED").

## Files per version
- `AUDIT_REPORT_vN.json` - machine-readable findings array (strict schema)
- `AUDIT_REPORT_vN.md` - human-readable executive summary + fix priority

> The earlier graphify audit was superseded and moved under `../archive/`.
