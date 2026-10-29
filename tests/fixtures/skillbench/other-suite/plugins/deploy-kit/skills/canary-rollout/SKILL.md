---
name: canary-rollout
description: Safe progressive traffic shifting and rollback verification for service deployments.
allowed-tools: view_file, write_to_file, run_command
---

# Canary Rollout

Plan and verify progressive canary traffic shifts with explicit abort thresholds.

## When no one can answer

When running unattended, default to conservative error-budget gates (1% 5xx threshold and 2x p99 latency guard) and document the rollout schedule directly.

## Steps

1. Inspect the routing and deployment configuration in the workspace.
2. Verify health-check invariants and rollback triggers before shifting traffic above 10%.
3. Output a step-by-step traffic shift schedule and verification checklist.
