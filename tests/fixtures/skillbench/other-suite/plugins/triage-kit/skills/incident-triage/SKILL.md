---
name: incident-triage
description: Structured incident triage workflow for service degradation and alert spikes.
allowed-tools: view_file, ask_question
---

# Incident Triage

Diagnose production alert spikes systematically before proposing mitigations.

## When no one can answer

If the operator is unavailable or running headlessly, proceed with the safest read-only diagnostic assumptions and state them explicitly in the triage report.

## Steps

1. Inspect the service code and telemetry configuration in the workspace.
2. Clarify scope with `ask_question` if operator context is needed.
3. Produce a structured incident triage report covering blast radius, root trigger, and immediate mitigation.
