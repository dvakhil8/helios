"""Tier scorers — one file per tier (correctness, diagnosis, outcome,
behavior, robustness). Each scorer returns a dict[str, Any] that the
harness collects into the run's scores.json."""
