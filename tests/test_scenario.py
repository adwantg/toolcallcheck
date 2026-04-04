"""Tests for the parametrized scenario runner."""

from __future__ import annotations

from agent_test.scenario import scenario

CASES = [
    {"id": "case_a", "value": 1, "expected": "one"},
    {"id": "case_b", "value": 2, "expected": "two"},
    {"id": "case_c", "value": 3, "expected": "three"},
]


@scenario(CASES)
def test_scenario_parametrize(case):
    """Verify that @scenario parametrizes correctly."""
    assert "id" in case
    assert "value" in case
    assert "expected" in case

    lookup = {1: "one", 2: "two", 3: "three"}
    assert lookup[case["value"]] == case["expected"]


# Test with custom id_key
CUSTOM_CASES = [
    {"name": "alpha", "x": 10},
    {"name": "beta", "x": 20},
]


@scenario(CUSTOM_CASES, id_key="name")
def test_custom_id_key(case):
    assert case["x"] in (10, 20)
