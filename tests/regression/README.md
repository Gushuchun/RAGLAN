# Regression Tests

Tests that prevent fixed bugs from recurring.

## Naming Convention

```
test_issue_{number}.py   — GitHub issue regression
test_bug_{description}.py — descriptive bug-fix regression
```

## Example

```python
# tests/regression/test_issue_42.py
def test_empty_query_does_not_crash():
    """Regression: GitHub issue #42 — empty query caused IndexError."""
    ...
```

## Adding a Regression Test

1. Create a new file in this directory
2. Name it after the issue or bug
3. Include a docstring referencing the original issue
4. Keep tests minimal — only the specific bug scenario
