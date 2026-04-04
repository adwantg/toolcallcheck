# Contributing to agent-test

Thank you for contributing! We welcome bug reports, feature requests, and pull requests.

## Developer Quick Start

```bash
# Clone the repository
git clone https://github.com/adwantg/agent-test.git
cd agent-test

# Create a virtual environment using Python 3.10+
python3 -m venv .venv
source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quality and Acceptance Gates

To get your PR merged, it must pass these local verification steps:

```bash
# 1. Formatting
ruff format .

# 2. Linting
ruff check .

# 3. Type Checking
mypy src

# 4. Tests and Coverage (minimum 90% required)
python -m pytest

# 5. Security Audit
pip-audit
```

## Pull Request Requirements

By contributing, you agree to:

1. **Maintain Quality**: All code must pass formatting, linting, type checking, and tests.
2. **Keep Coverage High**: Do not reduce test coverage below 90%.
3. **Update README & Docs**: Any behavior change must include an accompanying `README.md` update.
4. **Write Tests**: Every new feature or bug fix must include corresponding tests.
5. **Follow Conventions**: Adhere to the repository layout and coding standards.

## Issue Reporting

When reporting an issue, please describe:
1. Steps to reproduce the behavior.
2. Expected vs actual behavior.
3. Python version and OS.
4. Relevant error messages or logs.
