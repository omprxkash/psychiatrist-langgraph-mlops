# Contributing

This is a personal project, but issues and PRs are welcome.

## Setup

```bash
git clone <repo>
cd psychiatrist
pip install -e ".[dev]"
```

## Before opening a PR

```bash
make safety    # 16/16 is a hard gate — every PR must pass this
make test      # full suite
make lint      # ruff + mypy
```

The suicide-risk regression suite (`tests/safety/`) is a release gate. Any PR that lowers recall on that suite will not be merged. See [SAFETY.md](SAFETY.md) for the review process required to add or remove cases.

## Scope

Out-of-scope for this repo: real patient data, clinical deployment, paid LLM APIs. See [SAFETY.md](SAFETY.md) and the "Honest limitations" section of the README.

**Not a medical device.** If you are building on top of this, read [SAFETY.md](SAFETY.md) first.
