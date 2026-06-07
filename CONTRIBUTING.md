# Contributing

Thank you for improving `robustcov`.

## Development setup

```bash
git clone https://github.com/smiryusupov/robustcov.git
cd robustcov
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"