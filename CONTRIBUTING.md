# Contributing to PatchCascade SOC

Thank you for your interest in contributing to PatchCascade SOC! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Ayush-Kumar0207/PatchCascade-SOC.git
cd PatchCascade-SOC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio  # For running tests

# Run the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Run tests
python -m pytest tests/ -v

# Run smoke test
python smoke_test.py
```

## 📁 Project Structure

```
PatchCascade-SOC/
├── models.py           # Pydantic data models (enums, schemas)
├── environment.py      # Core RL environment logic
├── server.py           # FastAPI server (OpenEnv-compliant)
├── grader.py           # Multi-dimensional programmatic graders
├── client.py           # HTTP + local clients
├── inference.py        # Baseline LLM agent
├── smoke_test.py       # End-to-end validation script
├── tasks/              # Task definitions (5 levels)
│   ├── __init__.py     # Task registry
│   ├── easy.py
│   ├── medium.py
│   ├── hard.py
│   ├── incident_response.py
│   └── zero_day.py
├── tests/              # Test suite
│   ├── conftest.py     # Shared fixtures
│   ├── test_environment.py
│   ├── test_grader.py
│   ├── test_models.py
│   └── test_server.py
├── openenv.yaml        # OpenEnv configuration
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
├── README.md           # Documentation
├── ARCHITECTURE.md     # Technical deep-dive
└── EXAMPLES.md         # Strategy walkthroughs
```

## 🧪 Testing

We use pytest for testing. All tests should pass before submitting a PR:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_environment.py -v

# Run with coverage
python -m pytest tests/ -v --tb=short

# Run smoke test (end-to-end validation)
python smoke_test.py
```

## 📝 Code Style

- **Python 3.11+** with type annotations everywhere
- **Pydantic v2** for data models with rich `Field(description=...)`
- **Docstrings** on all public functions and classes
- **Constants** in SCREAMING_SNAKE_CASE at module level
- Keep lines under 100 characters where possible

## 🔧 Adding a New Task

1. Create `tasks/new_task.py` with a task dict
2. Create a grader class in `grader.py` inheriting from `TaskGrader`
3. Add the scenario generator in `environment.py`
4. Register in `tasks/__init__.py`
5. Update `openenv.yaml` with task metadata
6. Add tests in `tests/test_environment.py`

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
