# Contributing to Social Media Comment Judge

Thank you for your interest in contributing to **Social Media Comment Judge**! This guide outlines how to contribute code, report bugs, and propose enhancements.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/social-media-comment-judge.git
   cd social-media-comment-judge
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Verify Ollama runtime:**
   Ensure Ollama is installed and running:
   ```bash
   ollama --version
   ```

---

## 🧪 Running Tests

Run the test suite with `pytest`:
```bash
pytest -v
```

Lint and format checking with `ruff`:
```bash
ruff check .
```

---

## 🌿 Contribution Workflow

1. Fork the repository and create your feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes adhering to existing code conventions (type annotations, docstrings).
3. Add unit tests in `tests/` covering new features or bug fixes.
4. Verify all tests pass locally.
5. Commit your changes with clear, semantic commit messages:
   ```bash
   git commit -m "feat(api): add post context support to batch evaluation"
   ```
6. Push to your branch and submit a Pull Request.

---

## 📜 Pull Request Guidelines

- Ensure CI tests pass.
- Maintain documentation integrity in `README.md` and docstrings.
- Avoid committing large dataset dumps or model weights.
