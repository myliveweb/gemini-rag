# GEMINI.md

## Project Overview

This is a Python project named "my-gemini-api". Based on the `.env` file, it appears to be intended to interact with the Phoenix API. The project is in a very early stage of development. The main entry point is `main.py`, which currently contains functions for interacting with the Phoenix API. The project also contains an HTML file at `data/html/Leadslab.html` which might be used for data extraction or processing.

**Technologies:**
*   Python (version 3.14+freethreaded, as specified in `.python-version`)
*   `uv` for project and dependency management (inferred from the presence of `pyproject.toml` and modern Python project structure).

## Building and Running

### Setup

1.  **Install `uv`**: If you don't have `uv` installed, follow the official installation instructions.
2.  **Create a virtual environment**:
    ```bash
    uv venv
    ```
3.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```
    *Note: `requirements.txt` does not exist yet. This command is a placeholder for when dependencies are added to `pyproject.toml` and a requirements file is generated.*

### Running the application

To run the main script:
```bash
uv run python main.py
```

## Development Conventions

*   **Dependencies**: Project dependencies are managed using `uv` and are defined in `pyproject.toml`.
*   **Environment Variables**: The project uses a `.env` file for environment variables. It is expected that a `GOOGLE_CLOUD_PROJECT` variable is set.
*   **Code Style**: No specific linter or formatter is configured yet, but following PEP 8 is a good practice for Python projects.
