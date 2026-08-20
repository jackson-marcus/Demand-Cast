# PowerShell task runner mirroring the Makefile.
# Usage: .\tasks.ps1 <task>   e.g. .\tasks.ps1 test
param([Parameter(Mandatory = $true)][string]$Task)

switch ($Task) {
    "install"     { uv sync --group dev }
    "lint"        { uv run ruff check .; uv run ruff format --check . }
    "format"      { uv run ruff check --fix .; uv run ruff format . }
    "test"        { uv run pytest --cov }
    "data"        { uv run python scripts/make_synthetic.py; uv run python -m demandcast.data.prepare }
    "data-m5"     { uv run python -m demandcast.data.download; uv run python -m demandcast.data.prepare }
    "backtest"    { uv run python -m demandcast.models.backtest }
    "refresh"     { uv run python scripts/refresh_forecasts.py }
    "api"         { uv run uvicorn demandcast.api.main:app --reload --port 8020 }
    "ui"          { uv run streamlit run src/demandcast/ui/app.py }
    "mlflow"      { uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002 }
    "docker-up"   { docker compose up --build -d }
    "docker-down" { docker compose down }
    default       { Write-Host "Unknown task: $Task" }
}
