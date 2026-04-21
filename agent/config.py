"""
Central configuration module.

Loads environment variables from .env and exposes typed constants.
All other modules import from here — never os.getenv() directly elsewhere.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _require(var_name: str) -> str:
    """Fail fast if a required env var is missing or empty."""
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{var_name}' is missing. "
            f"Copy .env.example to .env and fill in values."
        )
    return value


def _optional(var_name: str, default: str) -> str:
    return os.getenv(var_name, default).strip() or default


# --- API keys ---
ANTHROPIC_API_KEY: str = _require("ANTHROPIC_API_KEY")
TAVILY_API_KEY: str = _require("TAVILY_API_KEY")

# --- Models ---
PLANNING_MODEL: str = _optional("PLANNING_MODEL", "claude-sonnet-4-5")
EXECUTION_MODEL: str = _optional("EXECUTION_MODEL", "claude-haiku-4-5")
JUDGE_MODEL: str = _optional("JUDGE_MODEL", "claude-sonnet-4-5")

# --- Budget (USD) ---
BUDGET_CAP_PER_QUERY: float = float(_optional("BUDGET_CAP_PER_QUERY", "0.10"))
BUDGET_CAP_TOTAL: float = float(_optional("BUDGET_CAP_TOTAL", "5.00"))

# --- Agent limits ---
MAX_ITERATIONS: int = int(_optional("MAX_ITERATIONS", "10"))
MAX_PARALLEL_TOOLS: int = int(_optional("MAX_PARALLEL_TOOLS", "5"))
TOOL_TIMEOUT_SECONDS: int = int(_optional("TOOL_TIMEOUT_SECONDS", "30"))

# --- Paths ---
PROJECT_ROOT: Path = _PROJECT_ROOT
LOGS_DIR: Path = _PROJECT_ROOT / "logs"
RUNS_DIR: Path = _PROJECT_ROOT / "runs"
DATA_DIR: Path = _PROJECT_ROOT / "data"
CHROMA_DIR: Path = _PROJECT_ROOT / "chroma_db"

# Ensure directories exist
for _dir in (LOGS_DIR, RUNS_DIR, DATA_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --- Model pricing (USD per 1M tokens) ---
# Used by budget tracker. Update if Anthropic changes prices.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5":  {"input": 1.0, "output": 5.0},
    # Fallbacks for older model IDs in case a user overrides
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022":  {"input": 1.0, "output": 5.0},
}


def get_pricing(model: str) -> dict[str, float]:
    """Return {input, output} USD-per-million-tokens for a given model."""
    if model not in MODEL_PRICING:
        # Conservative fallback — assume Sonnet pricing
        return MODEL_PRICING["claude-sonnet-4-5"]
    return MODEL_PRICING[model]