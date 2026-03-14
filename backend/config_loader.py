"""
Config Loader — Deep Research AI

Design pattern: Module-level Singleton — config is loaded once from YAML on first
access and cached in _config.  All other modules call get() with dot-notation keys.

Token resolution order (first non-empty wins):
  1. HF_TOKEN environment variable
  2. .env file in the project root  (loaded via python-dotenv if available)
  3. huggingface.token in config.yaml  (should be left blank in the repo)
"""
import os
import yaml
from pathlib import Path
from typing import Optional

_config: Optional[dict] = None

# Project root = two levels up from this file (backend/config_loader.py)
_ROOT = Path(__file__).parent.parent


def _load_dotenv() -> None:
    """Load .env from project root if python-dotenv is installed."""
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_file, override=False)  # don't override already-set vars
    except ImportError:
        # python-dotenv not installed — parse manually (key=value, skip comments)
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val


def load_config() -> dict:
    global _config
    if _config is not None:
        return _config

    # Load .env first so HF_TOKEN is available before YAML is read
    _load_dotenv()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        _config = yaml.safe_load(f)

    # Token resolution order: env var → .env (already in env) → YAML
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        _config.setdefault("huggingface", {})["token"] = env_token

    return _config


def get(key_path: str, default=None):
    """Dot-notation config access.  Example: get('models.writer')"""
    cfg = load_config()
    val = cfg
    for key in key_path.split("."):
        if not isinstance(val, dict):
            return default
        val = val.get(key)
        if val is None:
            return default
    return val
